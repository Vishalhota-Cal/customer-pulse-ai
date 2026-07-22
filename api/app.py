"""
API layer -- thin HTTP endpoints only.

No business logic lives here. Every route validates the request shape,
delegates to orchestration/persistence/brain, and formats a JSON
response. If a route starts making a business decision (e.g. deciding
WHAT counts as a valid category), that logic belongs in a lower layer,
not here.
"""

from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from flask import Flask, jsonify, request
from flask_cors import CORS

from brain.classifier import AnthropicClient, OpenAIClient
from brain.pii_redactor import redact_pii
from brain.summary_generator import generate_weekly_summary
from brain.theme_aggregator import OpenAIEmbeddingClient, aggregate_themes
from domain.feedback import FeedbackItem, ThemeTag
from orchestration.pipeline import run_pipeline
from persistence import store as persistence_store

app = Flask(__name__)

# The dashboard is a standalone HTML file opened directly in a browser
# (file:// origin, or a different port than this API) -- without CORS
# enabled, the browser blocks every fetch() call to this API with a
# generic "Failed to fetch" error that gives no indication CORS is the
# cause. This is a real, necessary piece of infrastructure for this
# project's shape (separate static frontend + API), not a security
# loosening -- there's no cookie-based auth here for CORS to expose.
CORS(app)

# Hard cap so a single request can't ask for unbounded work -- cheap to
# add now, expensive to retrofit later once real traffic exists.
MAX_BATCH_SIZE = 200

# --- Lightweight, in-memory rate limiting -----------------------------------
# NOTE (documented limitation, not an oversight): in-memory state resets on
# restart and isn't shared across multiple processes/workers. Fine for a
# single-process demo. A real multi-worker production deployment would need
# a shared store (Redis, etc.) instead. "No login system" was a scope
# decision for this project -- "no protection at all" is not the same thing,
# so this exists even without auth.
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 30
_request_log: dict[str, deque] = defaultdict(deque)


def _rate_limited(ip: str) -> bool:
    now = time.time()
    log = _request_log[ip]
    while log and now - log[0] > RATE_LIMIT_WINDOW_SECONDS:
        log.popleft()
    if len(log) >= RATE_LIMIT_MAX_REQUESTS:
        return True
    log.append(now)
    return False


@app.before_request
def _enforce_rate_limit():
    ip = request.remote_addr or "unknown"
    if _rate_limited(ip):
        return jsonify({"error": "Rate limit exceeded, try again shortly."}), 429


# --- AI client: constructed lazily and cached, swappable for tests ---------
_client = None


def _get_client():
    """
    Construct whichever real AI client the person actually has a key
    for. Checks ANTHROPIC_API_KEY first, then falls back to
    OPENAI_API_KEY -- both implement the same AIClient interface, so
    nothing downstream (classifier, sentiment, etc.) needs to know or
    care which one is in use.
    """
    global _client
    if _client is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            _client = AnthropicClient()
        elif os.environ.get("OPENAI_API_KEY"):
            _client = OpenAIClient()
        else:
            raise RuntimeError(
                "No API key found. Set either ANTHROPIC_API_KEY or "
                "OPENAI_API_KEY in your .env file (see .env.example)."
            )
    return _client


def _set_client_for_testing(client) -> None:
    """
    TEST-ONLY hook. Lets the automated test suite inject a fake client
    instead of constructing a real (paid, non-deterministic) Anthropic
    client. Never called from production code paths.
    """
    global _client
    _client = client


# --- Escalation client (tiered model cost control) --------------------
_escalation_client = None
_escalation_client_checked = False


def _get_escalation_client():
    """
    Construct a STRONGER, pricier model to escalate to on low-confidence
    classifications. Only implemented for OpenAI right now (cheap
    gpt-4o-mini -> stronger gpt-4o) -- there's no verified stronger
    Anthropic model string in this codebase to safely guess at, so when
    only ANTHROPIC_API_KEY is set, this returns None and tiered
    escalation is simply not active. That's an honest, documented
    limitation, not a silent failure: classify_feedback() treats a None
    escalation_client as "don't escalate," identical to before this
    feature existed.
    """
    global _escalation_client, _escalation_client_checked
    if not _escalation_client_checked:
        _escalation_client_checked = True
        if os.environ.get("OPENAI_API_KEY"):
            _escalation_client = OpenAIClient(model="gpt-4o")
    return _escalation_client


def _set_escalation_client_for_testing(client) -> None:
    """TEST-ONLY hook, same reasoning as _set_client_for_testing above."""
    global _escalation_client, _escalation_client_checked
    _escalation_client = client
    _escalation_client_checked = True


# --- Embedding client (semantic theme clustering) ----------------------
_embedding_client = None
_embedding_client_checked = False


def _get_embedding_client():
    """
    Construct an embeddings client for semantic theme clustering. Only
    implemented for OpenAI right now. Returns None if no OPENAI_API_KEY
    is set -- callers treat a None embedding_client as "fall back to
    string-similarity clustering," not an error.
    """
    global _embedding_client, _embedding_client_checked
    if not _embedding_client_checked:
        _embedding_client_checked = True
        if os.environ.get("OPENAI_API_KEY"):
            _embedding_client = OpenAIEmbeddingClient()
            print("[api] Embedding client constructed (OPENAI_API_KEY found) -- semantic theme clustering is ACTIVE")
        else:
            print("[api] No OPENAI_API_KEY found -- semantic theme clustering is OFF, falling back to string similarity")
    return _embedding_client


def _set_embedding_client_for_testing(client) -> None:
    """TEST-ONLY hook, same reasoning as _set_client_for_testing above."""
    global _embedding_client, _embedding_client_checked
    _embedding_client = client
    _embedding_client_checked = True


def _processed_to_json(p) -> dict:
    return {
        "id": p.feedback.id,
        "text": p.feedback.text,
        "source": p.feedback.source,
        "submitted_at": p.feedback.submitted_at.isoformat(),
        "category": p.classification.category.value,
        "confidence": p.classification.confidence,
        "sentiment": p.sentiment.sentiment.value,
        "sentiment_score": p.sentiment.sentiment_score,
        "urgency": p.sentiment.urgency.value,
        "themes": [t.theme for t in p.themes],
        "flagged_for_review": p.flagged_for_review,
        "review_reason": p.review_reason,
        "processing_time_ms": p.processing_time_ms,
    }


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    payload = request.get_json(silent=True)
    if payload is None or not isinstance(payload, list):
        return jsonify({"error": "Expected a JSON array of feedback items."}), 400

    if len(payload) > MAX_BATCH_SIZE:
        return jsonify(
            {"error": f"Batch too large: max {MAX_BATCH_SIZE} items per request, got {len(payload)}."}
        ), 400

    items = []
    pii_redaction_count = 0
    for row in payload:
        if not isinstance(row, dict) or "text" not in row:
            return jsonify({"error": "Each item must be an object with a 'text' field."}), 400

        # PII redaction happens here, before ANYTHING else -- earlier than
        # domain validation, earlier than persistence, earlier than any AI
        # call. This is a regex-based baseline (emails, phone numbers,
        # card-like numbers), not full NLP-grade PII detection -- an
        # honest, documented limitation, not a claim of complete coverage.
        redacted_text, pii_found = redact_pii(row["text"])
        if pii_found:
            pii_redaction_count += 1

        try:
            items.append(
                FeedbackItem(
                    id=str(uuid.uuid4()),
                    text=redacted_text,
                    source=row.get("source", "api"),
                )
            )
        except (TypeError, ValueError) as e:
            # Domain-layer validation (e.g. oversized text) surfaces as a
            # clean 400 here, not a raw stack trace to the client.
            return jsonify({"error": f"Invalid feedback item: {e}"}), 400

    try:
        client = _get_client()
    except RuntimeError as e:
        # Missing/invalid API key: a clear 503, not a crashed process.
        return jsonify({"error": str(e)}), 503

    if pii_redaction_count > 0:
        print(f"[api] Redacted PII from {pii_redaction_count} of {len(items)} incoming item(s)")

    results = run_pipeline(items, client, escalation_client=_get_escalation_client(), save=True)
    return jsonify([_processed_to_json(r) for r in results]), 200


@app.route("/api/feedback", methods=["GET"])
def list_feedback():
    records = persistence_store.load_all()
    return jsonify([_processed_to_json(r) for r in records]), 200


@app.route("/api/summary/weekly", methods=["GET"])
def weekly_summary():
    # Accepts ?days=N (default 7) to scope the summary to real recent
    # data by actual submitted_at timestamp, rather than summarizing
    # every record ever stored regardless of age. This is what makes
    # "weekly summary" an honest label instead of "all-time summary."
    try:
        days = int(request.args.get("days", 7))
    except ValueError:
        return jsonify({"error": "days must be an integer"}), 400

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    all_records = persistence_store.load_all()
    records = [r for r in all_records if r.feedback.submitted_at >= cutoff]

    try:
        client = _get_client()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    summary = generate_weekly_summary(records, client, embedding_client=_get_embedding_client())
    return jsonify({"summary": summary, "based_on_items": len(records)}), 200


@app.route("/api/themes/cluster", methods=["POST"])
def cluster_themes():
    """
    Accepts {"themes": ["theme string", "theme string", ...]} -- a flat
    list with duplicates preserved (one entry per mention, not pre-
    counted) -- and returns real clustered results using the same
    aggregate_themes() logic that powers the weekly summary's theme
    input. This exists so the dashboard's Top Themes display uses ACTUAL
    clustering instead of naive exact-string-match counting in
    JavaScript, which is what it did before this endpoint existed.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or "themes" not in payload or not isinstance(payload["themes"], list):
        return jsonify({"error": "Expected {\"themes\": [list of strings]}"}), 400

    theme_tags = [
        ThemeTag(feedback_id=str(i), theme=str(t))
        for i, t in enumerate(payload["themes"])
        if str(t).strip()
    ]
    aggregated = aggregate_themes(theme_tags, embedding_client=_get_embedding_client())
    return jsonify([
        {"label": a.label, "count": a.count} for a in aggregated
    ]), 200


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/api/data", methods=["DELETE"])
def clear_all_data():
    """
    Wipe all stored feedback. This is a destructive, deliberate action --
    intended for demo/dev resets (e.g. clearing legacy test records saved
    before a new field was added), not something the pipeline or any
    automated process ever calls on its own. The dashboard only exposes
    this behind an explicit confirmation dialog.
    """
    persistence_store.clear()
    return jsonify({"status": "cleared"}), 200


@app.route("/api/themes/status", methods=["GET"])
def get_theme_statuses():
    return jsonify(persistence_store.load_theme_statuses()), 200


@app.route("/api/themes/status", methods=["POST"])
def set_theme_status():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or "theme" not in payload or "status" not in payload:
        return jsonify({"error": "Expected {\"theme\": str, \"status\": str}"}), 400

    theme = str(payload["theme"]).strip()
    status = str(payload["status"]).strip()
    if not theme:
        return jsonify({"error": "theme cannot be blank"}), 400

    try:
        updated = persistence_store.set_theme_status(theme, status)
    except ValueError as e:
        # Unrecognized status (not Open/In Progress/Resolved) -- clean 400.
        return jsonify({"error": str(e)}), 400

    return jsonify(updated), 200


if __name__ == "__main__":
    app.run(debug=False, port=5000)