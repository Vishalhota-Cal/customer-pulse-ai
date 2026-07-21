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

from flask import Flask, jsonify, request
from flask_cors import CORS

from brain.classifier import AnthropicClient, OpenAIClient
from brain.summary_generator import generate_weekly_summary
from domain.feedback import FeedbackItem
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


def _processed_to_json(p) -> dict:
    return {
        "id": p.feedback.id,
        "text": p.feedback.text,
        "source": p.feedback.source,
        "category": p.classification.category.value,
        "confidence": p.classification.confidence,
        "sentiment": p.sentiment.sentiment.value,
        "sentiment_score": p.sentiment.sentiment_score,
        "urgency": p.sentiment.urgency.value,
        "themes": [t.theme for t in p.themes],
        "flagged_for_review": p.flagged_for_review,
        "review_reason": p.review_reason,
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
    for row in payload:
        if not isinstance(row, dict) or "text" not in row:
            return jsonify({"error": "Each item must be an object with a 'text' field."}), 400
        try:
            items.append(
                FeedbackItem(
                    id=str(uuid.uuid4()),
                    text=row["text"],
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

    results = run_pipeline(items, client, save=True)
    return jsonify([_processed_to_json(r) for r in results]), 200


@app.route("/api/feedback", methods=["GET"])
def list_feedback():
    records = persistence_store.load_all()
    return jsonify([_processed_to_json(r) for r in records]), 200


@app.route("/api/summary/weekly", methods=["GET"])
def weekly_summary():
    records = persistence_store.load_all()
    try:
        client = _get_client()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    summary = generate_weekly_summary(records, client)
    return jsonify({"summary": summary, "based_on_items": len(records)}), 200


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=False, port=5000)