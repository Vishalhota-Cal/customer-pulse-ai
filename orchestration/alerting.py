"""
Orchestration layer: webhook alerting for urgent feedback.

Config-driven and OFF by default: if no webhook URL is configured, this
module never attempts any network call at all. This is deliberate --
alerting should never fire anywhere the person didn't explicitly wire it
up themselves. Safe to ship even without a real webhook URL configured.

Compatible with Slack's incoming-webhook format ({"text": "..."}), which
is also accepted by many other tools (Discord via a compatibility mode,
Microsoft Teams connectors, or a custom internal endpoint that expects
the same shape).
"""

from __future__ import annotations

import os

import requests

WEBHOOK_TIMEOUT_SECONDS = 5


def get_webhook_url() -> str | None:
    """Reads ALERT_WEBHOOK_URL from the environment. None if not configured."""
    return os.environ.get("ALERT_WEBHOOK_URL") or None


def send_urgent_alert(feedback_id: str, category: str, text: str, webhook_url: str | None = None) -> bool:
    """
    Send a Slack-compatible webhook notification for a high-urgency item.

    Never raises -- a failed or misconfigured webhook should never take
    down feedback processing over a notification that couldn't be
    delivered. Returns True/False so callers CAN log the outcome, but
    aren't required to check it.

    webhook_url defaults to reading from the environment if not passed
    explicitly (mainly so tests can inject a fake URL without needing to
    mutate the real environment).
    """
    url = webhook_url if webhook_url is not None else get_webhook_url()
    if not url:
        return False  # alerting not configured -- silently a no-op, not an error

    preview = text[:180] + ("..." if len(text) > 180 else "")
    payload = {
        "text": (
            f":rotating_light: *High-urgency feedback received*\n"
            f"*Category:* {category}\n"
            f"*Feedback ID:* {feedback_id}\n"
            f"*Preview:* {preview}"
        )
    }

    try:
        response = requests.post(url, json=payload, timeout=WEBHOOK_TIMEOUT_SECONDS)
        return response.ok
    except requests.RequestException as e:
        print(f"[alerting] Failed to send webhook alert for feedback_id={feedback_id}: {e}")
        return False