"""Tests for orchestration/alerting.py."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from orchestration.alerting import send_urgent_alert


def test_no_webhook_configured_is_a_silent_no_op(monkeypatch):
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    result = send_urgent_alert("fb1", "Bug / Technical Issue", "some feedback text")
    assert result is False


def test_unreachable_url_fails_gracefully_without_raising():
    result = send_urgent_alert("fb2", "Bug / Technical Issue", "test", webhook_url="http://localhost:1/nonexistent")
    assert result is False


@pytest.fixture
def mock_webhook_server():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            received.append(json.loads(self.rfile.read(length)))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, received
    server.shutdown()


def test_successful_webhook_sends_correctly_formed_payload(mock_webhook_server):
    server, received = mock_webhook_server
    url = f"http://127.0.0.1:{server.server_port}/webhook"

    result = send_urgent_alert("fb3", "Billing / Payments", "duplicate charge on my account", webhook_url=url)

    assert result is True
    assert len(received) == 1
    assert "fb3" in received[0]["text"]
    assert "Billing / Payments" in received[0]["text"]
    assert "duplicate charge" in received[0]["text"]