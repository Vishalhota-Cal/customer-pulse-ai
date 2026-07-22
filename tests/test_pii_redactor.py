"""Tests for brain/pii_redactor.py."""

from brain.pii_redactor import redact_pii


def test_email_redacted():
    text, found = redact_pii("Contact me at jane.doe@example.com please.")
    assert "[EMAIL_REDACTED]" in text
    assert "jane.doe" not in text
    assert found == ["email"]


def test_phone_redacted():
    text, found = redact_pii("Call me at (555) 123-4567.")
    assert "[PHONE_REDACTED]" in text
    assert "phone" in found


def test_card_number_redacted_without_swallowing_trailing_space():
    text, found = redact_pii("My card 4111 1111 1111 1111 was declined.")
    assert "[CARD_REDACTED] was declined" in text
    assert "card_number" in found


def test_multiple_pii_types_in_one_message():
    text, found = redact_pii("Email test@test.com or call 555-987-6543.")
    assert "email" in found and "phone" in found


def test_no_false_positive_on_ordinary_short_numbers():
    original = "I gave it 5 stars and waited 3 days for a reply."
    text, found = redact_pii(original)
    assert found == []
    assert text == original


def test_non_string_input_does_not_crash():
    text, found = redact_pii(None)
    assert found == []