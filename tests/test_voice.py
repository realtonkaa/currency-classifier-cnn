"""Tests for the voice output module."""

import pytest
from src.voice_output import format_voice_output, CURRENCY_FULL_NAMES


def test_format_high_confidence():
    result = {"currency_code": "USD", "denomination": "20", "confidence": 0.95}
    text = format_voice_output(result)
    assert "20" in text
    assert "US Dollar" in text
    assert "95 percent" in text
    assert "appears to be" in text


def test_format_medium_confidence():
    result = {"currency_code": "EUR", "denomination": "50", "confidence": 0.65}
    text = format_voice_output(result)
    assert "50" in text
    assert "Euro" in text
    assert "not very sure" in text


def test_format_low_confidence():
    result = {"currency_code": "GBP", "denomination": "10", "confidence": 0.30}
    text = format_voice_output(result)
    assert "not sure" in text
    assert "clearer photo" in text


def test_format_unknown_currency():
    result = {"currency_code": "XYZ", "currency_name": "XYZ", "denomination": "100", "confidence": 0.90}
    text = format_voice_output(result)
    assert "100" in text
    assert "XYZ" in text


def test_currency_names_populated():
    assert len(CURRENCY_FULL_NAMES) > 5
    assert "USD" in CURRENCY_FULL_NAMES
    assert "EUR" in CURRENCY_FULL_NAMES
