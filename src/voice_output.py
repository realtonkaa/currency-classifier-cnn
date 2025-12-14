"""Text-to-speech voice output for accessibility."""

from typing import Dict

# Try to import pyttsx3, but don't fail if not available
try:
    import pyttsx3
    TTS_AVAILABLE = True
except (ImportError, RuntimeError):
    TTS_AVAILABLE = False


# Full currency name lookup
CURRENCY_FULL_NAMES = {
    "USD": "US Dollar",
    "EUR": "Euro",
    "GBP": "British Pound",
    "INR": "Indian Rupee",
    "JPY": "Japanese Yen",
    "CAD": "Canadian Dollar",
    "AUD": "Australian Dollar",
    "CHF": "Swiss Franc",
    "CNY": "Chinese Yuan",
    "BRL": "Brazilian Real",
    "MXN": "Mexican Peso",
    "KRW": "South Korean Won",
}


def format_voice_output(result: Dict) -> str:
    """Format prediction result as a spoken sentence.

    Args:
        result: Prediction dict with currency_code, denomination, confidence

    Returns:
        Human-readable sentence for text-to-speech
    """
    currency_name = CURRENCY_FULL_NAMES.get(
        result.get("currency_code", ""),
        result.get("currency_name", result.get("currency_code", "Unknown"))
    )
    denomination = result.get("denomination", "unknown")
    confidence = result.get("confidence", 0)
    confidence_pct = int(confidence * 100)

    if confidence_pct >= 80:
        return f"This appears to be a {denomination} {currency_name}. Confidence: {confidence_pct} percent."
    elif confidence_pct >= 50:
        return f"This might be a {denomination} {currency_name}, but I'm not very sure. Confidence: {confidence_pct} percent."
    else:
        return f"I'm not sure what this is. Best guess: {denomination} {currency_name}. Confidence: only {confidence_pct} percent. Try taking a clearer photo."


def speak_result(result: Dict, rate: int = 150) -> bool:
    """Speak the prediction result using text-to-speech.

    Args:
        result: Prediction dict
        rate: Speech rate (words per minute)

    Returns:
        True if speech was successful, False if TTS not available
    """
    if not TTS_AVAILABLE:
        print("Text-to-speech not available. Install pyttsx3: pip install pyttsx3")
        return False

    text = format_voice_output(result)

    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.say(text)
    engine.runAndWait()
    return True
