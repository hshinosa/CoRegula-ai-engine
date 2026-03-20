import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize student text for better NLP analysis.
    - Remove extra whitespaces
    - Normalize unicode characters
    - Remove special characters but keep basic punctuation
    - Lowercase (optional, depending on use case, here we keep case for some analysis)
    """
    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Basic cleaning but keep punctuation that might be HOT indicators (?, !), etc.
    # Keep alphanumeric, spaces, and . , ? ! ( ) -
    text = re.sub(r"[^\w\s\.,\?!\\(\\)\-]", "", text)

    # Remove extra whitespaces and strip at the end
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_for_ttr(text: str) -> str:
    """Specific cleaning for Type-Token Ratio calculation."""
    # Lowercase and remove all punctuation
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text
