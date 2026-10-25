"""
Multilingual & Code-Switched Preprocessing Pipeline
===================================================
Specialized for Indic (Devanagari script), Hinglish (Romanized Hindi),
and English sequence classification.

Key Preprocessing Tasks:
1. Unicode Normalization (NFC / NFKD) & Nukta reconciliation.
2. Zero-Width character stripping (ZWJ, ZWNJ, formatting markers).
3. Punctuation standardization (Devanagari Purna Viram '|' vs Latin '.').
4. Code-Switched / Romanized text cleaning (elongated character reduction, slang).
5. Regex entity masking ([URL], [EMAIL], [PHONE], [USER], [HASHTAG]).
6. Script & Language classification heuristic (Devanagari, Roman, Mixed).
"""

import re
import unicodedata
from typing import Dict, Any, Tuple


# Regex patterns for entity masking
URL_PATTERN = re.compile(
    r"(?:https?://|www\.)[a-zA-Z0-9.\-_~:/?#\[\]@!$&'()*+,;=%]+",
    re.IGNORECASE
)
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE
)
PHONE_PATTERN = re.compile(
    r"(?:\+91[\-\s]?)?[6-9]\d{9}|\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b"
)
MENTION_PATTERN = re.compile(r"@[\w_]+")
HASHTAG_PATTERN = re.compile(r"#[\w_]+")

# Zero-Width Characters and invisible control chars
ZERO_WIDTH_CHARS = re.compile(r"[\u200B\u200C\u200D\u200E\u200F\uFEFF]")

# Elongated characters (e.g., "sachhhhh" -> "sach", "plzzz" -> "plz")
ELONGATED_PATTERN = re.compile(r"(.)\1{2,}")

# Common Hinglish colloquial slang & phonetic mapping
HINGLISH_SLANG_MAP = {
    "plz": "please",
    "pls": "please",
    "bcz": "because",
    "bcoz": "because",
    "kewl": "cool",
    "ur": "your",
    "r": "are",
    "u": "you",
    "thx": "thanks",
    "omg": "oh my god",
    "bt": "but",
    "wid": "with",
    "msg": "message",
    "fwd": "forward",
    "frnd": "friend",
    "frnds": "friends",
    "govt": "government",
    "gormint": "government",
    "crore": "crore",
    "lakh": "lakh",
}

# Devanagari Unicode Range: U+0900 to U+097F
DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
LATIN_RANGE = re.compile(r"[a-zA-Z]")


def detect_script_distribution(text: str) -> Dict[str, Any]:
    """
    Computes character distribution across Devanagari, Latin, and other scripts.
    Identifies if text is Pure Hindi (Devanagari), Pure English, or Code-Switched Hinglish.
    """
    if not text:
        return {"devanagari_ratio": 0.0, "latin_ratio": 0.0, "primary_script": "unknown"}

    total_alpha = len([c for c in text if c.isalpha()])
    if total_alpha == 0:
        return {"devanagari_ratio": 0.0, "latin_ratio": 0.0, "primary_script": "unknown"}

    devanagari_count = len(DEVANAGARI_RANGE.findall(text))
    latin_count = len(LATIN_RANGE.findall(text))

    dev_ratio = devanagari_count / total_alpha
    lat_ratio = latin_count / total_alpha

    if dev_ratio >= 0.65:
        primary_script = "devanagari"
        detected_lang = "hindi"
    elif lat_ratio >= 0.65:
        # Check if Romanized Hindi or English using Hinglish keywords heuristic
        hinglish_keywords = {
            "hai", "hain", "ko", "se", "me", "mein", "par", "ke", "ki", "ka",
            "karein", "karo", "nahi", "raha", "rahe", "rahi", "hoga", "hogi",
            "karo", "bhai", "aaj", "kal", "jaldi", "yeh", "woh", "kuch", "sabhi"
        }
        tokens = set(re.findall(r"\b[a-z]{2,}\b", text.lower()))
        hinglish_overlap = len(tokens.intersection(hinglish_keywords))
        
        if hinglish_overlap >= 2:
            primary_script = "romanized_hinglish"
            detected_lang = "hinglish"
        else:
            primary_script = "latin"
            detected_lang = "english"
    else:
        primary_script = "code_mixed"
        detected_lang = "code_switched"

    return {
        "devanagari_ratio": round(dev_ratio, 3),
        "latin_ratio": round(lat_ratio, 3),
        "primary_script": primary_script,
        "detected_lang": detected_lang
    }


def clean_devanagari(text: str) -> str:
    """
    Normalizes Devanagari Unicode representations:
    1. NFC Unicode canonical composition.
    2. Zero-Width Non-Joiner (ZWNJ) and Joiner (ZWJ) stripping.
    3. Nukta unification and Purna Viram normalization.
    """
    # Canonical NFC composition
    text = unicodedata.normalize("NFC", text)
    # Remove invisible zero-width characters that disrupt tokenizer subwords
    text = ZERO_WIDTH_CHARS.sub("", text)
    # Standardize Devanagari Danda (| or ॥) to single sentence terminator
    text = re.sub(r"[\u0964\u0965]+", " । ", text)
    return text


def clean_hinglish_and_latin(text: str) -> str:
    """
    Cleans Romanized / Hinglish text:
    1. Reduces elongated characters ("plzzzz" -> "plz", "sachhhhh" -> "sach").
    2. Replaces common social media abbreviations with standardized tokens.
    """
    # Reduce 3+ consecutive duplicate characters to 2 (preserves legitimate doubles like 'look', 'bheed')
    text = ELONGATED_PATTERN.sub(r"\1\1", text)

    # Word-level slang normalization
    words = text.split()
    normalized_words = [
        HINGLISH_SLANG_MAP.get(w.lower(), w) for w in words
    ]
    return " ".join(normalized_words)


def mask_entities(text: str, mask_urls: bool = True, mask_phones: bool = True,
                  mask_handles: bool = True) -> str:
    """
    Replaces sensitive / dynamic entities with fixed semantic tokens.
    Prevents models from overfitting to specific URLs, phone numbers, or usernames.
    """
    if mask_urls:
        text = URL_PATTERN.sub("[URL]", text)
    if mask_phones:
        text = PHONE_PATTERN.sub("[PHONE]", text)
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    if mask_handles:
        text = MENTION_PATTERN.sub("[USER]", text)
        text = HASHTAG_PATTERN.sub("[HASHTAG]", text)
    return text


def preprocess_text(
    text: str,
    mask_entities_flag: bool = True,
    normalize_unicode: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """
    Complete end-to-end preprocessing pipeline for multilingual & code-switched inputs.
    
    Args:
        text: Raw input string.
        mask_entities_flag: Whether to replace URLs, phones, and handles with masks.
        normalize_unicode: Whether to apply NFC normalization.
        
    Returns:
        cleaned_text: Preprocessed text ready for Transformer tokenization.
        metadata: Language detection stats and script breakdown.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.strip()
    if not text:
        return "", {"devanagari_ratio": 0.0, "latin_ratio": 0.0, "detected_lang": "unknown"}

    # Script and language detection
    meta = detect_script_distribution(text)

    # Entity masking
    if mask_entities_flag:
        text = mask_entities(text)

    # Script-specific cleaning
    if meta["devanagari_ratio"] > 0:
        text = clean_devanagari(text)

    if meta["latin_ratio"] > 0 or meta["primary_script"] in ("romanized_hinglish", "code_mixed"):
        text = clean_hinglish_and_latin(text)

    # Strip excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text, meta


if __name__ == "__main__":
    test_samples = [
        "विश्व स्वास्थ्य संगठन (WHO) ने कोरोना वायरस के लिए गाइडलाइन जारी की https://who.int/news 🚨",
        "Bhaiii plzzzz ye message sabhi groups me fwd kardo 5G radiation se birds mar rahi hain 9876543210 pe call mat karo!",
        "NASA confirmed James Webb detected water vapor on K2-18b exoplanet. Visit https://nasa.gov",
        "Ek chammach haldi se cancer theek ho jata hai... sabhi frnds ko share karo!"
    ]

    print("=== Testing Multilingual Preprocessing Pipeline ===")
    for sample in test_samples:
        cleaned, meta = preprocess_text(sample)
        print(f"\n[Original] : {sample}")
        print(f"[Cleaned]  : {cleaned}")
        print(f"[Metadata] : {meta}")
