"""
Explainability & Token Attribution Engine
=========================================
Computes token importance scores to explain model predictions:
- Occlusion Sensitivity / Leave-One-Out Perturbation
- Attention-weight extraction
- Gradient saliency attribution
Highlights which tokens drive classification toward FAKE vs REAL.
"""

import re
from typing import List, Dict, Any, Callable, Optional
import numpy as np


# Linguistic cues typically correlated with misinformation / phishing in Indic/Hinglish
MISINFO_PATTERNS = {
    # Hindi cues
    "सावधान": 0.85, "खतरनाक": 0.80, "फ्री": 0.75, "चमत्कारी": 0.90, "दावा": 0.65,
    "क्लिक": 0.75, "शेयर": 0.70, "वायरल": 0.70, "छुपाते": 0.80, "100%": 0.85,
    # Hinglish cues
    "forward": 0.80, "fwd": 0.85, "bhai": 0.50, "jaldi": 0.75, "free": 0.75,
    "scam": 0.80, "radiation": 0.75, "theek": 0.60, "police": 0.65, "arrest": 0.70,
    "immediately": 0.80, "magic": 0.85, "secret": 0.80, "cure": 0.75,
    # Authentic / Credible cues (push toward REAL)
    "who": -0.85, "isro": -0.90, "rbi": -0.85, "ministry": -0.80, "guideline": -0.75,
    "official": -0.80, "clarified": -0.75, "supreme": -0.85, "court": -0.85,
    "विश्व": -0.70, "संगठन": -0.70, "सरकार": -0.65, "स्पष्ट": -0.75, "आदेश": -0.70
}


def explain_prediction_perturbation(
    text: str,
    predict_fn: Callable[[List[str]], np.ndarray],
    max_tokens: int = 40
) -> List[Dict[str, Any]]:
    """
    Computes token attribution via Occlusion Sensitivity (Leave-One-Out).
    Evaluates: Importance_i = P_base(FAKE) - P_masked(FAKE).
    """
    words = text.split()
    if len(words) > max_tokens:
        words = words[:max_tokens]

    if not words:
        return []

    # Get baseline probability for full sentence
    base_probs = predict_fn([text])[0]  # [prob_real, prob_fake]
    base_fake_prob = float(base_probs[1])

    # Generate leave-one-out variations
    perturbed_texts = []
    for i in range(len(words)):
        masked_words = words[:i] + words[i+1:]
        perturbed_texts.append(" ".join(masked_words) if masked_words else "[PAD]")

    perturbed_probs = predict_fn(perturbed_texts)  # Shape: (N, 2)
    fake_probs_perturbed = perturbed_probs[:, 1]

    # Delta: positive means removing the word decreases fake probability (word was suspicious)
    raw_importances = base_fake_prob - fake_probs_perturbed

    # Normalize scores between -1.0 and 1.0
    max_abs = np.max(np.abs(raw_importances)) if len(raw_importances) > 0 else 0
    if max_abs > 1e-5:
        norm_scores = raw_importances / max_abs
    else:
        norm_scores = raw_importances

    tokens_explained = []
    for word, score in zip(words, norm_scores):
        tokens_explained.append({
            "token": word,
            "importance": round(float(score), 4),
            "label_driver": "FAKE" if score > 0.05 else ("REAL" if score < -0.05 else "NEUTRAL")
        })

    return tokens_explained


def explain_heuristic_fallback(text: str, base_label: str) -> List[Dict[str, Any]]:
    """
    Fast heuristic attribution when model inference function is simulated or offline.
    Uses vocabulary pattern weights scaled by base prediction.
    """
    words = text.split()
    tokens_explained = []

    for word in words:
        clean_word = re.sub(r"[^\w\s]", "", word.lower())
        weight = 0.0

        for pattern, pat_weight in MISINFO_PATTERNS.items():
            if pattern in clean_word or clean_word in pattern:
                weight = pat_weight
                break

        if weight == 0.0:
            # Baseline small noise
            weight = 0.02 if base_label == "FAKE" else -0.02

        tokens_explained.append({
            "token": word,
            "importance": round(weight, 3),
            "label_driver": "FAKE" if weight > 0.1 else ("REAL" if weight < -0.1 else "NEUTRAL")
        })

    return tokens_explained
