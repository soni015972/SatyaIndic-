"""
Evaluation & Benchmarking Metrics Module
=========================================
Computes comprehensive sequence classification metrics:
- Overall: Accuracy, Macro-F1, Weighted-F1, Precision, Recall, Confusion Matrix
- Disaggregated / Script-Wise Breakdown:
  * Hindi (Devanagari script)
  * Hinglish (Romanized / Code-Switched script)
  * English (Latin script)
- Latency profiling and model parameter accounting
"""

import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Standard evaluation metric callback for Hugging Face Trainer.
    eval_pred is a tuple of (predictions, labels).
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall)
    }


def evaluate_model_comprehensive(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[List[float]] = None,
    languages: Optional[List[str]] = None,
    scripts: Optional[List[str]] = None,
    model_name: str = "model"
) -> Dict[str, Any]:
    """
    Generates full multi-faceted benchmark report with disaggregated script metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Class-wise metrics (0=REAL, 1=FAKE)
    prec_classes, rec_classes, f1_classes, supports = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Confusion matrix [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # Script-wise breakdown
    script_breakdown = {}
    if scripts is not None and len(scripts) == len(y_true):
        df_script = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "script": scripts})
        for script_val in df_script["script"].unique():
            sub = df_script[df_script["script"] == script_val]
            if len(sub) > 0:
                s_acc = accuracy_score(sub["y_true"], sub["y_pred"])
                _, _, s_f1, _ = precision_recall_fscore_support(
                    sub["y_true"], sub["y_pred"], average="macro", zero_division=0
                )
                script_breakdown[str(script_val)] = {
                    "count": int(len(sub)),
                    "accuracy": round(float(s_acc), 4),
                    "macro_f1": round(float(s_f1), 4)
                }

    # Language-wise breakdown
    lang_breakdown = {}
    if languages is not None and len(languages) == len(y_true):
        df_lang = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "language": languages})
        for lang_val in df_lang["language"].unique():
            sub = df_lang[df_lang["language"] == lang_val]
            if len(sub) > 0:
                l_acc = accuracy_score(sub["y_true"], sub["y_pred"])
                _, _, l_f1, _ = precision_recall_fscore_support(
                    sub["y_true"], sub["y_pred"], average="macro", zero_division=0
                )
                lang_breakdown[str(lang_val)] = {
                    "count": int(len(sub)),
                    "accuracy": round(float(l_acc), 4),
                    "macro_f1": round(float(l_f1), 4)
                }

    report = {
        "model_name": model_name,
        "sample_count": int(len(y_true)),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_macro), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "class_metrics": {
            "REAL (0)": {
                "precision": round(float(prec_classes[0]), 4) if len(prec_classes) > 0 else 0.0,
                "recall": round(float(rec_classes[0]), 4) if len(rec_classes) > 0 else 0.0,
                "f1_score": round(float(f1_classes[0]), 4) if len(f1_classes) > 0 else 0.0,
                "support": int(supports[0]) if len(supports) > 0 else 0
            },
            "FAKE (1)": {
                "precision": round(float(prec_classes[1]), 4) if len(prec_classes) > 1 else 0.0,
                "recall": round(float(rec_classes[1]), 4) if len(rec_classes) > 1 else 0.0,
                "f1_score": round(float(f1_classes[1]), 4) if len(f1_classes) > 1 else 0.0,
                "support": int(supports[1]) if len(supports) > 1 else 0
            }
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "matrix": cm.tolist()
        },
        "script_breakdown": script_breakdown,
        "language_breakdown": lang_breakdown
    }

    return report


def save_benchmark_summary(
    all_model_reports: Dict[str, Dict[str, Any]],
    output_json_path: str,
    output_csv_path: str
):
    """
    Saves cross-model comparison results to structured JSON and tabular CSV.
    """
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_model_reports, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved benchmark summary JSON to: {output_json_path}")

    # Build tabular rows
    rows = []
    for model_key, r in all_model_reports.items():
        row = {
            "model_key": model_key,
            "display_name": r.get("display_name", model_key),
            "accuracy": r.get("accuracy", 0.0),
            "macro_f1": r.get("macro_f1", 0.0),
            "macro_precision": r.get("macro_precision", 0.0),
            "macro_recall": r.get("macro_recall", 0.0),
            "weighted_f1": r.get("weighted_f1", 0.0),
            "hindi_macro_f1": r.get("language_breakdown", {}).get("hindi", {}).get("macro_f1", 0.0),
            "hinglish_macro_f1": r.get("language_breakdown", {}).get("hinglish", {}).get("macro_f1", 0.0),
            "english_macro_f1": r.get("language_breakdown", {}).get("english", {}).get("macro_f1", 0.0),
            "latency_ms": r.get("latency_ms", 0.0)
        }
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv_path, index=False)
    logger.info(f"Saved benchmark summary CSV to: {output_csv_path}")
