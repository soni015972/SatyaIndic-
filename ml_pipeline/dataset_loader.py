"""
Dataset Loader & Stratified Splitting Module
===========================================
Handles multilingual dataset ingestion (CSV, JSON, Hugging Face Datasets),
automatic column mapping for arbitrary user schemas, Devanagari/Hinglish
preprocessing, and Stratified K-Fold / 80-10-10 data splitting.
"""

import os
import json
import logging
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

try:
    from ml_pipeline.preprocess import preprocess_text
except ImportError:
    from preprocess import preprocess_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Column detection candidates
TEXT_COLUMN_CANDIDATES = [
    "text", "news", "content", "statement", "headline", "title",
    "article", "tweet", "message", "claim"
]

LABEL_COLUMN_CANDIDATES = [
    "label", "target", "fake", "class", "veracity", "is_fake",
    "category", "ground_truth", "status"
]

# Normalization mapping for labels to binary 0 (REAL) / 1 (FAKE)
LABEL_MAPPING = {
    "fake": 1, "1": 1, 1: 1, 1.0: 1, True: 1, "true_fake": 1, "false": 1,
    "misleading": 1, "rumor": 1, "unverified": 1, "hoax": 1,
    "real": 0, "0": 0, 0: 0, 0.0: 0, False: 0, "true": 0, "authentic": 0,
    "verified": 0, "genuine": 0
}


class MultilingualFakeNewsDataset(Dataset):
    """
    PyTorch Dataset wrapper for sequence classification tokenization.
    Encodes text on-the-fly or in-memory for Hugging Face Transformer models.
    """
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

        # Token type IDs (for models like BERT that use them)
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return item


def auto_detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Inspects DataFrame columns and matches against standard candidates.
    Raises ValueError if neither candidate is located.
    """
    lower_cols = {col.lower(): col for col in df.columns}

    text_col = None
    for cand in TEXT_COLUMN_CANDIDATES:
        if cand in lower_cols:
            text_col = lower_cols[cand]
            break

    label_col = None
    for cand in LABEL_COLUMN_CANDIDATES:
        if cand in lower_cols:
            label_col = lower_cols[cand]
            break

    if not text_col:
        raise ValueError(
            f"Could not auto-detect text column. Available columns: {list(df.columns)}. "
            f"Expected one of: {TEXT_COLUMN_CANDIDATES}"
        )

    if not label_col:
        raise ValueError(
            f"Could not auto-detect label column. Available columns: {list(df.columns)}. "
            f"Expected one of: {LABEL_COLUMN_CANDIDATES}"
        )

    return text_col, label_col


def normalize_labels(series: pd.Series) -> pd.Series:
    """
    Normalizes arbitrary string, boolean, or numeric labels into binary 0 (REAL) and 1 (FAKE).
    """
    def map_val(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float, bool)):
            return LABEL_MAPPING.get(val, int(val))
        clean_str = str(val).strip().lower()
        if clean_str in LABEL_MAPPING:
            return LABEL_MAPPING[clean_str]
        try:
            int_val = int(clean_str)
            return LABEL_MAPPING.get(int_val, int_val)
        except ValueError:
            pass
        return np.nan

    mapped = series.apply(map_val)
    if mapped.isna().any():
        dropped_count = mapped.isna().sum()
        logger.warning(f"Dropping {dropped_count} rows with unrecognized label values.")
    return mapped


def load_and_preprocess_dataset(
    file_path: Optional[str] = None,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    apply_preprocessing: bool = True
) -> pd.DataFrame:
    """
    Loads dataset from CSV or JSON file, applies Devanagari/Hinglish cleaning,
    detects scripts, and standardizes columns to 'text', 'cleaned_text', 'label', 'script'.
    """
    if not file_path or not os.path.exists(file_path):
        default_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_dataset.csv")
        default_path = os.path.abspath(default_path)
        logger.info(f"No valid user file provided. Loading bundled dataset from: {default_path}")
        file_path = default_path

    logger.info(f"Loading dataset from: {file_path}")
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".json") or file_path.endswith(".jsonl"):
        df = pd.read_json(file_path, lines=file_path.endswith(".jsonl"))
    else:
        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_json(file_path)

    # Auto-detect columns if not explicitly passed
    detected_text, detected_label = auto_detect_columns(df)
    active_text_col = text_col if text_col and text_col in df.columns else detected_text
    active_label_col = label_col if label_col and label_col in df.columns else detected_label

    logger.info(f"Using text column: '{active_text_col}', label column: '{active_label_col}'")

    # Rename to canonical column names
    processed_df = df.copy()
    processed_df["raw_text"] = processed_df[active_text_col].astype(str)
    processed_df["label"] = normalize_labels(processed_df[active_label_col])

    # Drop missing labels or text
    processed_df = processed_df.dropna(subset=["label", "raw_text"])
    processed_df["label"] = processed_df["label"].astype(int)

    # Preprocessing & language classification
    cleaned_texts = []
    scripts = []
    languages = []

    for raw in processed_df["raw_text"]:
        if apply_preprocessing:
            cleaned, meta = preprocess_text(raw)
        else:
            cleaned = raw.strip()
            meta = {"primary_script": "unknown", "detected_lang": "unknown"}

        cleaned_texts.append(cleaned)
        scripts.append(meta.get("primary_script", "unknown"))
        languages.append(meta.get("detected_lang", "unknown"))

    processed_df["text"] = cleaned_texts
    processed_df["script"] = scripts
    processed_df["language"] = languages

    # Remove rows where cleaned text is empty
    processed_df = processed_df[processed_df["text"].str.strip() != ""].reset_index(drop=True)

    logger.info(
        f"Dataset loaded: {len(processed_df)} samples. "
        f"Class Distribution: {dict(processed_df['label'].value_counts())}. "
        f"Scripts: {dict(processed_df['script'].value_counts())}"
    )

    return processed_df


def create_train_val_test_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates stratified 80% Train, 10% Validation, 10% Test partitions.
    Preserves class balance (REAL vs FAKE) in all partitions.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Splits must sum to 1.0"

    # Step 1: Split into Train (80%) and Temp (20%)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=df["label"]
    )

    # Step 2: Split Temp evenly into Val (50% of 20% = 10%) and Test (50% of 20% = 10%)
    relative_val_size = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val_size),
        random_state=random_state,
        stratify=temp_df["label"]
    )

    logger.info(
        f"Splits Created -> Train: {len(train_df)} ({len(train_df)/len(df):.1%}), "
        f"Val: {len(val_df)} ({len(val_df)/len(df):.1%}), "
        f"Test: {len(test_df)} ({len(test_df)/len(df):.1%})"
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_stratified_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Yields (train_df, val_df) folds for Stratified K-Fold cross validation.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        train_fold = df.iloc[train_idx].reset_index(drop=True)
        val_fold = df.iloc[val_idx].reset_index(drop=True)
        folds.append((train_fold, val_fold))
        logger.info(f"Fold {fold_idx + 1}/{n_splits} prepared: {len(train_fold)} train, {len(val_fold)} val.")
    return folds


if __name__ == "__main__":
    df = load_and_preprocess_dataset()
    train_df, val_df, test_df = create_train_val_test_splits(df)
    print("\n[Train Sample]:")
    print(train_df[["text", "language", "script", "label"]].head(3))
