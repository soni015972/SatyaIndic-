"""
Transformer Fine-Tuning Pipeline
================================
Supports training IndicBERT, MuRIL, XLM-RoBERTa, and English BERT on
Multilingual and Code-Switched Fake News sequence classification.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, Optional

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)

try:
    from ml_pipeline.dataset_loader import (
        load_and_preprocess_dataset,
        create_train_val_test_splits,
        get_stratified_kfold_splits,
        MultilingualFakeNewsDataset
    )
    from ml_pipeline.evaluate import compute_metrics, evaluate_model_comprehensive, save_benchmark_summary
except ImportError:
    from dataset_loader import (
        load_and_preprocess_dataset,
        create_train_val_test_splits,
        get_stratified_kfold_splits,
        MultilingualFakeNewsDataset
    )
    from evaluate import compute_metrics, evaluate_model_comprehensive, save_benchmark_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), "models_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_single_model(
    model_key: str,
    train_df,
    val_df,
    test_df,
    output_base_dir: str = "./checkpoints",
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    lr_override: Optional[float] = None
) -> Dict[str, Any]:
    """
    Fine-tunes a single Transformer model and evaluates on the hold-out test set.
    """
    config = load_config()
    model_specs = config["models"][model_key]
    hf_model_name = model_specs["hf_name"]
    hyperparams = model_specs["hyperparameters"]

    epochs = epochs_override or hyperparams["num_train_epochs"]
    batch_size = batch_size_override or hyperparams["batch_size"]
    learning_rate = lr_override or hyperparams["learning_rate"]
    max_length = hyperparams["max_length"]

    model_dir = os.path.join(output_base_dir, model_key)
    os.makedirs(model_dir, exist_ok=True)

    logger.info(f"=== Initializing Training for: {model_specs['display_name']} ({hf_model_name}) ===")
    logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}, Max Len: {max_length}")

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_name,
        num_labels=2,
        id2label={0: "REAL", 1: "FAKE"},
        label2id={"REAL": 0, "FAKE": 1}
    )

    # Wrap Datasets
    train_dataset = MultilingualFakeNewsDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    val_dataset = MultilingualFakeNewsDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    test_dataset = MultilingualFakeNewsDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    # Determine hardware acceleration
    use_fp16 = torch.cuda.is_available() and config["training_defaults"]["fp16"]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Training Device: {device.upper()} | Mixed Precision FP16: {use_fp16}")

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=hyperparams["weight_decay"],
        warmup_ratio=hyperparams["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=use_fp16,
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config["training_defaults"]["early_stopping_patience"])]
    )

    # Train
    logger.info(f"Starting training loop for {model_key}...")
    trainer.train()

    # Save best checkpoint
    best_checkpoint_dir = os.path.join(model_dir, "best_model")
    trainer.save_model(best_checkpoint_dir)
    tokenizer.save_pretrained(best_checkpoint_dir)
    logger.info(f"Best model & tokenizer saved to: {best_checkpoint_dir}")

    # Final Evaluation on Hold-Out Test Set
    logger.info(f"Evaluating {model_key} on hold-out Test Set ({len(test_df)} samples)...")
    test_predictions = trainer.predict(test_dataset)
    y_preds = np.argmax(test_predictions.predictions, axis=1).tolist()
    y_trues = test_df["label"].tolist()

    report = evaluate_model_comprehensive(
        y_true=y_trues,
        y_pred=y_preds,
        languages=test_df["language"].tolist(),
        scripts=test_df["script"].tolist(),
        model_name=model_specs["display_name"]
    )
    report["model_key"] = model_key
    report["display_name"] = model_specs["display_name"]
    report["architecture"] = model_specs["architecture"]
    report["params_millions"] = model_specs["params_millions"]

    logger.info(
        f"Test Results for {model_specs['display_name']} -> "
        f"Accuracy: {report['accuracy']:.4f}, Macro-F1: {report['macro_f1']:.4f}"
    )

    return report


def main():
    parser = argparse.ArgumentParser(description="Multilingual Fake News Detector Training Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="muril",
        choices=["indic-bert", "muril", "xlm-roberta", "bert-base-uncased", "all"],
        help="Model architecture to train, or 'all' to train all 4 architectures sequentially."
    )
    parser.add_argument("--data_path", type=str, default=None, help="Path to user CSV/JSON dataset file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save benchmark reports.")

    args = parser.parse_args()

    # Load and preprocess dataset
    df = load_and_preprocess_dataset(args.data_path)
    train_df, val_df, test_df = create_train_val_test_splits(df)

    os.makedirs(args.results_dir, exist_ok=True)
    config = load_config()

    models_to_train = list(config["models"].keys()) if args.model == "all" else [args.model]
    all_reports = {}

    for model_key in models_to_train:
        report = train_single_model(
            model_key=model_key,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            output_base_dir=args.output_dir,
            epochs_override=args.epochs,
            batch_size_override=args.batch_size,
            lr_override=args.lr
        )
        all_reports[model_key] = report

    # Export benchmark summary
    json_out = os.path.join(args.results_dir, "benchmark_results.json")
    csv_out = os.path.join(args.results_dir, "benchmark_results.csv")
    save_benchmark_summary(all_reports, json_out, csv_out)
    logger.info("Training and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
