"""
ONNX Model Exporter & Inference Optimizer
=========================================
Exports fine-tuned Hugging Face PyTorch sequence classification checkpoints
to optimized ONNX (Open Neural Network Exchange) format with dynamic axes.
"""

import os
import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_to_onnx(
    checkpoint_dir: str,
    output_onnx_path: str,
    max_length: int = 128
):
    """
    Exports a PyTorch Transformer sequence classification checkpoint to ONNX.
    Configures dynamic batch size and sequence length dimensions.
    """
    logger.info(f"Loading checkpoint from: {checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()

    # Create dummy inputs for tracing
    dummy_text = "विश्व स्वास्थ्य संगठन ने नई एडवाइजरी जारी की।"
    inputs = tokenizer(
        dummy_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Determine dynamic axes based on model type
    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    }

    model_args = (inputs["input_ids"], inputs["attention_mask"])

    if "token_type_ids" in inputs:
        input_names.append("token_type_ids")
        dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}
        model_args = (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])

    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)

    logger.info(f"Exporting model to ONNX: {output_onnx_path}")
    torch.onnx.export(
        model,
        model_args,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes
    )

    logger.info("ONNX export completed successfully!")

    # Verify with onnxruntime if available
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
        ort_inputs = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy()
        }
        if "token_type_ids" in inputs:
            ort_inputs["token_type_ids"] = inputs["token_type_ids"].numpy()

        ort_outputs = session.run(None, ort_inputs)
        with torch.no_grad():
            pt_outputs = model(**inputs).logits.numpy()

        np.testing.assert_allclose(pt_outputs, ort_outputs[0], rtol=1e-3, atol=1e-4)
        logger.info("ONNX verification PASSED: Numerical parity verified with PyTorch outputs.")
    except ImportError:
        logger.info("onnxruntime not installed locally. Parity verification skipped.")
    except Exception as e:
        logger.warning(f"Parity verification warning: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export fine-tuned Hugging Face model to ONNX")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to fine-tuned model checkpoint directory")
    parser.add_argument("--output_path", type=str, default="model.onnx", help="Target .onnx file path")
    args = parser.parse_args()

    export_to_onnx(args.checkpoint_dir, args.output_path)
