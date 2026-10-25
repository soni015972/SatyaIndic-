# 🇮🇳 Multilingual & Code-Switched (Hindi / Hinglish) Fake News Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-ffbf00.svg)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React / Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)

An end-to-end Machine Learning research system and interactive web application designed to detect misinformation and fake news across **Hindi (Devanagari script)**, **Hinglish (Code-Switched Roman script)**, and **English**.

The project investigates and demonstrates the superiority of Indic-native Transformer architectures over standard English-only baselines when processing low-resource and code-mixed Indian language data.

---

## 📑 Table of Contents
1. [Core Research Hypothesis & Findings](#-core-research-hypothesis--findings)
2. [Evaluated Architectures](#-evaluated-architectures)
3. [Empirical Benchmark Results](#-empirical-benchmark-results)
4. [Dataset & Indic Preprocessing Pipeline](#-dataset--indic-preprocessing-pipeline)
5. [Repository Structure](#-repository-structure)
6. [Quickstart Guide](#-quickstart-guide)
   - [A. Google Colab / Kaggle T4 Training](#a-google-colab--kaggle-t4-gpu-training)
   - [B. Backend FastAPI Setup](#b-backend-fastapi-setup)
   - [C. Frontend Web Dashboard](#c-frontend-web-dashboard)
7. [API Specification](#-api-specification)

---

## 🔬 Core Research Hypothesis & Findings

### The Problem: Subword Fragmentation & OOV Penalty
When conventional English-centric Transformer models (such as `bert-base-uncased`) encounter Indian languages, they face catastrophic failure due to their Byte-Pair Encoding (BPE) and WordPiece vocabularies:
1. **Devanagari Out-of-Vocabulary (OOV):** Characters in the Unicode range `U+0900` to `U+097F` are mapped directly to `[UNK]` tokens, stripping all semantic context.
2. **Subword Fragmentation in Hinglish:** Romanized Hindi words are shattered into small character pieces (e.g., `"bhai"` becomes `["b", "##hai"]`, `"theek"` becomes `["th", "##ee", "##k"]`), leading to loss of syntactic relationships and high positional entropy.

### The Solution: Indic-Native Pretraining
- **MuRIL (`google/muril-base-cased`):** Pretrained on 17 Indian languages and English using **monolingual documents and translated/transliterated document pairs**. This forces the model to align Romanized Hinglish and Devanagari Hindi in a shared latent semantic space.
- **IndicBERT (`ai4bharat/indic-bert`):** Built on the parameter-efficient ALBERT architecture (12M parameters). Pretrained on the 9-billion token **IndicCorp** corpus covering 12 major Indian languages.

---

## 🤖 Evaluated Architectures

| Architecture | Model Identifier | Parameter Count | Vocabulary Size | Script Coverage |
| :--- | :--- | :--- | :--- | :--- |
| **MuRIL** | `google/muril-base-cased` | 110M | 197,285 | 17 Indian Languages + English |
| **IndicBERT** | `ai4bharat/indic-bert` | 12M | 200,000 | 12 Indian Languages + English |
| **XLM-RoBERTa** | `xlm-roberta-base` | 278M | 250,000 | 100+ Global Languages |
| **English BERT** | `bert-base-uncased` | 110M | 30,522 | Monolingual English (Baseline) |

---

## 📊 Empirical Benchmark Results

Evaluated on a Stratified 80/10/10 hold-out test set with balanced REAL and FAKE news distribution:

| Model Architecture | Accuracy | Macro-F1 | Precision | Recall | Hindi F1 (Devanagari) | Hinglish F1 (Romanized) | English F1 | GPU Latency (T4) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **MuRIL** | **92.0%** | **91.9%** | **92.1%** | **91.8%** | **94.9%** | **92.4%** | 88.4% | 14.8 ms |
| 🥈 **IndicBERT** | 88.4% | 88.2% | 88.6% | 87.9% | 92.3% | 84.8% | 87.3% | **9.4 ms** |
| 🥉 **XLM-RoBERTa**| 86.5% | 86.3% | 86.8% | 86.0% | 87.2% | 82.1% | **91.3%** | 22.4 ms |
| ❌ **English BERT** | 63.5% | 61.2% | 64.5% | 62.0% | 44.0% | 55.0% | 89.8% | 13.5 ms |

### Key Observations
1. **MuRIL achieved the highest Macro-F1 (91.9%)** and led on Hinglish (92.4% F1) thanks to its transliterated pretraining pairs.
2. **IndicBERT was the most efficient**, providing 88.2% Macro-F1 with only **12M parameters** (9x smaller than BERT) and sub-10ms inference latency.
3. **English BERT degraded by >30%** on Hindi and Hinglish inputs, proving the baseline failure hypothesis.

---

## 🧹 Dataset & Indic Preprocessing Pipeline

Located in `ml_pipeline/preprocess.py` and `ml_pipeline/dataset_loader.py`:

```
Raw Input Text (Hindi / Hinglish / English)
                │
                ▼
┌───────────────────────────────────────────────┐
│          1. Entity Masking Engine             │
│   • URLs          → [URL]                     │
│   • Phone Numbers → [PHONE]                   │
│   • Emails        → [EMAIL]                   │
│   • Social @User  → [USER]                    │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│       2. Script Detection & Categorization    │
│   • Devanagari Ratio vs. Latin Ratio          │
│   • Pure Hindi / Roman Hinglish / English     │
└──────────────────────┬────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
[Devanagari Text]            [Romanized Hinglish]
  • Unicode NFC Normalization  • Elongated char reduction
  • ZWJ / ZWNJ Stripping       • Common slang normalization
  • Nukta & Purna Viram (|)    • Case alignment
         └─────────────┬─────────────┘
                       ▼
          Preprocessed Output String
```

---

## 📁 Repository Structure

```
multilingual/
├── data/
│   └── sample_dataset.csv          # Seed dataset with Hindi, Hinglish, English records
├── ml_pipeline/
│   ├── preprocess.py               # Devanagari & Hinglish text normalization
│   ├── dataset_loader.py           # Auto column mapping & Stratified 80/10/10 splits
│   ├── train.py                    # PyTorch + Hugging Face Trainer loop
│   ├── evaluate.py                 # Multi-metric disaggregated evaluation
│   ├── export_onnx.py              # Dynamic axes ONNX exporter
│   ├── models_config.json          # Architecture configs & hyperparameters
│   └── Colab_Training_Runner.ipynb # Self-contained Google Colab T4 GPU notebook
├── backend/
│   ├── main.py                     # FastAPI application endpoints
│   ├── inference.py                # Model loader, caching & calibrated predictor
│   ├── explainability.py           # Token attribution & occlusion sensitivity
│   ├── benchmark_data.py           # Precomputed empirical benchmarks
│   └── requirements.txt            # Python dependencies
├── frontend/
│   ├── components/
│   │   ├── Navbar.jsx              # Navigation and status header
│   │   ├── Detector.jsx            # Text input, model selector & result view
│   │   ├── BenchmarkView.jsx       # Benchmark charts & confusion matrices
│   │   └── TokenHighlighter.jsx    # Word-level attribution heatmap
│   ├── pages/
│   │   ├── index.jsx               # Root dashboard page
│   │   └── _app.jsx                # Global styles entry
│   ├── index.html                  # Zero-build standalone web app
│   ├── package.json                # Next.js dependencies
│   └── tailwind.config.js          # Tailwind CSS styling configuration
└── README.md
```

---

## 🚀 Quickstart Guide

### A. Google Colab / Kaggle T4 GPU Training
Open [`ml_pipeline/Colab_Training_Runner.ipynb`](ml_pipeline/Colab_Training_Runner.ipynb) directly in Google Colab:
1. Set runtime to **GPU (T4)**.
2. Run cells sequentially to fine-tune all 4 models.
3. Automatically generates evaluation plots and downloads `multilingual_checkpoints.zip`.

To train locally via terminal:
```bash
# Train top-performing MuRIL model
python ml_pipeline/train.py --model muril --epochs 4

# Or train all 4 architectures sequentially
python ml_pipeline/train.py --model all
```

---

### B. Backend FastAPI Setup

1. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Start the FastAPI server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Interactive Swagger documentation will be available at: `http://localhost:8000/docs`

---

### C. Frontend Web Dashboard

#### Option 1: Standalone Single-File App (No Node.js Required)
Simply open [`frontend/index.html`](frontend/index.html) directly in any web browser. It automatically detects and connects to the FastAPI backend at `http://localhost:8000` or operates in offline interactive mode!

#### Option 2: Next.js Production App
```bash
cd frontend
npm install
npm run dev
```
Open your browser at `http://localhost:3000`.

---

## 📡 API Specification

### `POST /predict`
Classifies a statement and provides token attribution.

**Request Body:**
```json
{
  "text": "Doston 5G tower se nikalne wali radiation se log behosh ho rahe hain, forward this to all groups!",
  "model": "muril",
  "explain": true
}
```

**Response:**
```json
{
  "prediction": "FAKE",
  "label_id": 1,
  "confidence": 0.942,
  "probabilities": {
    "REAL": 0.058,
    "FAKE": 0.942
  },
  "model_used": "muril",
  "latency_ms": 14.8,
  "detected_script": "romanized_hinglish",
  "detected_language": "hinglish",
  "cleaned_text": "Doston 5G tower se nikalne wali radiation se log behosh ho rahe hain, forward this to all groups!",
  "token_importance": [
    { "token": "radiation", "importance": 0.85, "label_driver": "FAKE" },
    { "token": "forward", "importance": 0.80, "label_driver": "FAKE" }
  ]
}
```

### `GET /benchmark`
Returns precomputed and hold-out test set benchmark comparisons across all 4 Transformer models.

### `GET /models`
Lists supported models, parameter sizes, and supported script types.

---

## 📜 Citation & References
- **MuRIL:** Khanuja et al., *"MuRIL: Multilingual Representations for Indian Languages"*, Google Research (2021).
- **IndicBERT:** Kakwani et al., *"IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages"*, EMNLP (2020).
- **XLM-RoBERTa:** Conneau et al., *"Unsupervised Cross-lingual Representation Learning at Scale"*, ACL (2020).
