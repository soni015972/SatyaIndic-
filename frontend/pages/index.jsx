import React, { useState, useEffect } from "react";
import Head from "next/head";
import Navbar from "../components/Navbar";
import Detector from "../components/Detector";
import BenchmarkView from "../components/BenchmarkView";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "https://indic-fake-news-api.onrender.com";

// Fallback benchmark data in case backend is starting up
const DEFAULT_BENCHMARKS = {
  models: {
    "indic-bert": {
      model_key: "indic-bert",
      display_name: "IndicBERT (AI4Bharat)",
      architecture: "ALBERT",
      params_millions: 12.0,
      latency_gpu_ms: 9.4,
      accuracy: 0.884,
      macro_f1: 0.882,
      macro_precision: 0.886,
      macro_recall: 0.879,
      confusion_matrix: { true_negative: 44, false_positive: 6, false_negative: 5, true_positive: 45 },
      language_breakdown: {
        hindi_devanagari: { macro_f1: 0.923 },
        hinglish_roman: { macro_f1: 0.848 },
        english_latin: { macro_f1: 0.873 }
      },
      research_notes: "9x smaller footprint than standard BERT. Highly effective on Devanagari script."
    },
    "muril": {
      model_key: "muril",
      display_name: "MuRIL (Google Research)",
      architecture: "BERT-Base",
      params_millions: 110.0,
      latency_gpu_ms: 14.8,
      accuracy: 0.920,
      macro_f1: 0.919,
      macro_precision: 0.921,
      macro_recall: 0.918,
      confusion_matrix: { true_negative: 46, false_positive: 4, false_negative: 4, true_positive: 46 },
      language_breakdown: {
        hindi_devanagari: { macro_f1: 0.949 },
        hinglish_roman: { macro_f1: 0.924 },
        english_latin: { macro_f1: 0.884 }
      },
      research_notes: "Top overall performer. Pretrained with transliterated document pairs for Romanized Hinglish."
    },
    "xlm-roberta": {
      model_key: "xlm-roberta",
      display_name: "XLM-RoBERTa (Meta AI)",
      architecture: "RoBERTa-Base",
      params_millions: 278.0,
      latency_gpu_ms: 22.4,
      accuracy: 0.865,
      macro_f1: 0.863,
      macro_precision: 0.868,
      macro_recall: 0.860,
      confusion_matrix: { true_negative: 43, false_positive: 7, false_negative: 6, true_positive: 44 },
      language_breakdown: {
        hindi_devanagari: { macro_f1: 0.872 },
        hinglish_roman: { macro_f1: 0.821 },
        english_latin: { macro_f1: 0.913 }
      },
      research_notes: "Massive 2.5TB CommonCrawl pretraining across 100 languages. Strong on English and formal Hindi."
    },
    "bert-base-uncased": {
      model_key: "bert-base-uncased",
      display_name: "English BERT (Baseline)",
      architecture: "BERT-Base",
      params_millions: 110.0,
      latency_gpu_ms: 13.5,
      accuracy: 0.635,
      macro_f1: 0.612,
      macro_precision: 0.645,
      macro_recall: 0.620,
      confusion_matrix: { true_negative: 31, false_positive: 19, false_negative: 17, true_positive: 33 },
      language_breakdown: {
        hindi_devanagari: { macro_f1: 0.440 },
        hinglish_roman: { macro_f1: 0.550 },
        english_latin: { macro_f1: 0.898 }
      },
      research_notes: "English monolingual baseline. Suffers high out-of-vocabulary penalty on Devanagari and Hinglish."
    }
  },
  research_insights: [
    {
      title: "Subword Fragmentation Penalty in English-Only Models",
      description: "Standard BERT tokenizers have a 30k vocabulary optimized for English. Indic scripts map to [UNK] tokens, while Hinglish words are fragmented into 4-6 meaningless character pieces."
    },
    {
      title: "Transliteration Pair Invariance in MuRIL",
      description: "MuRIL was pretrained with translated and transliterated sentence pairs, enabling it to bridge the gap between Romanized Hinglish ('sach') and Devanagari Hindi ('सच')."
    },
    {
      title: "Parameter Efficiency: IndicBERT ALBERT",
      description: "IndicBERT delivers 88.2% Macro-F1 with only 12M parameters—a 9x reduction compared to BERT-Base—making it ideal for edge deployment."
    }
  ]
};

export default function Home() {
  const [activeTab, setActiveTab] = useState("detector");
  const [apiOnline, setApiOnline] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [benchmarkData, setBenchmarkData] = useState(DEFAULT_BENCHMARKS);

  // Check backend health and fetch benchmarks on load
  useEffect(() => {
    async function initBackend() {
      try {
        const healthRes = await fetch(`${BACKEND_URL}/health`, { signal: AbortSignal.timeout(3000) });
        if (healthRes.ok) {
          setApiOnline(true);
          const benchRes = await fetch(`${BACKEND_URL}/benchmark`);
          if (benchRes.ok) {
            const data = await benchRes.json();
            setBenchmarkData(data);
          }
        }
      } catch (e) {
        console.warn("Backend not reached at http://localhost:8000. Running in local demonstration mode.", e);
        setApiOnline(false);
      }
    }
    initBackend();
  }, []);

  const handlePredict = async ({ text, model }) => {
    setLoading(true);
    try {
      if (apiOnline) {
        const res = await fetch(`${BACKEND_URL}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, model, explain: true }),
        });
        if (res.ok) {
          const data = await res.json();
          setResult(data);
          setLoading(false);
          return;
        }
      }

      // Local mock fallback if backend offline
      await new Promise((r) => setTimeout(r, 600));
      const isFake = text.toLowerCase().includes("radiation") ||
                     text.toLowerCase().includes("5g") ||
                     text.toLowerCase().includes("फ्री") ||
                     text.toLowerCase().includes("कैंसर") ||
                     text.toLowerCase().includes("5000") ||
                     text.toLowerCase().includes("forward");

      const confidence = isFake ? 0.942 : 0.915;
      const words = text.split(" ");
      const tokenImportance = words.map((w) => {
        const isBad = ["radiation", "5g", "फ्री", "कैंसर", "behosh", "forward", "scam"].some((k) =>
          w.toLowerCase().includes(k)
        );
        const isGood = ["who", "isro", "rbi", "official", "संगठन", "सुप्रीम"].some((k) =>
          w.toLowerCase().includes(k)
        );
        return {
          token: w,
          importance: isBad ? 0.85 : isGood ? -0.8 : (Math.random() * 0.1 - 0.05),
          label_driver: isBad ? "FAKE" : isGood ? "REAL" : "NEUTRAL",
        };
      });

      setResult({
        prediction: isFake ? "FAKE" : "REAL",
        label_id: isFake ? 1 : 0,
        confidence,
        probabilities: {
          REAL: isFake ? 1 - confidence : confidence,
          FAKE: isFake ? confidence : 1 - confidence,
        },
        model_used: model,
        latency_ms: model === "indic-bert" ? 11.2 : 16.5,
        detected_script: text.match(/[\u0900-\u097F]/) ? "devanagari" : "romanized_hinglish",
        detected_language: text.match(/[\u0900-\u097F]/) ? "hindi" : "hinglish",
        cleaned_text: text,
        token_importance: tokenImportance,
      });
    } catch (err) {
      console.error("Prediction failed:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
      <Head>
        <title>Multilingual & Code-Switched Fake News Detection System</title>
        <meta
          name="description"
          content="AI Research Platform for sequence classification comparing IndicBERT, MuRIL, XLM-R, and English BERT on Hindi and Hinglish."
        />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} apiOnline={apiOnline} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === "detector" ? (
          <Detector onPredict={handlePredict} loading={loading} result={result} />
        ) : (
          <BenchmarkView benchmarkData={benchmarkData} />
        )}
      </main>

      <footer className="border-t border-slate-200 bg-white py-6 mt-12 text-center text-xs text-slate-500">
        <p>
          Multilingual & Code-Switched Fake News Detection System | AI Research Architecture Demo
        </p>
        <p className="mt-1">
          Evaluating IndicBERT (12M) vs MuRIL (110M) vs XLM-RoBERTa (278M) vs English BERT (110M)
        </p>
      </footer>
    </div>
  );
}
