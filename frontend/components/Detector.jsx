import React, { useState } from "react";
import { Send, Sparkles, AlertTriangle, CheckCircle2, Cpu, Globe, Clock, Layers } from "lucide-react";
import TokenHighlighter from "./TokenHighlighter";

const PRESET_EXAMPLES = [
  {
    label: "Hinglish WhatsApp Rumor",
    lang: "Hinglish",
    text: "Doston 5G tower se nikalne wali radiation se birds mar rahi hain aur log behosh ho rahe hain, forward this to all your WhatsApp groups immediately!",
    model: "muril"
  },
  {
    label: "Hindi Fake Health Claim",
    lang: "Hindi",
    text: "नींबू और बेकिंग सोडा का गर्म पानी में मिश्रण पीने से कैंसर और कोरोना 100% जड़ से खत्म हो जाता है, डॉक्टर इसे छुपाते हैं।",
    model: "indic-bert"
  },
  {
    label: "Hindi Authentic Press Release",
    lang: "Hindi",
    text: "भारतीय अंतरिक्ष अनुसंधान संगठन (ISRO) ने श्रीहरिकोटा से अपने नए मौसम उपग्रह INSAT-3DS का सफल प्रक्षेपण किया।",
    model: "indic-bert"
  },
  {
    label: "Hinglish Financial News (Real)",
    lang: "Hinglish",
    text: "RBI ne monetary policy meeting me repo rate ko 6.5% par unchanged rakhne ka faisla kiya hai. Inflation target achieve hone tak sthiti bani rahegi.",
    model: "muril"
  },
  {
    label: "English Phishing Scam",
    lang: "English",
    text: "Government offering free smart laptops to all students who forward this link to at least 15 WhatsApp groups before midnight.",
    model: "bert-base-uncased"
  }
];

export default function Detector({ onPredict, loading, result }) {
  const [inputText, setInputText] = useState(PRESET_EXAMPLES[0].text);
  const [selectedModel, setSelectedModel] = useState("muril");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;
    onPredict({ text: inputText, model: selectedModel });
  };

  const handleApplyPreset = (preset) => {
    setInputText(preset.text);
    setSelectedModel(preset.model);
  };

  return (
    <div className="space-y-6">
      {/* Intro Header */}
      <div className="bg-gradient-to-r from-indigo-900 via-indigo-800 to-slate-900 rounded-2xl p-6 sm:p-8 text-white shadow-xl">
        <div className="max-w-3xl">
          <span className="text-xs font-semibold tracking-wider uppercase text-indigo-300 bg-indigo-950/60 px-3 py-1 rounded-full border border-indigo-700/50">
            Transformer Sequence Classification
          </span>
          <h2 className="text-2xl sm:text-3xl font-extrabold mt-3 tracking-tight">
            Verify News Credibility in Hindi, Hinglish & English
          </h2>
          <p className="text-indigo-200 mt-2 text-sm sm:text-base leading-relaxed">
            Test and contrast Indic-native Transformers (<strong>MuRIL</strong>, <strong>IndicBERT</strong>) against
            multilingual (<strong>XLM-R</strong>) and English baselines. Detects viral misinformation, health myths, and code-mixed scams.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Input Form */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
            {/* Preset Example Chips */}
            <div className="mb-4">
              <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center space-x-1.5">
                <Sparkles className="w-3.5 h-3.5 text-amber-500" />
                <span>Quick Test Queries (Presets)</span>
              </label>
              <div className="flex flex-wrap gap-2">
                {PRESET_EXAMPLES.map((ex, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => handleApplyPreset(ex)}
                    className="text-xs px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-indigo-50 hover:text-indigo-700 text-slate-700 border border-slate-200 transition font-medium text-left"
                  >
                    <span className="font-semibold text-indigo-600 mr-1">[{ex.lang}]</span>
                    {ex.label}
                  </button>
                ))}
              </div>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-semibold text-slate-800 mb-1.5">
                  Input Statement / Social Media Text
                </label>
                <textarea
                  rows={4}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Type or paste Hindi (Devanagari), Hinglish (Roman script), or English statement here..."
                  className="w-full p-3.5 rounded-xl border border-slate-300 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 text-slate-900 text-sm sm:text-base transition outline-none resize-y"
                  required
                />
              </div>

              {/* Model Selection Dropdown */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-600 mb-1.5 flex items-center space-x-1.5">
                    <Cpu className="w-3.5 h-3.5 text-indigo-600" />
                    <span>Transformer Architecture</span>
                  </label>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full p-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm font-medium text-slate-800 focus:border-indigo-500 outline-none"
                  >
                    <option value="muril">MuRIL (google/muril-base-cased) [Recommended]</option>
                    <option value="indic-bert">IndicBERT (ai4bharat/indic-bert) [Lightweight 12M]</option>
                    <option value="xlm-roberta">XLM-RoBERTa (xlm-roberta-base) [278M]</option>
                    <option value="bert-base-uncased">English BERT (bert-base-uncased) [Baseline]</option>
                  </select>
                </div>

                <div className="flex items-end">
                  <button
                    type="submit"
                    disabled={loading || !inputText.trim()}
                    className="w-full py-2.5 px-4 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white font-semibold rounded-xl text-sm shadow-md shadow-indigo-100 transition flex items-center justify-center space-x-2"
                  >
                    {loading ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        <span>Classifying...</span>
                      </>
                    ) : (
                      <>
                        <Send className="w-4 h-4" />
                        <span>Analyze Credibility</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>

        {/* Right Column: Prediction Results & Explainability */}
        <div className="lg:col-span-5 space-y-4">
          {result ? (
            <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-5">
              {/* Prediction Result Badge */}
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                  Prediction Result
                </span>
                <div className="flex items-center space-x-2">
                  <span className="flex items-center text-xs text-slate-500 bg-slate-100 px-2.5 py-1 rounded-md">
                    <Clock className="w-3 h-3 mr-1 text-slate-400" />
                    {result.latency_ms} ms
                  </span>
                </div>
              </div>

              <div
                className={`p-4 rounded-xl border flex items-center space-x-4 ${
                  result.prediction === "FAKE"
                    ? "bg-red-50/70 border-red-200 text-red-900"
                    : "bg-emerald-50/70 border-emerald-200 text-emerald-900"
                }`}
              >
                <div
                  className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 ${
                    result.prediction === "FAKE"
                      ? "bg-red-600 text-white shadow-lg shadow-red-100"
                      : "bg-emerald-600 text-white shadow-lg shadow-emerald-100"
                  }`}
                >
                  {result.prediction === "FAKE" ? (
                    <AlertTriangle className="w-7 h-7" />
                  ) : (
                    <CheckCircle2 className="w-7 h-7" />
                  )}
                </div>
                <div>
                  <div className="text-2xl font-extrabold tracking-tight">
                    {result.prediction === "FAKE" ? "MISINFORMATION / FAKE" : "AUTHENTIC / REAL"}
                  </div>
                  <div className="text-xs font-semibold opacity-80 mt-0.5">
                    Confidence: {(result.confidence * 100).toFixed(1)}% | Model: {result.model_used}
                  </div>
                </div>
              </div>

              {/* Confidence Progress Meter */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-semibold text-slate-600">
                  <span>REAL Probability: {((result.probabilities?.REAL || 0) * 100).toFixed(1)}%</span>
                  <span>FAKE Probability: {((result.probabilities?.FAKE || 0) * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-3 bg-slate-100 rounded-full overflow-hidden flex">
                  <div
                    style={{ width: `${(result.probabilities?.REAL || 0) * 100}%` }}
                    className="h-full bg-emerald-500 transition-all duration-500"
                  />
                  <div
                    style={{ width: `${(result.probabilities?.FAKE || 0) * 100}%` }}
                    className="h-full bg-red-500 transition-all duration-500"
                  />
                </div>
              </div>

              {/* Language and Script Metadata */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                  <span className="text-slate-400 block font-medium">Detected Script</span>
                  <span className="font-bold text-slate-700 capitalize">
                    {result.detected_script}
                  </span>
                </div>
                <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                  <span className="text-slate-400 block font-medium">Detected Language</span>
                  <span className="font-bold text-slate-700 capitalize">
                    {result.detected_language}
                  </span>
                </div>
              </div>

              {/* Token Attribution Explainability */}
              <div className="pt-2 border-t border-slate-100">
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-2 flex items-center space-x-1.5">
                  <Layers className="w-3.5 h-3.5 text-indigo-600" />
                  <span>Explainability: Token Importance Attribution</span>
                </label>
                <TokenHighlighter tokens={result.token_importance} />
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-2xl p-10 border border-slate-200 shadow-sm text-center flex flex-col items-center justify-center min-h-[360px] text-slate-400">
              <Globe className="w-12 h-12 text-slate-300 mb-3 animate-pulse" />
              <p className="font-semibold text-slate-700">Awaiting Input</p>
              <p className="text-xs max-w-xs text-slate-500 mt-1">
                Select a preset query or enter Hindi/Hinglish text and click "Analyze Credibility" to inspect predictions and token attribution.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
