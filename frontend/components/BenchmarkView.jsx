import React, { useState } from "react";
import { Award, Zap, AlertCircle, CheckCircle, BarChart2, BookOpen, Layers } from "lucide-react";

export default function BenchmarkView({ benchmarkData }) {
  const [selectedConfusionModel, setSelectedConfusionModel] = useState("muril");

  const models = benchmarkData?.models || {};
  const insights = benchmarkData?.research_insights || [];
  const modelKeys = Object.keys(models);

  const activeModelData = models[selectedConfusionModel] || models["muril"] || {};
  const cm = activeModelData.confusion_matrix || { true_negative: 0, false_positive: 0, false_negative: 0, true_positive: 0 };

  return (
    <div className="space-y-8">
      {/* Top Research Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="bg-white p-6 rounded-2xl border border-indigo-100 shadow-sm relative overflow-hidden">
          <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-50 rounded-bl-full -z-0" />
          <div className="flex items-center space-x-2 text-indigo-700 font-bold text-xs uppercase tracking-wider mb-2">
            <Award className="w-4 h-4" />
            <span>Top Empirical Accuracy</span>
          </div>
          <div className="text-3xl font-extrabold text-slate-900 tracking-tight">MuRIL</div>
          <div className="text-sm font-semibold text-indigo-600 mt-1">91.9% Macro-F1 | 92.0% Acc</div>
          <p className="text-xs text-slate-500 mt-2">
            Transliterated document pre-training provides robust resilience against spelling variations in Romanized Hinglish.
          </p>
        </div>

        <div className="bg-white p-6 rounded-2xl border border-emerald-100 shadow-sm relative overflow-hidden">
          <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-50 rounded-bl-full -z-0" />
          <div className="flex items-center space-x-2 text-emerald-700 font-bold text-xs uppercase tracking-wider mb-2">
            <Zap className="w-4 h-4" />
            <span>Best Efficiency / Edge Deploy</span>
          </div>
          <div className="text-3xl font-extrabold text-slate-900 tracking-tight">IndicBERT</div>
          <div className="text-sm font-semibold text-emerald-600 mt-1">12M Params | 9.4 ms Latency</div>
          <p className="text-xs text-slate-500 mt-2">
            9x smaller footprint than standard BERT with cross-layer parameter sharing; delivers 88.2% Macro-F1.
          </p>
        </div>

        <div className="bg-white p-6 rounded-2xl border border-amber-100 shadow-sm relative overflow-hidden">
          <div className="absolute top-0 right-0 w-24 h-24 bg-amber-50 rounded-bl-full -z-0" />
          <div className="flex items-center space-x-2 text-amber-700 font-bold text-xs uppercase tracking-wider mb-2">
            <AlertCircle className="w-4 h-4" />
            <span>English Baseline Deficit</span>
          </div>
          <div className="text-3xl font-extrabold text-slate-900 tracking-tight">BERT (English)</div>
          <div className="text-sm font-semibold text-rose-600 mt-1">-30.7% F1 Drop on Indic</div>
          <p className="text-xs text-slate-500 mt-2">
            Suffers from severe out-of-vocabulary penalty and subword fragmentation on Devanagari and Hinglish.
          </p>
        </div>
      </div>

      {/* Model Benchmark Table */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-6 border-b border-slate-200 flex flex-col sm:flex-row justify-between sm:items-center gap-2">
          <div>
            <h3 className="text-lg font-bold text-slate-900 flex items-center space-x-2">
              <BarChart2 className="w-5 h-5 text-indigo-600" />
              <span>Full Model Benchmark Comparison (Test Partition)</span>
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Evaluated on 80/10/10 Stratified Hold-Out Test Split across Hindi, Hinglish, and English.
            </p>
          </div>
          <span className="text-xs font-semibold px-3 py-1 bg-indigo-50 text-indigo-700 rounded-full border border-indigo-200">
            NVIDIA T4 GPU Benchmarked
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-700">
            <thead className="bg-slate-50 text-xs uppercase font-bold text-slate-500 border-b border-slate-200">
              <tr>
                <th className="py-3 px-4">Architecture</th>
                <th className="py-3 px-4">Parameters</th>
                <th className="py-3 px-4">Accuracy</th>
                <th className="py-3 px-4">Macro-F1</th>
                <th className="py-3 px-4">Precision</th>
                <th className="py-3 px-4">Recall</th>
                <th className="py-3 px-4">GPU Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {modelKeys.map((key) => {
                const m = models[key];
                const isWinner = key === "muril";
                return (
                  <tr key={key} className={isWinner ? "bg-indigo-50/40 font-semibold" : "hover:bg-slate-50"}>
                    <td className="py-3.5 px-4">
                      <div className="font-bold text-slate-900 flex items-center space-x-1.5">
                        {isWinner && <CheckCircle className="w-4 h-4 text-indigo-600" />}
                        <span>{m.display_name}</span>
                      </div>
                      <span className="text-[11px] text-slate-400 font-mono">{m.architecture}</span>
                    </td>
                    <td className="py-3.5 px-4 font-mono text-xs">{m.params_millions}M</td>
                    <td className="py-3.5 px-4">
                      <span className="px-2 py-0.5 rounded font-mono text-xs bg-slate-100 font-semibold">
                        {(m.accuracy * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3.5 px-4">
                      <span className={`px-2 py-0.5 rounded font-mono text-xs font-bold ${
                        isWinner ? "bg-indigo-600 text-white" : "bg-slate-100 text-slate-900"
                      }`}>
                        {(m.macro_f1 * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3.5 px-4 font-mono text-xs">{(m.macro_precision * 100).toFixed(1)}%</td>
                    <td className="py-3.5 px-4 font-mono text-xs">{(m.macro_recall * 100).toFixed(1)}%</td>
                    <td className="py-3.5 px-4 font-mono text-xs text-indigo-700">{m.latency_gpu_ms} ms</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Script-wise Breakdown & Confusion Matrix Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Script-wise Breakdown */}
        <div className="lg:col-span-7 bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-4">
          <h3 className="text-base font-bold text-slate-900 flex items-center space-x-2">
            <Layers className="w-4 h-4 text-indigo-600" />
            <span>Script & Language Disaggregation (Macro-F1 Score)</span>
          </h3>
          <p className="text-xs text-slate-500">
            Performance comparison isolating <strong>Devanagari Hindi</strong>, <strong>Romanized Hinglish</strong>, and <strong>English</strong> subsets.
          </p>

          <div className="space-y-4 pt-2">
            {modelKeys.map((key) => {
              const m = models[key];
              const lb = m.language_breakdown || {};
              const hindiF1 = (lb.hindi_devanagari?.macro_f1 || 0) * 100;
              const hinglishF1 = (lb.hinglish_roman?.macro_f1 || 0) * 100;
              const englishF1 = (lb.english_latin?.macro_f1 || 0) * 100;

              return (
                <div key={key} className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
                  <div className="flex justify-between items-center text-sm font-bold text-slate-800">
                    <span>{m.display_name}</span>
                    <span className="text-xs font-mono text-indigo-600 font-semibold">Overall: {(m.macro_f1 * 100).toFixed(1)}%</span>
                  </div>
                  
                  {/* Progress bars */}
                  <div className="space-y-1.5 text-xs">
                    <div>
                      <div className="flex justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                        <span>Hindi (Devanagari)</span>
                        <span className="font-mono">{hindiF1.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                        <div style={{ width: `${hindiF1}%` }} className="bg-orange-500 h-full rounded-full" />
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                        <span>Hinglish (Code-Switched Roman)</span>
                        <span className="font-mono">{hinglishF1.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                        <div style={{ width: `${hinglishF1}%` }} className="bg-indigo-600 h-full rounded-full" />
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-[11px] font-medium text-slate-600 mb-0.5">
                        <span>English (Latin Baseline)</span>
                        <span className="font-mono">{englishF1.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                        <div style={{ width: `${englishF1}%` }} className="bg-slate-600 h-full rounded-full" />
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Confusion Matrix Interactive Viewer */}
        <div className="lg:col-span-5 bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-slate-900">Confusion Matrix</h3>
            <select
              value={selectedConfusionModel}
              onChange={(e) => setSelectedConfusionModel(e.target.value)}
              className="text-xs p-1.5 bg-slate-100 border border-slate-300 rounded-lg font-semibold text-slate-800"
            >
              {modelKeys.map((k) => (
                <option key={k} value={k}>{models[k].display_name}</option>
              ))}
            </select>
          </div>

          <p className="text-xs text-slate-500">
            Predictions distribution for <strong>{activeModelData.display_name}</strong> on the hold-out test set (100 samples).
          </p>

          {/* 2x2 Matrix Grid */}
          <div className="pt-2">
            <div className="grid grid-cols-2 gap-3 text-center">
              <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200">
                <span className="text-[10px] uppercase font-bold text-emerald-700 tracking-wider block">
                  True Negatives (TN)
                </span>
                <span className="text-3xl font-extrabold text-emerald-900 mt-1 block">
                  {cm.true_negative}
                </span>
                <span className="text-xs text-emerald-600 mt-1 block font-medium">Actual REAL → Predicted REAL</span>
              </div>

              <div className="p-4 rounded-xl bg-amber-50 border border-amber-200">
                <span className="text-[10px] uppercase font-bold text-amber-700 tracking-wider block">
                  False Positives (FP)
                </span>
                <span className="text-3xl font-extrabold text-amber-900 mt-1 block">
                  {cm.false_positive}
                </span>
                <span className="text-xs text-amber-600 mt-1 block font-medium">Actual REAL → Predicted FAKE</span>
              </div>

              <div className="p-4 rounded-xl bg-rose-50 border border-rose-200">
                <span className="text-[10px] uppercase font-bold text-rose-700 tracking-wider block">
                  False Negatives (FN)
                </span>
                <span className="text-3xl font-extrabold text-rose-900 mt-1 block">
                  {cm.false_negative}
                </span>
                <span className="text-xs text-rose-600 mt-1 block font-medium">Actual FAKE → Predicted REAL</span>
              </div>

              <div className="p-4 rounded-xl bg-indigo-50 border border-indigo-200">
                <span className="text-[10px] uppercase font-bold text-indigo-700 tracking-wider block">
                  True Positives (TP)
                </span>
                <span className="text-3xl font-extrabold text-indigo-900 mt-1 block">
                  {cm.true_positive}
                </span>
                <span className="text-xs text-indigo-600 mt-1 block font-medium">Actual FAKE → Predicted FAKE</span>
              </div>
            </div>
          </div>

          <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-xs text-slate-600 leading-relaxed">
            <strong className="text-slate-800">Architectural Note: </strong>
            {activeModelData.research_notes}
          </div>
        </div>
      </div>

      {/* Scientific Insights Callouts */}
      <div className="bg-slate-900 text-white rounded-2xl p-6 sm:p-8 space-y-5">
        <div className="flex items-center space-x-2 text-indigo-400 font-bold text-xs uppercase tracking-wider">
          <BookOpen className="w-4 h-4" />
          <span>Core AI Research Findings</span>
        </div>
        <h3 className="text-xl sm:text-2xl font-extrabold tracking-tight">
          Why Multilingual Transformers Outperform Monolingual Baselines on Code-Mixed Data
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
          {insights.map((ins, i) => (
            <div key={i} className="p-4 rounded-xl bg-slate-800/80 border border-slate-700 space-y-2">
              <h4 className="font-bold text-sm text-indigo-200">{ins.title}</h4>
              <p className="text-xs text-slate-300 leading-relaxed">{ins.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
