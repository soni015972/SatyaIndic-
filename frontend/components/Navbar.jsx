import React from "react";
import { ShieldAlert, BarChart3, Search, Sparkles, ExternalLink } from "lucide-react";

export default function Navbar({ activeTab, setActiveTab, apiOnline }) {
  return (
    <header className="border-b border-slate-200 bg-white/80 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center text-white shadow-md shadow-indigo-100">
            <ShieldAlert className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-bold text-slate-900 text-lg tracking-tight">SatyaIndic</h1>
              <span className="text-[10px] font-semibold uppercase tracking-wider bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded-full border border-indigo-200">
                Research AI
              </span>
            </div>
            <p className="text-xs text-slate-500 hidden sm:block">
              Multilingual & Code-Switched (Hindi/Hinglish) Fake News Detection
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center space-x-1 bg-slate-100 p-1 rounded-xl">
          <button
            onClick={() => setActiveTab("detector")}
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition ${
              activeTab === "detector"
                ? "bg-white text-indigo-700 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Search className="w-4 h-4" />
            <span>Detector</span>
          </button>
          <button
            onClick={() => setActiveTab("benchmark")}
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition ${
              activeTab === "benchmark"
                ? "bg-white text-indigo-700 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            <span>Research Benchmarks</span>
          </button>
        </div>

        {/* Status Indicator */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 border border-slate-200">
            <span
              className={`w-2 h-2 rounded-full ${
                apiOnline ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
              }`}
            />
            <span className="text-slate-600 hidden md:inline">
              {apiOnline ? "Backend Connected" : "Local Mode"}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
