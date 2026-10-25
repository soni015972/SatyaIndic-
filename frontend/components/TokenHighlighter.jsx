import React from "react";

export default function TokenHighlighter({ tokens }) {
  if (!tokens || tokens.length === 0) {
    return <p className="text-slate-400 text-sm italic">No token attribution data available.</p>;
  }

  const getHighlightStyle = (item) => {
    const imp = item.importance;
    const abs = Math.min(Math.abs(imp), 1.0);

    if (item.label_driver === "FAKE" || imp > 0.05) {
      // Red tint for fake drivers
      const alpha = Math.max(0.15, abs * 0.7);
      return {
        backgroundColor: `rgba(239, 68, 68, ${alpha})`,
        borderBottom: "2px solid rgba(220, 38, 38, 0.8)",
      };
    } else if (item.label_driver === "REAL" || imp < -0.05) {
      // Emerald tint for real drivers
      const alpha = Math.max(0.15, abs * 0.7);
      return {
        backgroundColor: `rgba(16, 185, 129, ${alpha})`,
        borderBottom: "2px solid rgba(5, 150, 105, 0.8)",
      };
    }
    return {
      backgroundColor: "transparent",
    };
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-1.5 p-4 bg-slate-50 rounded-xl border border-slate-200 leading-relaxed font-sans text-base">
        {tokens.map((item, idx) => (
          <span
            key={idx}
            style={getHighlightStyle(item)}
            className="px-1.5 py-0.5 rounded cursor-pointer transition-all hover:scale-105 inline-block group relative"
          >
            {item.token}
            {/* Tooltip on hover */}
            <span className="invisible group-hover:visible absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 px-2 py-1 text-[11px] font-medium bg-slate-900 text-white rounded shadow-lg whitespace-nowrap z-20 pointer-events-none">
              Driver: <strong className={item.importance > 0 ? "text-red-400" : "text-emerald-400"}>{item.label_driver}</strong> ({item.importance > 0 ? `+${item.importance}` : item.importance})
            </span>
          </span>
        ))}
      </div>

      <div className="flex items-center justify-between text-xs text-slate-500 px-1">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1.5">
            <span className="w-3 h-3 rounded bg-red-500/30 border-b-2 border-red-600 inline-block" />
            <span>Pushes toward <strong>FAKE</strong> (Misinformation Cue)</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-3 h-3 rounded bg-emerald-500/30 border-b-2 border-emerald-600 inline-block" />
            <span>Pushes toward <strong>REAL</strong> (Credibility Cue)</span>
          </div>
        </div>
        <span className="italic">Hover over words to view attribution values</span>
      </div>
    </div>
  );
}
