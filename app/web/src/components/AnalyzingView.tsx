'use client';

export default function AnalyzingView() {
  return (
    <div className="space-y-6 text-center">
      <div className="relative w-32 h-32 mx-auto">
        <svg className="w-full h-full text-green-500" viewBox="0 0 100 100">
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke="currentColor"
            strokeWidth="8"
            className="opacity-25"
          />
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke="currentColor"
            strokeWidth="8"
            strokeDasharray="283"
            strokeDashoffset="0"
            strokeLinecap="round"
            className="animate-spin transform -rotate-90"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-3xl font-bold text-gray-900">AI</span>
        </div>
      </div>
      <div>
        <h2 className="text-xl font-semibold text-gray-900">Analyzing your leaf…</h2>
        <p className="mt-1 text-gray-500">Running multi-task model (detection + classification + segmentation + severity + CAM)</p>
      </div>
      <div className="h-2 bg-gray-200 rounded-full overflow-hidden w-3/4 mx-auto">
        <div className="h-full bg-green-500 rounded-full animate-pulse" style={{ width: '100%' }} />
      </div>
      <p className="text-xs text-gray-400">This usually takes 2–5 seconds</p>
    </div>
  );
}