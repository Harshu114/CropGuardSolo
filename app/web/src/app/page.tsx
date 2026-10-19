'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import UploadArea from '@/components/UploadArea';
import AnalyzingView from '@/components/AnalyzingView';
import ResultView from '@/components/ResultView';
import { predictImage, checkHealth } from '@/lib/api';
import type { PredictResponse } from '@/types/api';

export default function Home() {
  const [state, setState] = useState<'idle' | 'analyzing' | 'result' | 'error'>('idle');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiReady, setApiReady] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    checkHealth().then(() => setApiReady(true)).catch(() => setApiReady(false));
  }, []);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
    setError(null);
  }, []);

  const startAnalysis = useCallback(async () => {
    if (!file || !preview) return;
    setState('analyzing');
    setElapsed(0);
    timerRef.current = setInterval(() => setElapsed(e => e + 100), 100);

    try {
      const res = await predictImage(file);
      setResult(res);
      setState('result');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Analysis failed');
      setState('error');
    } finally {
      if (timerRef.current) clearInterval(timerRef.current);
    }
  }, [file, preview]);

  const reset = useCallback(() => {
    setState('idle');
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setElapsed(0);
    if (timerRef.current) clearInterval(timerRef.current);
  }, []);

  return (
    <main className="min-h-screen bg-gray-50 py-8 px-4 sm:py-12 sm:px-6">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900">CropGuard</h1>
          <p className="mt-2 text-gray-600">
            AI-powered tomato leaf disease diagnosis — detection, classification, segmentation, severity & explainability
          </p>
          {!apiReady && (
            <p className="mt-2 text-sm text-orange-600">⚠ Running in mock mode (no API connected)</p>
          )}
        </header>

        {/* Main Content */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 sm:p-8">
          {state === 'idle' && (
            <UploadArea onFile={handleFile} disabled={state !== 'idle'} />
          )}

          {state === 'analyzing' && (
            <>
              <AnalyzingView />
              <p className="text-center text-sm text-gray-500 mt-4">
                Elapsed: {(elapsed / 1000).toFixed(1)}s
              </p>
            </>
          )}

          {state === 'result' && result && preview && (
            <ResultView
              result={result}
              originalPreview={preview}
              onNewScan={reset}
            />
          )}

          {state === 'error' && (
            <div className="text-center py-8">
              <div className="text-red-500 text-4xl mb-3">⚠</div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Analysis Failed</h3>
              <p className="text-gray-600 mb-6">{error}</p>
              <button
                onClick={reset}
                className="px-6 py-3 bg-gray-100 text-gray-700 font-semibold rounded-xl hover:bg-gray-200 transition-colors"
              >
                Try Again
              </button>
            </div>
          )}

          {file && state === 'idle' && (
            <div className="mt-6 text-center">
              <button
                onClick={startAnalysis}
                disabled={state !== 'idle'}
                className="w-full sm:w-auto px-8 py-4 bg-green-600 text-white font-semibold text-lg rounded-xl hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Analyze Leaf
              </button>
            </div>
          )}
        </div>

        {/* Footer */}
        <footer className="mt-8 text-center text-sm text-gray-500">
          <p>Built with Next.js + FastAPI • Research model — not for clinical use</p>
        </footer>
      </div>
    </main>
  );
}