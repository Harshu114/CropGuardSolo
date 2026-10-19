'use client';

import { useState } from 'react';
import { CLASSES, STAGE_COLORS, DISEASE_INFO, humanize } from '@/lib/api';
import type { PredictResponse, Detection } from '@/types/api';

interface ResultViewProps {
  result: PredictResponse;
  originalPreview: string;
  onNewScan: () => void;
}

const STAGE_LABELS = ['Healthy', 'Mild', 'Moderate', 'Severe'] as const;

export default function ResultView({ result, originalPreview, onNewScan }: ResultViewProps) {
  // Off by default: the segmentation head predicts the leaf REGION (its training
  // target is the filled leaf box, not lesions), so on a close-up leaf photo this
  // overlay covers most of the frame and reads as a broken result. Still available
  // behind the checkbox.
  const [showMask, setShowMask] = useState(false);
  const [showCam, setShowCam] = useState(false);
  const [maskLoaded, setMaskLoaded] = useState(false);
  const [camLoaded, setCamLoaded] = useState(false);

  const { classification, severity, segmentation, detections, cam_png_url, model_version } = result;
  const disease = classification.class;
  const info = DISEASE_INFO[disease] || { symptoms: 'Information not available.', treatment: 'Consult an expert.' };

  const severityPercent = Math.round(severity.continuous * 100);
  const stageColor = STAGE_COLORS[severity.stage] || 'bg-gray-500';
  // Shortest side of the ORIGINAL upload. The corpus is 100% square 320x320 and
  // measured test accuracy falls 0.862 -> 0.589 when a photo is only 160 px.
  const minSide = Math.min(result.width, result.height);

  return (
    <div className="space-y-6">
      {/* Headline Card */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-gray-900">
              {humanize(disease)}
            </h1>
            <div className="mt-2 flex flex-wrap items-center gap-3">
              {/* The severity stage is deliberately NOT shown beside the
                  diagnosis unless the head is validated. On v5 `reliable` is
                  false for every image: the old badge printed "Severe" next to
                  the disease name on virtually every scan, which made correct
                  diagnoses look broken. */}
              {severity.reliable && (
                <span className={`px-3 py-1 rounded-full text-sm font-semibold text-white ${stageColor}`}>
                  {severity.stage}
                </span>
              )}
              <span className="text-lg font-mono text-gray-700">
                {Math.round(classification.confidence * 100)}% confidence
              </span>
              <span className="text-sm text-gray-500 px-2 py-1 rounded bg-gray-100">
                {model_version}
              </span>
            </div>
          </div>
          <button
            onClick={onNewScan}
            className="w-full md:w-auto px-6 py-3 bg-green-600 text-white font-semibold rounded-xl hover:bg-green-700 transition-colors"
          >
            New Scan
          </button>
        </div>
      </div>

      {/* Low-confidence honesty banner */}
      {classification.confidence < 0.6 && (
        <div className="bg-amber-50 border border-amber-300 text-amber-900 rounded-xl px-4 py-3 text-sm">
          <strong>Low confidence ({Math.round(classification.confidence * 100)}%).</strong>{' '}
          The model is uncertain about this photo — check the Top Alternatives below
          and consult an expert before acting on this diagnosis.
        </div>
      )}

      {/* Honesty panel: input quality + model scope.
          Both of these explain real failures seen in testing. The user's
          low-resolution uploads (148x148) were the dominant cause, and nutrient
          deficiencies are simply not classes this model has (the old 16-class
          list made it look otherwise). */}
      <div className="bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 text-sm text-gray-700 space-y-2">
        {minSide < 200 && (
          <p className="text-rose-800">
            <strong>
              Low-resolution photo ({result.width}×{result.height} px).
            </strong>{' '}
            CropGuard was trained on 320×320 images and its measured accuracy drops
            sharply below about 200 px, so treat this result as unreliable — a wrong
            answer here is more likely to be the photo than the model. Re-shoot closer
            or upload a larger version if you can.
          </p>
        )}
        <p>
          <strong>Scope:</strong> CropGuard recognises these 11 classes only —{' '}
          {CLASSES.map(humanize).join(', ')}. Nutrient deficiencies (magnesium,
          nitrogen, potassium), spotted wilt virus and other conditions are outside
          its training set: it will map such a leaf onto the nearest class it knows,
          which is a wrong answer rather than an &quot;unknown&quot;.
        </p>
      </div>
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="p-4 border-b border-gray-100 flex flex-wrap gap-2">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showMask}
              onChange={e => setShowMask(e.target.checked)}
              className="w-4 h-4 text-green-600 rounded border-gray-300"
            />
            <span className="text-sm font-medium text-gray-700" title="Predicted leaf/plant region — not damaged tissue.">
              Leaf Region
            </span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showCam}
              onChange={e => setShowCam(e.target.checked)}
              className="w-4 h-4 text-orange-600 rounded border-gray-300"
            />
            <span className="text-sm font-medium text-gray-700">LayerCAM</span>
          </label>
        </div>
        <div
          className="relative w-full bg-gray-100 overflow-hidden"
          style={{ aspectRatio: `${result.width} / ${result.height}` }}
        >
          {/* Container aspect == photo aspect, so box_norm percentages map
              exactly onto the visible image (fixes boxes floating in the
              letterbox gray bars of the old object-contain layout). */}
          <img
            src={originalPreview}
            alt="Original leaf photo"
            className="absolute inset-0 w-full h-full"
          />
          {showMask && maskLoaded && (
            <img
              src={result.segmentation.mask_png_url}
              alt="Predicted leaf region overlay"
              className="absolute inset-0 w-full h-full opacity-50"
              onLoad={() => setMaskLoaded(true)}
            />
          )}
          {showCam && camLoaded && (
            <img
              src={cam_png_url}
              alt="LayerCAM heatmap overlay"
              className="absolute inset-0 w-full h-full opacity-60"
              onLoad={() => setCamLoaded(true)}
            />
          )}
          {detections.map((det, i) => (
            <div
              key={i}
              className="absolute border-2 border-green-500 bg-green-500/20"
              style={{
                left: `${(det.box_norm[0] * 100)}%`,
                top: `${(det.box_norm[1] * 100)}%`,
                width: `${((det.box_norm[2] - det.box_norm[0]) * 100)}%`,
                height: `${((det.box_norm[3] - det.box_norm[1]) * 100)}%`,
              }}
            >
              <div className="absolute -top-5 left-0 bg-green-500 text-white text-xs px-1.5 rounded whitespace-nowrap">
                {humanize(det.class)} {Math.round(det.score * 100)}%
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Top-3 Alternates */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Alternatives</h3>
        <div className="space-y-3">
          {classification.top3.slice(1).map((item, idx) => (
            <div key={idx} className="flex items-center gap-3">
              <span className="w-32 text-sm text-gray-600">{humanize(item.class)}</span>
              <div className="flex-1 h-3 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full transition-all duration-500"
                  style={{ width: `${item.confidence * 100}%` }}
                />
              </div>
              <span className="w-16 text-right text-sm font-mono text-gray-700">
                {Math.round(item.confidence * 100)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Severity Meter.
          Marked experimental on purpose. On the v5 checkpoint this head scores
          sev_r2 = -1.41 and its MAE (0.068) is WORSE than always predicting the
          average (0.009), because 99.6% of the training data sits in one
          severity bucket -- so it reports "Severe" for essentially every leaf.
          The server therefore returns severity.reliable = false, and this card
          shows the explanation instead of a stage until that flag flips. The
          coloured meter below is kept (behind the flag) so it comes back on its
          own once the head actually beats the constant baseline. */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold text-gray-900">Severity</h3>
            <span
              className="px-2 py-0.5 rounded-full text-xs font-semibold bg-amber-100 text-amber-800 border border-amber-300"
              title="This head is not validated and currently over-predicts 'Severe'."
            >
              Experimental
            </span>
          </div>
          {severity.reliable && (
            <span className="text-sm font-mono text-gray-700">{severityPercent}%</span>
          )}
        </div>
        <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
          Not reported. This head was trained on labels that track how much of the
          frame the leaf covers, so it answers the same stage for almost every photo
          and cannot be used as a severity estimate. It is being reworked; the number
          is only meaningful once the metric beats a constant predictor.
        </p>
        {severity.reliable && (
          <>
            <div className="mt-3 h-4 bg-gray-200 rounded-full overflow-hidden relative">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${severityPercent}%`,
                  background: `linear-gradient(90deg, #22c55e, #eab308, #f97316, #ef4444)`,
                }}
              />
              <div className="absolute top-full left-0 right-0 flex justify-between text-xs text-gray-500 mt-1">
                <span>Healthy</span>
                <span>Mild</span>
                <span>Moderate</span>
                <span>Severe</span>
              </div>
            </div>
            <div className="mt-3 flex items-center gap-3">
              <span className={`px-3 py-1 rounded-full text-sm font-semibold text-white ${stageColor}`}>
                {severity.stage}
              </span>
            </div>
          </>
        )}
        <div className="mt-3 flex items-center gap-3">
          <span className="text-sm text-gray-600">
            Leaf region: {Math.round(segmentation.leaf_fraction * 100)}% of photo
          </span>
        </div>
      </div>

      {/* Info Panel */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">{humanize(disease)} — Info</h3>
        <dl className="space-y-4 text-gray-700">
          <div>
            <dt className="font-medium">Symptoms</dt>
            <dd className="mt-1">{info.symptoms}</dd>
          </div>
          <div>
            <dt className="font-medium">Basic Guidance</dt>
            <dd className="mt-1">{info.treatment}</dd>
          </div>
        </dl>
      </div>

      {/* Disclaimer */}
      <footer className="text-center text-sm text-gray-500 border-t border-gray-200 pt-4">
        <p>
          <strong>Disclaimer:</strong> CropGuard is a research model, not a certified plant-disease diagnostic tool.
          Results show confidence levels and should not substitute expert agricultural advice.
        </p>
      </footer>
    </div>
  );
}