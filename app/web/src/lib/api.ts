import type { PredictResponse } from '@/types/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
// Mock ONLY when explicitly requested -- a silent mock fallback once made the
// site show fake results (random boxes, 1x1 CAM) when the env var was missing.
const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK === 'true';

// The 11 classes the served model (cropguard_model_v5.pt) was actually trained
// on, in data.yaml order. This list used to carry 16 names left over from the v2
// model -- magnesium/nitrogen/potassium deficiency, spotted_wilt_virus and the
// generic 'tomato' -- which made the site look like it covered nutrient
// deficiencies that the model has never seen and always answers wrongly.
// Keep in sync with api/model_server.py CLASS_NAMES and datasetclaude/data.yaml.
const CLASSES = [
  'Healthy', 'bacterial_spot', 'early_blight', 'late_blight', 'leaf_miner',
  'leaf_mold', 'mosaic_virus', 'septoria_leaf_spot', 'spider_mites',
  'target_spot', 'yellow_leaf_curl_virus'
];

const STAGE_COLORS: Record<string, string> = {
  Healthy: 'bg-green-500',
  Mild: 'bg-yellow-500',
  Moderate: 'bg-orange-500',
  Severe: 'bg-red-500',
};

const DISEASE_INFO: Record<string, { symptoms: string; treatment: string }> = {
  Healthy: { symptoms: 'No visible disease symptoms.', treatment: 'No treatment needed. Continue regular care.' },
  bacterial_spot: { symptoms: 'Small, dark, water-soaked lesions on leaves.', treatment: 'Copper-based bactericides; avoid overhead watering.' },
  early_blight: { symptoms: 'Concentric rings on lower leaves, yellowing.', treatment: 'Fungicides (chlorothalonil, mancozeb); rotate crops.' },
  late_blight: { symptoms: 'Large, dark, greasy lesions; white mold on underside.', treatment: 'Fungicides (mandipropamid); destroy infected plants.' },
  leaf_miner: { symptoms: 'Winding white tunnels inside leaves.', treatment: 'Neem oil; parasitic wasps; remove affected leaves.' },
  leaf_mold: { symptoms: 'Yellow spots on top, olive-green mold underneath.', treatment: 'Improve ventilation; fungicides; resistant varieties.' },
  mosaic_virus: { symptoms: 'Mottled light/dark green patterns, leaf distortion.', treatment: 'Remove infected plants; control aphids; use virus-free seeds.' },
  septoria_leaf_spot: { symptoms: 'Small circular spots with gray centers.', treatment: 'Copper fungicides; mulch; crop rotation.' },
  spider_mites: { symptoms: 'Fine webbing; stippled, bronze leaves.', treatment: 'Insecticidal soap; neem oil; increase humidity.' },
  target_spot: { symptoms: 'Concentric brown rings with yellow halos.', treatment: 'Fungicides; improve air flow; avoid wet foliage.' },
  yellow_leaf_curl_virus: { symptoms: 'Upward curling, yellowing, stunted growth.', treatment: 'Remove plants; control whiteflies; resistant varieties.' },
};

function humanize(cls: string): string {
  return cls.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function mockPredict(file: File): Promise<PredictResponse> {
  return new Promise(resolve => {
    setTimeout(() => {
      const isHealthy = Math.random() > 0.3;
      const mainClass = isHealthy ? 'Healthy' : CLASSES[1 + Math.floor(Math.random() * (CLASSES.length - 1))];
      const conf = isHealthy ? 0.92 + Math.random() * 0.06 : 0.65 + Math.random() * 0.3;
      const top3 = CLASSES.slice(0, 3).map((c, i) => ({ class: c, confidence: conf - i * 0.15 }));
      const lesion = isHealthy ? 0.02 : 0.15 + Math.random() * 0.5;
      const sevCont = Math.min(0.95, lesion * 1.2);
      const stageIdx = sevCont < 0.1 ? 0 : sevCont < 0.35 ? 1 : sevCont < 0.65 ? 2 : 3;
      const stages = ['Healthy', 'Mild', 'Moderate', 'Severe'] as const;
      const stage = stages[stageIdx];

      const w = 1920, h = 1080;
      const detections = isHealthy ? [] : [{
        box: [w * 0.2, h * 0.2, w * 0.7, h * 0.7],
        class: mainClass,
        score: 0.4 + Math.random() * 0.4,
        box_norm: [0.2, 0.2, 0.7, 0.7],
      }];

      resolve({
        image_id: crypto.randomUUID(),
        width: w,
        height: h,
        detections,
        classification: { class: mainClass, confidence: conf, top3 },
        segmentation: { mask_png_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==', leaf_fraction: lesion },
        severity: { continuous: sevCont, stage, stage_idx: stageIdx },
        cam_png_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==',
        timings_ms: { preprocess: 12, model: 45, postprocess: 6 },
        model_version: 'cropguard-mock-v1',
      });
    }, 1500 + Math.random() * 1000);
  });
}

export async function predictImage(file: File, conf = 0.25, withCam = true): Promise<PredictResponse> {
  if (USE_MOCK) return mockPredict(file);

  const form = new FormData();
  form.append('image', file);
  form.append('conf', String(conf));
  form.append('with_cam', String(withCam));

  const res = await fetch(`${API_URL}/v1/predict`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function checkHealth() {
  if (USE_MOCK) return { status: 'ok', model: 'mock', device: 'cpu' };
  const res = await fetch(`${API_URL}/v1/health`);
  return res.json();
}

export function getMaskUrl(id: string) {
  return USE_MOCK ? 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==' : `${API_URL}/v1/masks/${id}.png`;
}

export function getCamUrl(id: string) {
  return USE_MOCK ? 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==' : `${API_URL}/v1/cams/${id}.png`;
}

export { CLASSES, STAGE_COLORS, DISEASE_INFO, humanize };