export interface Detection {
  box: number[];
  class: string;
  score: number;
  box_norm: number[];
  source?: string;
}

export interface ClassificationResult {
  class: string;
  confidence: number;
  top3: Array<{ class: string; confidence: number }>;
}

export interface SegmentationResult {
  mask_png_url: string;
  /**
   * Fraction of the photo the segmentation head marks as LEAF/plant region.
   * Named `lesion_fraction` until the corpus masks were checked: they are binary
   * {0,255} leaf masks (the 128 "lesion" value never occurs), so this head
   * predicts a leaf region, not damaged tissue.
   */
  leaf_fraction: number;
}

export interface SeverityResult {
  continuous: number;
  stage: 'Healthy' | 'Mild' | 'Moderate' | 'Severe';
  stage_idx: number;
  /**
   * False when the severity head is not a validated signal. On v5 it is always
   * false (sev_r2 -1.41, mae worse than a constant predictor, and its labels
   * were derived from leaf-box area so nearly every image is "Severe").
   * The UI must not present an unreliable stage as a diagnosis.
   * Undefined on older/mock payloads and treated as reliable there.
   */
  reliable?: boolean;
}

export interface Timings {
  preprocess: number;
  model: number;
  postprocess: number;
}

export interface PredictResponse {
  image_id: string;
  width: number;
  height: number;
  /** What the UI draws: whole-leaf boxes from the detection head. */
  detections: Detection[];
  /** Alias of `detections` (both are the detection head's leaf boxes). */
  leaf_boxes?: Detection[];
  classification: ClassificationResult;
  segmentation: SegmentationResult;
  severity: SeverityResult;
  cam_png_url: string;
  timings_ms: Timings;
  model_version: string;
}

export interface HealthResponse {
  status: string;
  model: string;
  device: string;
}