import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

/// Client contract for the live CROPGuard v5 API.
///
/// Mirrors [app/web/src/lib/api.ts] so the web and mobile fronts agree on the
/// shape of [PredictResponse]. The server (`app/api/model_server.py`) is the
/// single authority for preprocessing: it LETTERBOXES the uploaded image to
/// 320x320 on a gray-114 pad (matching the training augmentation), then runs
/// detection + classification + segmentation + CAM. Clients send the **raw**
/// image bytes — they must not resize to 320x320 themselves.
///
/// [PredictResponse] is documented against the server's return dict. The server
/// does NOT emit `lesion_fraction` (the value is the predicted LEAF REGION). It
/// also does NOT emit a top-level `warnings` key today; the `reliable` field on
/// [Severity] is the honest signal clients should use.

class PredictResponse {
  final String imageId;
  final int width;
  final int height;
  final List<Detection> detections;
  final Classification classification;
  final Segmentation segmentation;
  final Severity severity;
  final Map<String, int> timingsMs;
  final String modelVersion;
  final String? patchTier;

  PredictResponse.fromJson(Map<String, dynamic> j)
      : imageId = j['image_id'] as String,
        width = j['width'] as int,
        height = j['height'] as int,
        detections = (j['detections'] as List<dynamic>?)
                ?.map((e) => Detection.fromJson(e as Map<String, dynamic>))
                .toList() ?? [],
        classification = Classification.fromJson(j['classification'] as Map<String, dynamic>),
        segmentation = Segmentation.fromJson(j['segmentation'] as Map<String, dynamic>),
        severity = Severity.fromJson(j['severity'] as Map<String, dynamic>),
        timingsMs = Map<String, int>.from(j['timings_ms'] as Map),
        modelVersion = j['model_version'] as String? ?? 'unknown',
        patchTier = j['patches_tier'] as String? ?? null;

  Map<String, dynamic> toJson() => <String, dynamic>{
        'image_id': imageId,
        'width': width,
        'height': height,
        'detections': detections.map((e) => e.toJson()).toList(),
        'classification': classification.toJson(),
        'segmentation': segmentation.toJson(),
        'severity': severity.toJson(),
        'timings_ms': timingsMs,
        'model_version': modelVersion,
        'patches_tier': patchTier,
      };

  // Attachments the overlay adds after it downloads the two generated PNGs.
  Uint8List? maskBytes;
  Uint8List? camBytes;
}

class Detection {
  final List<double> box;
  final String cls;
  final double score;
  final List<double> boxNorm;
  final String source;

  Detection.fromJson(Map<String, dynamic> j)
      : box = (j['box'] as List).cast<double>(),
        cls = j['class'] as String,
        score = j['score'] as double,
        boxNorm = (j['box_norm'] as List).cast<double>(),
        source = j['source'] as String;

  Map<String, dynamic> toJson() => <String, dynamic>{
        'box': box,
        'class': cls,
        'score': score,
        'box_norm': boxNorm,
        'source': source,
      };
}

class Classification {
  final String cls;
  final double confidence;
  final List<Top3> top3;

  Classification.fromJson(Map<String, dynamic> j)
      : cls = j['class'] as String,
        confidence = j['confidence'] as double,
        top3 = (j['top3'] as List<dynamic>)
            .map((e) => Top3.fromJson(e as Map<String, dynamic>))
            .toList();

  const Classification.empty()
      : cls = '',
        confidence = 0.0,
        top3 = const [];

  Map<String, dynamic> toJson() => <String, dynamic>{
        'class': cls,
        'confidence': confidence,
        'top3': top3.map((e) => e.toJson()).toList(),
      };
}

class Top3 {
  final String cls;
  final double confidence;
  Top3.fromJson(Map<String, dynamic> j)
      : cls = j['class'] as String,
        confidence = j['confidence'] as double;

  Map<String, dynamic> toJson() => <String, dynamic>{
        'class': cls,
        'confidence': confidence,
      };
}

class Segmentation {
  final String maskPngUrl;
  final String? camPngUrl;
  final double leafFraction;

  Segmentation.fromJson(Map<String, dynamic> j)
      : maskPngUrl = j['mask_png_url'] as String,
        camPngUrl = j['cam_png_url'] as String?,
        leafFraction = j['leaf_fraction'] as double;

  const Segmentation.empty()
      : maskPngUrl = '',
        camPngUrl = null,
        leafFraction = 0.0;

  Map<String, dynamic> toJson() => <String, dynamic>{
        'mask_png_url': maskPngUrl,
        'cam_png_url': camPngUrl,
        'leaf_fraction': leafFraction,
      };
}

class Severity {
  final double continuous;
  final String stage;
  final int stageIdx;
  final bool reliable;

  Severity.fromJson(Map<String, dynamic> j)
      : continuous = j['continuous'] as double,
        stage = j['stage'] as String,
        stageIdx = j['stage_idx'] as int,
        reliable = j['reliable'] as bool;

  const Severity.empty()
      : continuous = 0.0,
        stage = 'Healthy',
        stageIdx = 0,
        reliable = false;

  Map<String, dynamic> toJson() => <String, dynamic>{
        'continuous': continuous,
        'stage': stage,
        'stage_idx': stageIdx,
        'reliable': reliable,
      };
}

/// Sends [rawBytes] (a JPEG/PNG as captured or picked) to `/v1/predict`.
///
/// The server does its own letterbox + normalization, so the client sends the
/// full-resolution bytes and lets the API be the canonical preprocessor.
Future<PredictResponse> predictImage({
  required List<int> rawBytes,
  String apiBase = 'http://10.0.2.2:8000',
  double conf = 0.15,
  bool withCam = true,
}) async {
  final url = Uri.parse('$apiBase/v1/predict');

  final request = http.MultipartRequest('POST', url);

  request.files.add(http.MultipartFile(
    'image',
    http.ByteStream.fromBytes(rawBytes),
    rawBytes.length,
    contentType: null, // server decodes any image via cv2.imdecode
  ));

  request.fields['conf'] = conf.toString();
  request.fields['with_cam'] = (withCam ? 'true' : 'false');

  final streamed = await request.send();
  final resp = await http.Response.fromStream(streamed);

  if (resp.statusCode != 200) {
    final body = resp.body;
    String msg;
    try {
      final j = jsonDecode(body);
      msg = j is Map<String, dynamic> ? (j['error'] as String?) ?? body : body;
    } catch (_) {
      msg = body;
    }
    throw Exception('predict failed (HTTP ${resp.statusCode}): $msg');
  }

  final j = jsonDecode(resp.body) as Map<String, dynamic>;
  return PredictResponse.fromJson(j);
}
