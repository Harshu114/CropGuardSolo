import 'package:flutter/material.dart';

import 'main.dart';

class SectionHeading extends StatelessWidget {
  final String title;
  const SectionHeading(this.title, {super.key});

  @override
  Widget build(BuildContext context) {
    return Text(
      title,
      style: const TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w700,
          color: Color(0xFF2E7D32)),
    );
  }
}

class DemoImageCard extends StatelessWidget {
  final DemoResult result;
  final VoidCallback onRun;
  const DemoImageCard({super.key, required this.result, required this.onRun});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onRun,
      borderRadius: BorderRadius.circular(20),
      child: Container(
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: result.isHealthy ? Colors.green : AppTheme.primary.withAlpha(51),
            width: 1.5,
          ),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                color: result.isHealthy
                    ? Colors.green.withAlpha(24)
                    : AppTheme.primary.withAlpha(17),
                shape: BoxShape.circle,
              ),
              child: Icon(
                result.isHealthy ? Icons.check_circle_rounded : Icons.warning_amber_rounded,
                color: result.isHealthy ? Colors.green : AppTheme.primary,
                size: 40,
              ),
            ),
            const SizedBox(height: 12),
            Text(result.disease, style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 14)),
            const SizedBox(height: 4),
            Text(
              '${(result.confidence * 100).round()}% confidence',
              style: TextStyle(color: Colors.grey[600], fontSize: 12),
            ),
            const SizedBox(height: 8),
            Text(
              result.severity,
              style: TextStyle(
                color: result.severity == 'None'
                    ? Colors.green
                    : result.severity == 'High'
                        ? Colors.red
                        : Colors.orange,
                fontSize: 12,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class DemoResult {
  final String imageName;
  final String imageAsset;
  final String disease;
  final String crop;
  final double confidence;
  final String severity;
  final double affectedArea;
  final int detectedRegions;
  final String severityDescription;
  final String symptoms;
  final String recommendedActions;
  final String explanation;
  final String? maskAsset;
  final String? heatmapAsset;
  final bool isHealthy;

  DemoResult({
    required this.imageName,
    required this.imageAsset,
    required this.disease,
    required this.crop,
    required this.confidence,
    required this.severity,
    required this.affectedArea,
    required this.detectedRegions,
    required this.severityDescription,
    required this.symptoms,
    required this.recommendedActions,
    required this.explanation,
    this.maskAsset,
    this.heatmapAsset,
    this.isHealthy = false,
  });
}

class DemoData {
  static final allResults = [
    DemoResult(
      imageName: 'Late Blight',
      imageAsset: 'late_blight.jpg',
      disease: 'Late Blight',
      crop: 'Tomato',
      confidence: 1.0,
      severity: 'Moderate',
      affectedArea: 18.4,
      detectedRegions: 2,
      severityDescription: 'Moderate infection detected. Lesions are actively spreading but plant remains viable.',
      symptoms: 'Dark brown lesions with concentric rings on leaves. White sporulation may appear on undersides in humid conditions.',
      recommendedActions: 'Apply fungicide immediately. Remove affected leaves. Improve air circulation. Avoid overhead watering.',
      explanation: 'The model detected characteristic target-like lesions with concentric rings, a hallmark of Phytophthora infestans infection. The lesion pattern and distribution suggest moderate disease pressure.',
      maskAsset: 'late_blight_mask.png',
      heatmapAsset: 'late_blight_heatmap.png',
    ),
    DemoResult(
      imageName: 'Early Blight',
      imageAsset: 'early_blight.jpg',
      disease: 'Early Blight',
      crop: 'Tomato',
      confidence: 1.0,
      severity: 'Moderate',
      affectedArea: 14.7,
      detectedRegions: 3,
      severityDescription: 'Moderate infection. Multiple lesions present but plant canopy remains intact.',
      symptoms: 'Target-like concentric rings with dark margins. Older leaves show larger lesions. Yellow halos may surround spots.',
      recommendedActions: 'Prune affected leaves. Rotate crops next season. Use resistant varieties. Apply copper-based fungicide.',
      explanation: 'Alternaria solani infection identified by characteristic concentric ring patterns. Multiple detection regions suggest the disease has progressed beyond initial infection sites.',
      maskAsset: 'early_blight_mask.png',
      heatmapAsset: 'early_blight_heatmap.png',
    ),
    DemoResult(
      imageName: 'Bacterial Spot',
      imageAsset: 'bacterial_spot.jpg',
      disease: 'Bacterial Spot',
      crop: 'Tomato',
      confidence: 1.0,
      severity: 'High',
      affectedArea: 23.1,
      detectedRegions: 5,
      severityDescription: 'High severity. Multiple lesions detected across the leaf surface.',
      symptoms: 'Small, water-soaked lesions with angular margins. Dark spots on leaves and possibly fruit. Leaves may yellow and drop.',
      recommendedActions: 'Destroy infected plants. Do not compost. Use bactericide sprays. Avoid working with wet foliage.',
      explanation: 'Xanthomonas campestris pv. vesicatoria identified by small angular lesions with water-soaked appearance. The bacterial nature is indicated by the distinctive lesion margins.',
      maskAsset: 'bacterial_spot_mask.png',
      heatmapAsset: 'bacterial_spot_heatmap.png',
    ),
    DemoResult(
      imageName: 'Healthy',
      imageAsset: 'healthy.jpg',
      disease: 'Healthy',
      crop: 'Tomato',
      confidence: 1.0,
      severity: 'None',
      affectedArea: 0.0,
      detectedRegions: 0,
      severityDescription: 'No disease detected. Plant appears healthy.',
      symptoms: 'No symptoms detected. Leaf tissue appears normal with uniform green coloration.',
      recommendedActions: 'Continue regular monitoring. Maintain proper irrigation and fertilization schedule.',
      explanation: 'The leaf shows no signs of disease. Normal chlorophyll distribution and tissue structure confirmed. No lesions, discoloration, or abnormal patterns detected.',
      maskAsset: 'healthy_mask.png',
      heatmapAsset: 'healthy_heatmap.png',
      isHealthy: true,
    ),
  ];

  static DemoResult? getByImageName(String name) {
    return allResults.firstWhere((r) => r.imageName == name, orElse: () => allResults[3]);
  }
}

class LiveResultCard extends StatelessWidget {
  final LiveResult result;
  final VoidCallback onDismiss;
  const LiveResultCard({super.key, required this.result, required this.onDismiss});

  @override
  Widget build(BuildContext context) {
    final statusColor = result.classification.cls == 'Healthy'
        ? Colors.green
        : result.severity.stage == 'Severe'
            ? Colors.red
            : result.severity.stage == 'Moderate'
                ? Colors.orange
                : Colors.orange;

    return Container(
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        color: statusColor.withAlpha(26),
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        children: [
          Icon(
            result.classification.cls == 'Healthy'
                ? Icons.check_circle_rounded
                : Icons.warning_amber_rounded,
            color: statusColor,
            size: 54,
          ),
          const SizedBox(height: 10),
          Text(
            result.classification.cls,
            textAlign: TextAlign.center,
            style: const TextStyle(fontSize: 26, fontWeight: FontWeight.w800),
          ),
          const SizedBox(height: 6),
          Text(
            '${(result.classification.confidence * 100).round()}% confidence',
            style: const TextStyle(fontSize: 16, color: Colors.grey),
          ),
          const SizedBox(height: 6),
          if (!result.severityReliable)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text(
                'Severity not reliable — model is trained on leaf-area, not lesions',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 12, color: Colors.grey[600]),
              ),
            ),
        ],
      ),
    );
  }
}
