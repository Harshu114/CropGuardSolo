import 'dart:io';
import 'dart:async';
import 'package:flutter/material.dart';

void main() {
  runApp(const CropGuardApp());
}

// ─── THEME ────────────────────────────────────────────────────────────────────
class AppTheme {
  static const primary = Color(0xFF2E7D32);
  static const accent = Color(0xFFA5D66A);
  static const darkBg = Color(0xFF0C120D);
  static const error = Color(0xFFC62828);

  static final light = ThemeData(
    useMaterial3: true,
    colorScheme: ColorScheme.fromSeed(seedColor: primary, brightness: Brightness.light),
    scaffoldBackgroundColor: const Color(0xFFF7F8F2),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: Colors.white,
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(16), borderSide: BorderSide.none),
    ),
  );

  static final dark = ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    colorScheme: ColorScheme.fromSeed(seedColor: primary, brightness: Brightness.dark),
    scaffoldBackgroundColor: darkBg,
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: const Color(0xFF172019),
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(16), borderSide: BorderSide.none),
    ),
  );
}

// ─── DEMO DATA MODEL ──────────────────────────────────────────────────────────
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

  const DemoResult({
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
  static const allResults = [
    DemoResult(
      imageName: 'Late Blight',
      imageAsset: 'demo_late_blight.jpg',
      disease: 'Late Blight',
      crop: 'Tomato',
      confidence: 0.912,
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
      imageAsset: 'demo_early_blight.jpg',
      disease: 'Early Blight',
      crop: 'Tomato',
      confidence: 0.894,
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
      imageAsset: 'demo_bacterial_spot.jpg',
      disease: 'Bacterial Spot',
      crop: 'Tomato',
      confidence: 0.876,
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
      imageAsset: 'demo_healthy.jpg',
      disease: 'Healthy',
      crop: 'Tomato',
      confidence: 0.976,
      severity: 'None',
      affectedArea: 0.0,
      detectedRegions: 0,
      severityDescription: 'No disease detected. Plant appears healthy.',
      symptoms: 'No symptoms detected. Leaf tissue appears normal with uniform green coloration.',
      recommendedActions: 'Continue regular monitoring. Maintain proper irrigation and fertilization schedule.',
      explanation: 'The leaf shows no signs of disease. Normal chlorophyll distribution and tissue structure confirmed. No lesions, discoloration, or abnormal patterns detected.',
      maskAsset: 'empty_mask.png',
      heatmapAsset: 'empty_heatmap.png',
      isHealthy: true,
    ),
  ];

  static DemoResult? getByImageName(String name) {
    return allResults.firstWhere((r) => r.imageName == name, orElse: () => allResults[3]);
  }
}

// ─── APP ──────────────────────────────────────────────────────────────────────
class CropGuardApp extends StatelessWidget {
  const CropGuardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CROPGuard AI',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light,
      darkTheme: AppTheme.dark,
      themeMode: ThemeMode.system,
      home: const HomeScreen(),
    );
  }
}

// ─── HOME SCREEN ──────────────────────────────────────────────────────────────
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(20, 22, 20, 110),
          children: [
            Row(
              children: [
                const CircleAvatar(radius: 25, backgroundColor: Color(0xFFDCEFC8), child: Icon(Icons.eco, color: AppTheme.primary, size: 29)),
                const SizedBox(width: 12),
                Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: const [Text('Good morning 👋', style: TextStyle(fontSize: 14, color: Colors.grey)), SizedBox(height: 2), Text('CROPGuard AI', style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800))])),
                IconButton(onPressed: () {}, icon: const Icon(Icons.notifications_none_rounded)),
              ],
            ),
            const SizedBox(height: 24),
            Container(
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(borderRadius: BorderRadius.circular(30), gradient: const LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [Color(0xFF245D2A), Color(0xFF438A3F)])),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: const [
                Text('Know your crop.\nGrow with confidence.', style: TextStyle(color: Colors.white, fontSize: 27, height: 1.12, fontWeight: FontWeight.w800)),
                SizedBox(height: 10),
                Text('Select a demo image to simulate AI disease analysis.', style: TextStyle(color: Colors.white70, height: 1.4)),
              ]),
            ),
            const SizedBox(height: 24),
            const SectionHeading('Select a Demo Image'),
            const SizedBox(height: 12),
            GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: DemoData.allResults.length,
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(crossAxisCount: 2, crossAxisSpacing: 12, mainAxisSpacing: 12, childAspectRatio: 0.75),
              itemBuilder: (context, index) {
                final result = DemoData.allResults[index];
                return DemoImageCard(result: result);
              },
            ),
            const SizedBox(height: 26),
            const SectionHeading('About This Demo'),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(color: Theme.of(context).colorScheme.surface, borderRadius: BorderRadius.circular(20)),
              child: const Row(children: [Icon(Icons.info_outline, color: AppTheme.primary), SizedBox(width: 12), Expanded(child: Text('This is a prototype simulation. Results are preconfigured for demonstration. The full ML pipeline will replace this simulation when the model is ready.', style: TextStyle(color: Colors.grey, height: 1.4)))]),
            ),
          ],
        ),
      ),
    );
  }
}

class DemoImageCard extends StatelessWidget {
  final DemoResult result;
  const DemoImageCard({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => AnalysisScreen(result: result))),
      borderRadius: BorderRadius.circular(20),
      child: Container(
        decoration: BoxDecoration(color: Theme.of(context).colorScheme.surface, borderRadius: BorderRadius.circular(20), border: Border.all(color: result.isHealthy ? Colors.green : AppTheme.primary.withOpacity(.2), width: 1.5)),
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          Container(width: 80, height: 80, decoration: BoxDecoration(color: result.isHealthy ? Colors.green.withOpacity(.15) : AppTheme.primary.withOpacity(.1), shape: BoxShape.circle), child: Icon(result.isHealthy ? Icons.check_circle_rounded : Icons.warning_amber_rounded, color: result.isHealthy ? Colors.green : AppTheme.primary, size: 40)),
          const SizedBox(height: 12),
          Text(result.disease, style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 14)),
          const SizedBox(height: 4),
          Text('${(result.confidence * 100).round()}% confidence', style: TextStyle(color: Colors.grey[600], fontSize: 12)),
          const SizedBox(height: 8),
          Text(result.severity, style: TextStyle(color: result.severity == 'None' ? Colors.green : result.severity == 'High' ? Colors.red : Colors.orange, fontSize: 12, fontWeight: FontWeight.w700)),
        ]),
      ),
    );
  }
}

// ─── ANALYSIS SCREEN (simulates inference) ────────────────────────────────────
class AnalysisScreen extends StatefulWidget {
  final DemoResult result;
  const AnalysisScreen({super.key, required this.result});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen> with TickerProviderStateMixin {
  late AnimationController _animController;
  late Animation<double> _progressAnim;
  double _progress = 0.0;
  bool _complete = false;

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(vsync: this, duration: const Duration(seconds: 2));
    _progressAnim = Tween<double>(begin: 0, end: 1).animate(CurvedAnimation(parent: _animController, curve: const Interval(0, 0.8, curve: Curves.easeInOut)));
    _animController.addListener(() { setState(() => _progress = _progressAnim.value); });
    _animController.addStatusListener((status) {
      if (status == AnimationStatus.completed) setState(() => _complete = true);
    });
    _animController.forward();
  }

  @override
  void dispose() { _animController.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    if (_complete) {
      return ResultScreen(result: widget.result);
    }
    return Scaffold(
      backgroundColor: AppTheme.darkBg,
      body: SafeArea(
        child: Column(children: [
          const SizedBox(height: 20),
          Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
            IconButton(onPressed: () => Navigator.pop(context), icon: const Icon(Icons.arrow_back, color: Colors.white)),
            const Text('CROPGuard AI', style: TextStyle(color: Colors.white70, fontSize: 14)),
            const SizedBox(width: 48),
          ]),
          const SizedBox(height: 30),
          Text('Analyzing ${widget.result.disease}...', style: const TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.w700)),
          const SizedBox(height: 30),
          Container(width: 200, height: 200, decoration: BoxDecoration(color: Colors.black.withOpacity(.3), shape: BoxShape.circle, border: Border.all(color: AppTheme.primary.withOpacity(.3), width: 2)), child: Center(child: ClipOval(child: Image.asset('assets/images/${widget.result.imageAsset}', fit: BoxFit.cover)))),
          const SizedBox(height: 30),
          Text('${(_progress * 100).round()}%', style: const TextStyle(color: AppTheme.primary, fontSize: 48, fontWeight: FontWeight.w900)),
          const SizedBox(height: 8),
          SizedBox(width: 200, height: 8, child: ClipRRect(borderRadius: BorderRadius.circular(4), child: LinearProgressIndicator(value: _progress, backgroundColor: Colors.grey[800], valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.primary), minHeight: 8))),
          const SizedBox(height: 12),
          const Text('Running detection pipeline...', style: TextStyle(color: Colors.white54, fontSize: 14)),
          const SizedBox(height: 6),
          const Text('• Disease Classification', style: TextStyle(color: Colors.white38, fontSize: 12)),
          const Text('• Segmentation Mask', style: TextStyle(color: Colors.white38, fontSize: 12)),
          const Text('• LayerCAM Explainability', style: TextStyle(color: Colors.white38, fontSize: 12)),
          const Text('• Severity Estimation', style: TextStyle(color: Colors.white38, fontSize: 12)),
        ]),
      ),
    );
  }
}

// ─── RESULT SCREEN ────────────────────────────────────────────────────────────
class ResultScreen extends StatelessWidget {
  final DemoResult result;
  const ResultScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final statusColor = result.isHealthy ? Colors.green : (result.severity == 'High' ? Colors.red : Colors.orange);
    return Scaffold(
      appBar: AppBar(title: const Text('Analysis Result'), backgroundColor: AppTheme.primary, foregroundColor: Colors.white),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Container(
            padding: const EdgeInsets.all(22),
            decoration: BoxDecoration(color: statusColor.withOpacity(.10), borderRadius: BorderRadius.circular(24)),
            child: Column(children: [
              Icon(result.isHealthy ? Icons.check_circle_rounded : Icons.warning_amber_rounded, color: statusColor, size: 54),
              const SizedBox(height: 10),
              Text(result.disease, textAlign: TextAlign.center, style: const TextStyle(fontSize: 26, fontWeight: FontWeight.w800)),
              const SizedBox(height: 6),
              Text('${(result.confidence * 100).round()}% confidence', style: const TextStyle(fontSize: 16, color: Colors.grey)),
            ]),
          ),
          const SizedBox(height: 18),
          _buildDetailRow('Disease', result.disease, Icons.bug_report),
          _buildDetailRow('Confidence', '${(result.confidence * 100).round()}%', Icons.assessment),
          _buildDetailRow('Severity', result.severity, Icons.speed),
          _buildDetailRow('Affected Area', '${result.affectedArea}%', Icons.crop_square),
          _buildDetailRow('Detected Regions', '${result.detectedRegions}', Icons.detect_important),
          const SizedBox(height: 20),
          const SectionHeading('Full Analysis'),
          const SizedBox(height: 12),
          _buildDetailCard('Severity Assessment', '${result.severityDescription}\n\nAffected area: ${result.affectedArea}% of leaf surface.', Icons.speed),
          _buildDetailCard('Symptoms Detected', result.symptoms, Icons.visibility),
          _buildDetailCard('Recommended Actions', result.recommendedActions, Icons.health_and_safety),
          _buildDetailCard('AI Explanation', result.explanation, Icons.auto_awesome),
          const SizedBox(height: 10),
          // Navigation to sub-screens
          _buildSectionButton('Detection', 'View ${result.detectedRegions} detected regions', Icons.search, () => Navigator.push(context, MaterialPageRoute(builder: (_) => DetectionScreen(result: result)))),
          _buildSectionButton('Segmentation', 'View lesion mask overlay', Icons.image, () => Navigator.push(context, MaterialPageRoute(builder: (_) => SegmentationScreen(result: result)))),
          _buildSectionButton('Explainability', 'View LayerCAM heatmap', Icons.auto_awesome, () => Navigator.push(context, MaterialPageRoute(builder: (_) => ExplainabilityScreen(result: result)))),
          _buildSectionButton('Severity', 'View severity breakdown', Icons.bar_chart, () => Navigator.push(context, MaterialPageRoute(builder: (_) => SeverityScreen(result: result)))),
        ],
      ),
    );
  }

  Widget _buildDetailRow(String label, String value, IconData icon) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(color: Theme.of(context).colorScheme.surface, borderRadius: BorderRadius.circular(20)),
      child: Row(children: [
        Icon(icon, color: AppTheme.primary, size: 28),
        const SizedBox(width: 16),
        Expanded(child: Text(label, style: TextStyle(color: Colors.grey[600], fontSize: 14))),
        Text(value, style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 16)),
      ]),
    );
  }

  Widget _buildDetailCard(String title, String content, IconData icon) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(color: Theme.of(context).colorScheme.surface, borderRadius: BorderRadius.circular(20)),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [Icon(icon, color: AppTheme.primary), const SizedBox(width: 9), Text(title, style: const TextStyle(fontWeight: FontWeight.w800))]),
        const SizedBox(height: 10),
        Text(content, style: const TextStyle(height: 1.45)),
      ]),
    );
  }

  Widget _buildSectionButton(String title, String subtitle, IconData icon, VoidCallback onTap) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(20),
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(color: Theme.of(context).colorScheme.surface, borderRadius: BorderRadius.circular(20), border: Border.all(color: AppTheme.primary.withOpacity(.15))),
        child: Row(children: [
          Container(padding: const EdgeInsets.all(10), decoration: BoxDecoration(color: AppTheme.primary.withOpacity(.1), borderRadius: BorderRadius.circular(12)), child: Icon(icon, color: AppTheme.primary)),
          const SizedBox(width: 14),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(title, style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 16)), const SizedBox(height: 2), Text(subtitle, style: TextStyle(color: Colors.grey[600], fontSize: 13))])),
          const Icon(Icons.chevron_right),
        ]),
      ),
    );
  }
}

// ─── DETECTION SCREEN ─────────────────────────────────────────────────────────
class DetectionScreen extends StatelessWidget {
  final DemoResult result;
  const DetectionScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Detection'), backgroundColor: AppTheme.primary, foregroundColor: Colors.white),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          const SectionHeading('Detection Regions'),
          const SizedBox(height: 12),
          Text('Found ${result.detectedRegions} region${result.detectedRegions != 1 ? 's' : ''}', style: const TextStyle(color: Colors.grey)),
          const SizedBox(height: 16),
          Container(
            height: 300,
            decoration: BoxDecoration(color: Theme.of(context).colorScheme.surface, borderRadius: BorderRadius.circular(24)),
            clipBehavior: Clip.antiAlias,
            child: Stack(children: [
              Image.asset('assets/images/${result.imageAsset}', fit: BoxFit.cover, width: double.infinity),
              if (result.maskAsset != null) ...[
                Positioned.fill(child: Image.asset('assets/masks/${result.maskAsset}', fit: BoxFit.contain, color: Color(0xFF44FF44).withOpacity(0.4))),
              ],
              if (result.detectedRegions > 0)
                ...List.generate(result.detectedRegions, (i) {
                  final boxes = [
                    (120, 100, 240, 170),
                    (220, 150, 320, 230),
                    (130, 120, 230, 190),
                    (150, 80, 200, 130),
                    (140, 130, 170, 160),
                  ];
                  if (i < boxes.length) {
                    final b = boxes[i];
                    return Positioned(
                      left: b.$1.toDouble(), top: b.$2.toDouble(),
                      child: Container(
                        width: (b.$3 - b.$1).toDouble(), height: (b.$4 - b.$2).toDouble(),
                        decoration: BoxDecoration(border: Border.all(color: Colors.red, width: 3), borderRadius: BorderRadius.circular(4)),
                        child: Container(padding: const EdgeInsets.all(4), decoration: BoxDecoration(color: Colors.red.withOpacity(.8), borderRadius: BorderRadius.circular(4)), child: Text('Region ${i + 1}', style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.w700))),
                      ),
                    );
                  }
                  return const SizedBox.shrink();
                }),
            ]),
          ),
          const SizedBox(height: 16),
          _buildRegionList(),
        ],
      ),
    );
  }

  Widget _buildRegionList() {
    final regions = result.isHealthy
        ? [{'name': 'No lesions detected', 'confidence': 97.6}]
        : List.generate(result.detectedRegions, (i) {
            final confs = [91.2, 87.0];
            return {'name': 'Region ${i + 1}', 'confidence': confs[i % confs.length]};
          });
    return ...regions.map((r) => Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Theme.of(context).colorScheme.surface, borderRadius: BorderRadius.circular(16)),
      child: Row(children: [
        Container(width: 12, height: 12, decoration: const BoxDecoration(shape: BoxShape.circle, color: Colors.red)),
        const SizedBox(width: 12),
        Expanded(child: Text(r['name']!, style: const TextStyle(fontWeight: FontWeight.w700))),
        Text('${r['confidence']}%', style: const TextStyle(fontWeight: FontWeight.w800)),
      ]),
    ));
  }
}

// ─── SEGMENTATION SCREEN ──────────────────────────────────────────────────────
class SegmentationScreen extends StatelessWidget {
  final DemoResult result;
  const SegmentationScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Segmentation'), backgroundColor: AppTheme.primary, foregroundColor: Colors.white),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          const SectionHeading('Lesion Mask Overlay'),
          const SizedBox(height: 12),
          Text('Segmentation mask shows detected lesion areas (green overlay).', style: TextStyle(color: Colors.grey[600])),
          const SizedBox(height: 16),
          Container(
            height: 350,
            decoration: BoxDecoration(color: Theme.of(context).colorScheme.surface, borderRadius: BorderRadius.circular(24)),
            clipBehavior: Clip.antiAlias,
            child: Stack(children: [
              Image.asset('assets/images/${result.imageAsset}', fit: BoxFit.cover, width: double.infinity),
              if (result.maskAsset != null && result.maskAsset != 'empty_mask.png')
                Positioned.fill(child: Image.asset('assets/masks/${result.maskAsset}', fit: BoxFit.cover, color: Color(0xFF44FF44).withOpacity(0.35))),
              if (result.maskAsset == 'empty_mask.png' || result.isHealthy)
                const Center(child: Icon(Icons.check_circle_rounded, size: 60, color: Colors.green)),
            ]),
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(color: Theme.of(context).colorScheme.surface, borderRadius: BorderRadius.circular(20)),
            child: Column(children: [
              const Text('Mask Summary', style: TextStyle(fontWeight: FontWeight.w800, fontSize: 18)),
              const SizedBox(height: 12),
              if (result.isHealthy)
                const Text('No lesion areas detected. The plant appears healthy.', style: TextStyle(color: Colors.green)),
              if (!result.isHealthy)
                Text('Detected ${result.detectedRegions} lesion region(s) covering ${result.affectedArea}% of the leaf surface.', style: const TextStyle(height: 1.4)),
            ]),
          ),
        ],
      ),
    );
  }
}

// ─── EXPLAINABILITY SCREEN ────────────────────────────────────────────────────
class ExplainabilityScreen extends StatelessWidget {
  final DemoResult result;
  const ExplainabilityScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Explainability'), backgroundColor: AppTheme.primary, foregroundColor: Colors.white),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          const SectionHeading('LayerCAM Heatmap'),
          const SizedBox(height: 12),
          Text('The heatmap highlights regions the model focused on for its prediction.', style: TextStyle(color: Colors.grey[600])),
          const SizedBox(height: 16),
          Container(
            height: 350,
            decoration: BoxDecoration(color: Theme.of(context).colorScheme.surface, borderRadius: BorderRadius.circular(24)),
            clipBehavior: Clip.antiAlias,
            child: Stack(children: [
              Image.asset('assets/images/${result.imageAsset}', fit: BoxFit.cover, width: double.infinity),
              if (result.heatmapAsset != null && result.heatmapAsset != 'empty_heatmap.png')
                Positioned.fill(child: Image.asset('assets/heatmaps/${result.heatmapAsset}', fit: BoxFit.cover, color: Color(0xFFFF4444).withOpacity(0.4))),
              if (result.heatmapAsset == 'empty_heatmap.png')
                const Center(child: Icon(Icons.check_circle_rounded, size: 60, color: Colors.green)),
            ]),
          ),
          const SizedBox(height: 16),
          _buildDetailCard('How LayerCAM Works', 'LayerCAM (Layer-wise Class Activation Mapping) generates a heatmap that highlights the regions in the input image that are most important for the model\'s classification decision. Red areas indicate high activation, meaning the model focused heavily on those regions.', Icons.lightbulb),
          _buildDetailCard('Model Focus Areas', result.isHealthy ? 'The model found no significant patterns indicating disease.' : 'The heatmap highlights the lesion areas, confirming the model correctly identified the diseased tissue rather than background features.', Icons.auto_awesome),
          _buildDetailCard('Confidence Calibration', '${(result.confidence * 100).round()}% confidence indicates the model is highly certain in its classification.', Icons.assessment),
        ],
      ),
    );
  }

  Widget _buildDetailCard(String title, String content, IconData icon) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(color: Theme.of(context).colorScheme.surface, borderRadius: BorderRadius.circular(20)),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [Icon(icon, color: AppTheme.primary), const SizedBox(width: 9), Text(title, style: const TextStyle(fontWeight: FontWeight.w800))]),
        const SizedBox(height: 10),
        Text(content, style: const TextStyle(height: 1.45)),
      ]),
    );
  }
}

// ─── SEVERITY SCREEN ──────────────────────────────────────────────────────────
class SeverityScreen extends StatelessWidget {
  final DemoResult result;
  const SeverityScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Severity'), backgroundColor: AppTheme.primary, foregroundColor: Colors.white),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          const SectionHeading('Severity Assessment'),
          const SizedBox(height: 12),
          const Text('Estimated affected area by severity level'),
          const SizedBox(height: 24),
          _buildSeverityBar('Low', 30, result.severity == 'None' || result.severity == 'Low', result.affectedArea < 10),
          _buildSeverityBar('Moderate', 60, result.severity == 'Moderate', result.affectedArea >= 10 && result.affectedArea < 25),
          _buildSeverityBar('High', 100, result.severity == 'High', result.affectedArea >= 25),
          const SizedBox(height: 24),
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(color: Theme.of(context).colorScheme.surface, borderRadius: BorderRadius.circular(24)),
            child: Column(children: [
              const Text('Current Assessment', style: TextStyle(fontWeight: FontWeight.w800, fontSize: 18)),
              const SizedBox(height: 16),
              Text(result.severity, style: TextStyle(fontSize: 36, fontWeight: FontWeight.w900, color: result.severity == 'High' ? Colors.red : result.severity == 'Moderate' ? Colors.orange : Colors.green)),
              const SizedBox(height: 8),
              Text('Estimated affected area: ${result.affectedArea}%', style: const TextStyle(fontSize: 16, color: Colors.grey)),
              const SizedBox(height: 12),
              Text(result.severityDescription, style: TextStyle(height: 1.4, color: Colors.grey[700])),
            ]),
          ),
          const SizedBox(height: 16),
          if (!result.isHealthy) ...[
            _buildDetailCard('Severity Criteria', 'Low: <10% leaf area affected\nModerate: 10-25% leaf area affected\nHigh: >25% leaf area affected', Icons.info),
          ] else ...[
            const Text('No disease detected. Plant is healthy and shows no signs of severity.', style: TextStyle(color: Colors.green, height: 1.4)),
          ],
        ],
      ),
    );
  }

  Widget _buildSeverityBar(String level, double widthPercent, bool isActive, bool matches) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Theme.of(context).colorScheme.surface, borderRadius: BorderRadius.circular(16), border: Border.all(color: isActive ? AppTheme.primary : Colors.transparent, width: 2)),
      child: Column(children: [
        Row(children: [
          Text(level, style: TextStyle(fontWeight: FontWeight.w800, fontSize: 16, color: isActive ? AppTheme.primary : Colors.grey)),
          const Spacer(),
          Text(matches ? '◀ Current' : '', style: const TextStyle(color: AppTheme.primary, fontSize: 12)),
        ]),
        const SizedBox(height: 8),
        ClipRRect(borderRadius: BorderRadius.circular(8), child: SizedBox(height: 20, child: Row(children: [
          for (int i = 0; i < (widthPercent / 10).floor(); i++)
            Container(width: 8, height: 20, color: matches ? AppTheme.primary : Colors.grey[300]),
          for (int i = 0; i < 10 - (widthPercent / 10).floor(); i++)
            Container(width: 8, height: 20, color: Colors.grey[200]),
        ]))),
      ]),
    );
  }

  Widget _buildDetailCard(String title, String content, IconData icon) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(color: Theme.of(context).colorScheme.surface, borderRadius: BorderRadius.circular(20)),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [Icon(icon, color: AppTheme.primary), const SizedBox(width: 9), Text(title, style: const TextStyle(fontWeight: FontWeight.w800))]),
        const SizedBox(height: 10),
        Text(content, style: const TextStyle(height: 1.45)),
      ]),
    );
  }
}

// ─── SECTION HEADING ──────────────────────────────────────────────────────────
class SectionHeading extends StatelessWidget {
  final String text;
  const SectionHeading(this.text, {super.key});

  @override
  Widget build(BuildContext context) {
    return Text(text, style: const TextStyle(fontSize: 19, fontWeight: FontWeight.w800));
  }
}
