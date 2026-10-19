import 'dart:async';
import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;

import 'api_client.dart';
import 'demo_data.dart' show DemoData, DemoImageCard, DemoResult;
import 'preview_overlay.dart' show CropGuardOverlay;


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
    scaffoldBackgroundColor: Color(0xFFF7F8F2),
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
      fillColor: Color(0xFF172019),
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(16), borderSide: BorderSide.none),
    ),
  );
}

// ─── LIVE RESULT MODEL ────────────────────────────────────────────────────────────
class LiveResult {
  final int imageId;
  final String imageFile;
  final Uint8List? maskBytes;
  final Uint8List? camBytes;
  final int origW;
  final int origH;
  final String apiBase;
  final Classification classification;
  final List<Detection> detections;
  final Segmentation segmentation;
  final Severity severity;
  final Map<String, int> timingsMs;
  final String modelVersion;

  LiveResult({
    this.imageId = 0,
    this.imageFile = '',
    this.maskBytes,
    this.camBytes,
    this.origW = 0,
    this.origH = 0,
    this.apiBase = 'http://10.0.2.2:8000',
    Classification? classification,
    List<Detection>? detections,
    Segmentation? segmentation,
    Severity? severity,
    Map<String, int>? timingsMs,
    String? modelVersion,
  }) : classification = classification ?? Classification.empty(),
       detections = detections ?? const [],
       segmentation = segmentation ?? Segmentation.empty(),
       severity = severity ?? Severity.empty(),
       timingsMs = timingsMs ?? const {},
       modelVersion = modelVersion ?? 'unknown';

  factory LiveResult.fromJson(Map<String, dynamic> j) {
    return LiveResult(
      imageId: j['image_id'] as int,
      imageFile: j['image_file'] as String? ?? '',
      maskBytes: j['maskBytes'] as Uint8List?,
      camBytes: j['camBytes'] as Uint8List?,
      origW: j['width'] as int,
      origH: j['height'] as int,
      apiBase: j['api_base'] as String? ?? 'http://10.0.2.2:8000',
      classification: Classification.fromJson(j['classification'] as Map<String, dynamic>),
      detections: (j['detections'] as List?)?.map((e) => Detection.fromJson(e as Map<String, dynamic>)).toList() ?? const [],
      segmentation: Segmentation.fromJson(j['segmentation'] as Map<String, dynamic>),
      severity: Severity.fromJson(j['severity'] as Map<String, dynamic>),
      timingsMs: Map<String, int>.from(j['timings_ms'] as Map? ?? {}),
      modelVersion: j['model_version'] as String? ?? 'unknown',
    );
  }

  double get leafFraction => segmentation.leafFraction;
  bool get severityReliable => severity.reliable;
  List<Detection> get detectionsSorted => List.unmodifiable(
    detections..sort((a, b) => b.score.compareTo(a.score)),
  );
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
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _picker = ImagePicker();
  XFile? _pending;
  bool _busy = false;
  LiveResult? _latest;

  Future<void> _pick() async {
    if (_busy) return;
    final picked = await _picker.pickImage(
      source: ImageSource.gallery,
      maxWidth: 1600.0,
      maxHeight: 1600.0,
      imageQuality: 92,
    );
    if (picked == null) return;

    setState(() {
      _pending = picked;
      _busy = true;
    });
  }

  Future<void> _run() async {
    final f = _pending;
    if (f == null || !mounted) return;

    setState(() { _busy = true; });

    try {
      final bytes = await f.readAsBytes();

      final res = await predictImage(
        rawBytes: bytes,
        apiBase: 'http://10.0.2.2:8000',
        conf: 0.15,
        withCam: true,
      );

      if (!mounted) return;

      final maskUrl = Uri.parse('http://10.0.2.2:8000/v1/masks/' + res.imageId + '.png');
      final camUrl = Uri.parse('http://10.0.2.2:8000/v1/cams/' + res.imageId + '.png');

      final maskBytes = await http.readBytes(maskUrl);
      final camBytes = await http.readBytes(camUrl);          if (!mounted) return;
      setState(() {
        _latest = LiveResult.fromJson(
          res.toJson()
            ..['maskBytes'] = maskBytes
            ..['camBytes'] = camBytes,
        );
      });
    } catch (e) {
      if (!mounted) return;
      _latest = null;
      if (mounted)
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Prediction failed: $e'),
            backgroundColor: AppTheme.error,
          ),
        );
    } finally {
      if (mounted) setState(() { _busy = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(20, 22, 20, 110),
          children: [
            Row(
              children: [
                const CircleAvatar(
                  radius: 25,
                  backgroundColor: Color(0xFFDCEFC8),
                  child: Icon(Icons.eco, color: AppTheme.primary, size: 29),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: const [
                      Text(
                        'Good morning 👋',
                        style: TextStyle(fontSize: 14, color: Colors.grey),
                      ),
                      SizedBox(height: 2),
                      Text(
                        'CROPGuard AI',
                        style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  onPressed: () {},
                  icon: const Icon(Icons.notifications_none_rounded),
                ),
              ],
            ),
            const SizedBox(height: 24),

            // ─── Upload + run area ─────────────────────────────────────────────────
            ElevatedButton.icon(
              onPressed: _busy ? null : _pick,
              icon: Icon(_pending == null ? Icons.upload : Icons.refresh),
              label: Text(_pending == null
                  ? 'Pick a leaf photo'
                  : '✓ ${_pending!.name}  •  tap to run'),
            ),
            const SizedBox(height: 12),

            // ─── Live result card ─────────────────────────────────────────────────
            if (_latest != null) ...[
              const SizedBox(height: 10),
              LiveResultCard(
                result: _latest!,
                onDismiss: () => setState(() { _latest = null; }),
              ),
            ],

            // ─── Demo cards ───────────────────────────────────────────────────────
            const SizedBox(height: 24),
            const SectionHeading('Select a Demo Image'),
            const SizedBox(height: 12),
            GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: DemoData.allResults.length,
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                crossAxisSpacing: 12,
                mainAxisSpacing: 12,
                childAspectRatio: 0.75,
              ),
              itemBuilder: (context, index) {
                final result = DemoData.allResults[index];
                return DemoImageCard(
                  result: result,
                  onRun: () {},
                );
              },
            ),

            const SizedBox(height: 26),
            const SectionHeading('About This App'),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surface,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                children: [
                  Icon(Icons.info_outline, color: AppTheme.primary),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      'CROPGuard gives you an AI crop-disease diagnosis from a single leaf photo. '
                      'Take a photo, and the model runs detection, classification, segmentation, '
                      'and an explanation heatmap — all in one pass. Behind the scenes we run the '
                      'CropGuard v5 multi-task model, trained on over 32,000 annotated tomato-leaf '
                      'images. CROPGuard was built for Indian farmers so the advice is practical '
                      'rather than textbook.',
                      style: TextStyle(
                        color: Colors.grey[600],
                        height: 1.4,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
