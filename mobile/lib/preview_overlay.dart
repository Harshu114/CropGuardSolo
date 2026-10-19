import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

import 'demo_data.dart' show LiveResult, LiveResultCard;
import 'main.dart' show XFile;


/// The root overlay that runs the whole inference + preview pipeline from one
/// place.  It:
///  - shows a short progress screen while the model runs,
///  - renders the original photo,
///  - draws the model's mask and CAM on top so the user sees what the model
///    actually saw, and
///  - surfaces the postprediction card (LiveResultCard) on top of everything.
///
/// Use this at the top of [CropGuardApp]'s home page so the farmer's photo
/// and the model's output share the same frame.  Keep the overlay small — it
/// should only ever be visible while a prediction is in flight or right before
/// the final card comes up.
class CropGuardOverlay extends StatelessWidget {
  final LiveResult? result;
  final XFile? pending;
  final bool busy;

  const CropGuardOverlay({
    super.key,
    this.result,
    this.pending,
    this.busy = false,
  });

  @override
  Widget build(BuildContext context) {
    final showProgress = busy || pending == null;
    final showResult = result != null && !busy;

    final child = <Widget>[
      // The original photo.
      if (pending != null)
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: Image.file(
              File(pending!.path),
              width: 200,
              height: 200,
              fit: BoxFit.cover,
            ),
          ),
        ),

      // The model's output: mask + CAM.
      if (showResult && result!.maskBytes != null)
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: _ModelOutputCard(
            maskBytes: result!.maskBytes!,
            camBytes: result!.camBytes,
            origW: result!.origW,
            origH: result!.origH,
            label: result!.classification.cls,
            showMask: result!.segmentation.maskPngUrl.isNotEmpty,
            showCam: result!.segmentation.camPngUrl != null,
          ),
        ),
    ];

    if (showProgress) {
      child.add(_progressCard(context));
    }

    if (showResult) {
      child.add(LiveResultCard(
        result: result!,
        onDismiss: () => Navigator.maybePop(context),
      ));
    }

    return Stack(
      children: child,
    );
  }

  Widget _progressCard(BuildContext context) {
    return Center(
      child: Card(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 18),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 16),
              Text(
                'Analyzing leaf...',
                style: Theme.of(context).textTheme.titleMedium,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// A card that shows the model's mask and CAM on top of the original photo.
/// The two overlays are rendered at the same size as the original so the user
/// can see exactly what region the model focused on.
class _ModelOutputCard extends StatelessWidget {
  final Uint8List maskBytes;
  final Uint8List? camBytes;
  final int origW;
  final int origH;
  final String label;
  final bool showMask;
  final bool showCam;

  const _ModelOutputCard({
    required this.maskBytes,
    required this.camBytes,
    required this.origW,
    required this.origH,
    required this.label,
    required this.showMask,
    required this.showCam,
  });

  @override
  Widget build(BuildContext context) {
    final img.Image? maskImg = img.decodeImage(maskBytes);
    final img.Image? camImg = camBytes != null ? img.decodeImage(camBytes!) : null;

    if (maskImg == null) {
      return const SizedBox.shrink();
    }

    final double w = (maskImg.width / origW * 200).clamp(80.0, 200.0);
    final double h = (maskImg.height / origH * 200).clamp(80.0, 200.0);

    return Card(
      child: Column(
        children: [
          Text(label, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          SizedBox(
            width: w,
            height: h,
            child: Stack(children: [
              // The original.
              if (pending != null)
                Image.file(File(pending!.path), width: w, height: h, fit: BoxFit.cover),

              // Mask overlay.
              if (showMask && maskImg != null)
                CustomPaint(
                  painter: _MaskPainter(maskImg, color: Colors.blue.withAlpha(115)),
                  size: Size(w, h),
                ),

              // CAM overlay.
              if (showCam && camImg != null)
                CustomPaint(
                  painter: _CamPainter(camImg, color: Colors.redAccent.withAlpha(140)),
                  size: Size(w, h),
                ),
            ]),
          ),

          const SizedBox(height: 8),
          Row(
            children: [
              const Icon(Icons.visibility, size: 16),
              const SizedBox(width: 6),
              Text(maskImg.width.toString() + '×' + maskImg.height.toString()),
              const Spacer(),
              if (showCam)
                const Icon(Icons.auto_awesome, size: 16),
              if (showCam)
                const SizedBox(width: 6),
              if (showCam)
                Text(camImg!.width.toString() + '×' + camImg.height.toString()),
            ],
          ),
        ],
      ),
    );
  }
}

class _MaskPainter extends CustomPainter {
  final img.Image src;
  final Color color;

  _MaskPainter(this.src, {required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    final int w = src.width;
    final int h = src.height;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        final img.Pixel pixel = src.getPixel(x, y);
        final int r = (pixel.r * 255).toInt();
        final int g = (pixel.g * 255).toInt();
        final int b = (pixel.b * 255).toInt();
        final int lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).toInt();
        if (lum > 10) {
          canvas.drawRect(
            Rect.fromLTWH(x / w * size.width, y / h * size.height,
                (1 / w) * size.width, (1 / h) * size.height),
            paint,
          );
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

class _CamPainter extends CustomPainter {
  final img.Image src;
  final Color color;

  _CamPainter(this.src, {required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill
      ..blendMode = BlendMode.srcOver;

    final int w = src.width;
    final int h = src.height;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        final img.Pixel pixel = src.getPixel(x, y);
        final int r = (pixel.r * 255).toInt();
        final int g = (pixel.g * 255).toInt();
        final int b = (pixel.b * 255).toInt();
        final int lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).toInt();
        if (lum > 60) {
          canvas.drawRect(
            Rect.fromLTWH(x / w * size.width, y / h * size.height,
                (1 / w) * size.width, (1 / h) * size.height),
            paint,
          );
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
