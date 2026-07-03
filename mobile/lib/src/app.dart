import 'dart:typed_data';

import 'package:flutter/foundation.dart' show kDebugMode;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'inference_service.dart';
import 'prediction.dart';

class PapaDiseaseApp extends StatelessWidget {
  const PapaDiseaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Papa Sana',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xff3f6b35),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final InferenceService _inference = InferenceService();
  final ImagePicker _picker = ImagePicker();
  Uint8List? _imageBytes;
  Prediction? _prediction;
  String? _error;
  bool _loadingModel = true;
  bool _analyzing = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      await _inference.load();
    } catch (error) {
      _error = 'No se pudo cargar el modelo: $error';
    }
    if (mounted) setState(() => _loadingModel = false);
  }

  Future<void> _pick(ImageSource source) async {
    setState(() {
      _error = null;
      _prediction = null;
    });
    try {
      final selected = await _picker.pickImage(
        source: source,
        imageQuality: 95,
        requestFullMetadata: false,
      );
      if (selected == null) return;
      final bytes = await selected.readAsBytes();
      setState(() {
        _imageBytes = bytes;
        _analyzing = true;
      });
      final prediction = await _inference.predict(bytes, selected.name);
      if (!mounted) return;
      setState(() => _prediction = prediction);
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = _friendlyError(error));
    } finally {
      if (mounted) setState(() => _analyzing = false);
    }
  }

  String _friendlyError(Object error) {
    if (error is FormatException) return error.message;
    return 'No se pudo analizar la imagen. Inténtalo nuevamente.';
  }

  void _reset() {
    setState(() {
      _imageBytes = null;
      _prediction = null;
      _error = null;
    });
  }

  @override
  void dispose() {
    _inference.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Papa Sana')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: <Widget>[
            Text(
              'Revisa una hoja de papa',
              style: Theme.of(
                context,
              ).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            const Text(
              'Toma una foto clara de una sola hoja o elige una imagen. '
              'El análisis se realiza en tu celular y no se envía a internet.',
            ),
            const SizedBox(height: 16),
            const Card(
              child: Padding(
                padding: EdgeInsets.all(14),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    Icon(Icons.info_outline),
                    SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Herramienta educativa. El resultado no sustituye '
                        'un diagnóstico agronómico.',
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            if (_imageBytes != null)
              ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: Image.memory(
                  _imageBytes!,
                  height: 260,
                  fit: BoxFit.cover,
                  semanticLabel: 'Hoja seleccionada',
                ),
              ),
            if (_imageBytes != null) const SizedBox(height: 16),
            if (_loadingModel || _analyzing) ...<Widget>[
              const Center(child: CircularProgressIndicator()),
              const SizedBox(height: 8),
              Center(
                child: Text(
                  _loadingModel ? 'Preparando modelo…' : 'Analizando hoja…',
                ),
              ),
            ] else if (_prediction != null)
              _ResultCard(prediction: _prediction!)
            else ...<Widget>[
              FilledButton.icon(
                onPressed: () => _pick(ImageSource.camera),
                icon: const Icon(Icons.photo_camera_outlined),
                label: const Text('Tomar foto'),
              ),
              const SizedBox(height: 10),
              OutlinedButton.icon(
                onPressed: () => _pick(ImageSource.gallery),
                icon: const Icon(Icons.photo_library_outlined),
                label: const Text('Elegir de galería'),
              ),
            ],
            if (_error != null) ...<Widget>[
              const SizedBox(height: 12),
              Text(
                _error!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ],
            if (_imageBytes != null || _error != null) ...<Widget>[
              const SizedBox(height: 12),
              TextButton(
                onPressed: _reset,
                child: const Text('Analizar otra imagen'),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.prediction});

  final Prediction prediction;

  String _percentage(double value) => '${(value * 100).toStringAsFixed(1)} %';

  /// Color semantico por clase: verde para hoja sana, ambar y rojo para las
  /// dos enfermedades. El orden coincide con [modelClassLabels].
  static Color _classColor(int index) {
    switch (index) {
      case 2:
        return const Color(0xff2e7d32); // Hoja sana
      case 0:
        return const Color(0xffef6c00); // Tizón temprano
      default:
        return const Color(0xffc62828); // Tizón tardío
    }
  }

  static IconData _classIcon(int index) =>
      index == 2 ? Icons.check_circle_outline : Icons.warning_amber_outlined;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final conclusive = prediction.isConclusive;
    final best = prediction.bestIndex;
    final accent = conclusive ? _classColor(best) : theme.colorScheme.tertiary;

    return Card(
      color: theme.colorScheme.surfaceContainerHighest,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Row(
              children: <Widget>[
                Icon(
                  conclusive ? _classIcon(best) : Icons.help_outline,
                  color: accent,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    prediction.label,
                    style: theme.textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: accent,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 4),
            if (conclusive)
              Text('Confianza: ${_percentage(prediction.confidence)}')
            else
              Text(
                'La confianza más alta fue ${_percentage(prediction.confidence)}, '
                'por debajo del 70 % necesario para un resultado.',
                style: theme.textTheme.bodyMedium,
              ),
            if (!conclusive) ...<Widget>[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: theme.colorScheme.tertiaryContainer,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    Icon(
                      Icons.tips_and_updates_outlined,
                      color: theme.colorScheme.onTertiaryContainer,
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'Probá otra foto: una sola hoja centrada, con buena '
                        'luz natural y fondo simple.',
                        style: TextStyle(
                          color: theme.colorScheme.onTertiaryContainer,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
            const Divider(height: 24),
            for (var index = 0; index < modelClassLabels.length; index++)
              _ProbabilityBar(
                label: modelClassLabels[index],
                percentage: _percentage(prediction.probabilities[index]),
                value: prediction.probabilities[index],
                color: _classColor(index),
                highlighted: index == best,
              ),
            if (kDebugMode) ...<Widget>[
              const SizedBox(height: 4),
              Text(
                'Inferencia: ${prediction.elapsed.inMilliseconds} ms',
                style: theme.textTheme.bodySmall,
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _ProbabilityBar extends StatelessWidget {
  const _ProbabilityBar({
    required this.label,
    required this.percentage,
    required this.value,
    required this.color,
    required this.highlighted,
  });

  final String label;
  final String percentage;
  final double value;
  final Color color;
  final bool highlighted;

  @override
  Widget build(BuildContext context) {
    final weight = highlighted ? FontWeight.bold : FontWeight.normal;
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: <Widget>[
              Text(label, style: TextStyle(fontWeight: weight)),
              Text(percentage, style: TextStyle(fontWeight: weight)),
            ],
          ),
          const SizedBox(height: 4),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(
              value: value.clamp(0.0, 1.0),
              minHeight: 8,
              backgroundColor: color.withValues(alpha: 0.15),
              valueColor: AlwaysStoppedAnimation<Color>(color),
            ),
          ),
        ],
      ),
    );
  }
}
