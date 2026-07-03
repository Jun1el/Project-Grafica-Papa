import 'dart:typed_data';

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

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Theme.of(context).colorScheme.secondaryContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              prediction.label,
              style: Theme.of(
                context,
              ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 4),
            Text('Confianza: ${_percentage(prediction.confidence)}'),
            const Divider(height: 24),
            for (var index = 0; index < modelClassLabels.length; index++)
              Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(
                  children: <Widget>[
                    Expanded(child: Text(modelClassLabels[index])),
                    Text(_percentage(prediction.probabilities[index])),
                  ],
                ),
              ),
            Text(
              'Inferencia: ${prediction.elapsed.inMilliseconds} ms',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}
