import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'prediction.dart';

const int modelImageSize = 224;
const int maxImageBytes = 10 * 1024 * 1024;
const int maxImagePixels = 25000000;

class InferenceService {
  Interpreter? _interpreter;

  Future<void> load() async {
    if (_interpreter != null) return;
    final interpreter = await Interpreter.fromAsset(
      'assets/models/modelo_papas.tflite',
      options: InterpreterOptions()..threads = 2,
    );
    _validateContract(interpreter);
    _interpreter = interpreter;
  }

  Future<Prediction> predict(Uint8List bytes, String fileName) async {
    _validateFile(bytes, fileName);
    final interpreter = _interpreter;
    if (interpreter == null) {
      throw StateError('El modelo todavía no está cargado.');
    }

    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw const FormatException('No se pudo decodificar la imagen.');
    }
    if (decoded.width * decoded.height > maxImagePixels) {
      throw const FormatException(
        'La imagen supera el límite de 25 megapíxeles.',
      );
    }

    final oriented = img.bakeOrientation(decoded);
    final resized = img.copyResize(
      oriented,
      width: modelImageSize,
      height: modelImageSize,
      interpolation: img.Interpolation.linear,
    );
    final input = Float32List(modelImageSize * modelImageSize * 3);
    var offset = 0;
    for (final pixel in resized) {
      input[offset++] = pixel.r.toDouble();
      input[offset++] = pixel.g.toDouble();
      input[offset++] = pixel.b.toDouble();
    }

    final output = List<List<double>>.generate(
      1,
      (_) => List<double>.filled(modelClassLabels.length, 0),
    );
    final stopwatch = Stopwatch()..start();
    interpreter.run(
      input.reshape(<int>[1, modelImageSize, modelImageSize, 3]),
      output,
    );
    stopwatch.stop();
    return Prediction.fromOutput(output.single, stopwatch.elapsed);
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }

  static void _validateFile(Uint8List bytes, String fileName) {
    if (bytes.isEmpty) throw const FormatException('La imagen está vacía.');
    if (bytes.length > maxImageBytes) {
      throw const FormatException('La imagen supera el límite de 10 MB.');
    }
    final lowerName = fileName.toLowerCase();
    final allowedExtension =
        lowerName.endsWith('.jpg') ||
        lowerName.endsWith('.jpeg') ||
        lowerName.endsWith('.png');
    final isJpeg =
        bytes.length >= 3 &&
        bytes[0] == 0xff &&
        bytes[1] == 0xd8 &&
        bytes[2] == 0xff;
    final isPng =
        bytes.length >= 8 &&
        bytes[0] == 0x89 &&
        bytes[1] == 0x50 &&
        bytes[2] == 0x4e &&
        bytes[3] == 0x47;
    if (!allowedExtension || (!isJpeg && !isPng)) {
      throw const FormatException('Selecciona una imagen JPG o PNG válida.');
    }
  }

  static void _validateContract(Interpreter interpreter) {
    final input = interpreter.getInputTensor(0);
    final output = interpreter.getOutputTensor(0);
    if (input.shape.join(',') != '1,224,224,3' ||
        input.type != TensorType.float32) {
      interpreter.close();
      throw StateError('El modelo no cumple la entrada [1,224,224,3] float32.');
    }
    if (output.shape.join(',') != '1,3' || output.type != TensorType.float32) {
      interpreter.close();
      throw StateError('El modelo no cumple la salida [1,3] float32.');
    }
  }
}
