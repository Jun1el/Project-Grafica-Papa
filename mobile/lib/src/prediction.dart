const double uncertaintyThreshold = 0.70;

const List<String> modelClassIds = <String>[
  'Potato___Early_blight',
  'Potato___Late_blight',
  'Potato___healthy',
];

const List<String> modelClassLabels = <String>[
  'Tizón temprano',
  'Tizón tardío',
  'Hoja sana',
];

class Prediction {
  const Prediction({required this.probabilities, required this.elapsed});

  factory Prediction.fromOutput(List<double> output, Duration elapsed) {
    if (output.length != modelClassLabels.length) {
      throw ArgumentError(
        'El modelo debe devolver exactamente ${modelClassLabels.length} '
        'probabilidades.',
      );
    }
    if (output.any((value) => !value.isFinite || value < 0 || value > 1)) {
      throw ArgumentError('El modelo devolvió probabilidades inválidas.');
    }

    return Prediction(
      probabilities: List<double>.unmodifiable(output),
      elapsed: elapsed,
    );
  }

  final List<double> probabilities;
  final Duration elapsed;

  int get bestIndex {
    var index = 0;
    for (var i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[index]) {
        index = i;
      }
    }
    return index;
  }

  double get confidence => probabilities[bestIndex];
  bool get isConclusive => confidence >= uncertaintyThreshold;
  String get label =>
      isConclusive ? modelClassLabels[bestIndex] : 'Resultado no concluyente';
}
