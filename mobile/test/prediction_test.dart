import 'package:flutter_test/flutter_test.dart';
import 'package:papa_disease_mobile/src/prediction.dart';

void main() {
  test('preserva el orden de las tres clases', () {
    expect(modelClassIds, <String>[
      'Potato___Early_blight',
      'Potato___Late_blight',
      'Potato___healthy',
    ]);
  });

  test('devuelve la clase con mayor probabilidad sobre el umbral', () {
    final prediction = Prediction.fromOutput(<double>[
      0.12,
      0.81,
      0.07,
    ], const Duration(milliseconds: 40));
    expect(prediction.label, 'Tizón tardío');
    expect(prediction.confidence, 0.81);
    expect(prediction.isConclusive, isTrue);
  });

  test('marca como no concluyente una confianza menor a 0.70', () {
    final prediction = Prediction.fromOutput(<double>[
      0.20,
      0.69,
      0.11,
    ], Duration.zero);
    expect(prediction.label, 'Resultado no concluyente');
    expect(prediction.isConclusive, isFalse);
  });

  test('rechaza una salida que no contiene tres probabilidades', () {
    expect(
      () => Prediction.fromOutput(<double>[0.5, 0.5], Duration.zero),
      throwsArgumentError,
    );
  });
}
