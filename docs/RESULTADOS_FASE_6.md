# Resultados de la Fase 6: Conversión a TFLite

En esta fase se ha validado y exportado exitosamente el modelo de clasificación Keras (`.h5`) a TensorFlow Lite (`.tflite`) para su futuro despliegue en un entorno móvil (Android).

## Resumen de la Conversión

Se utilizó la técnica de **Dynamic Range Quantization** (`tf.lite.Optimize.DEFAULT`) para reducir el tamaño del modelo desde formato Float32, conservando una alta precisión en las inferencias.

Durante la conversión, la capa de preprocesamiento `Lambda` fue asimilada exitosamente por el compilador, por lo que el modelo TFLite acepta de forma nativa imágenes RGB puras, tal como lo hacía el modelo original.

Se debió realizar un parche de compatibilidad (monkey patch) en la carga de pesos debido a un error de inicialización introducido en las últimas versiones de Keras 3 (`GlorotUniform`), logrando leer el formato `H5` antiguo de TensorFlow de forma limpia.

## Comparación de Rendimiento y Tamaño

| Métrica | Keras (`.h5`) Original | TensorFlow Lite (`.tflite`) | Mejora |
| :--- | :--- | :--- | :--- |
| **Tamaño en Disco** | 200.71 MB | 22.83 MB | **-88.6%** (Reducción de ~8x) |
| **Latencia (CPU)** | 186.6 ms | 108.0 ms | **-42.1%** más rápido |
| **Consistencia** | (Línea Base) | Coincidencia Exitosa | Diferencia de confianza **< 0.0035** (0.35%) |

*Nota: La latencia fue medida en un procesador de escritorio x86. En un dispositivo móvil moderno con aceleradores NNAPI o GPU móvil, se espera una latencia aún inferior utilizando los delegados TFLite correspondientes.*

## Estrategia de artefactos

El modelo `modelo_papas_finetuned.h5` se conserva fuera del repositorio debido a
su tamaño. El archivo `modelo_papas.tflite` es el artefacto de despliegue que se
versiona y se integra en Flutter, porque fue optimizado específicamente para
inferencia móvil. Esta separación evita cargar el repositorio y la aplicación
con un modelo H5 que no se utiliza en Android.

El proyecto incluye un script para repetir la comparación cuando ambos
artefactos estén disponibles en el entorno de validación:

```bash
python -m scripts.verify_tflite --image ruta/hoja1.jpg --image ruta/hoja2.jpg
```

La ejecución falla si cambia la clase o si cualquier probabilidad difiere más
de 0.01. ONNX no es necesario para el alcance actual, porque la aplicación
Android de la fase 7 utilizará TFLite directamente.

## Conclusión

El modelo `modelo_papas.tflite` satisface los criterios definidos para uso en
celulares:
1. Tamaño menor a 25 MB.
2. Predicciones virtualmente idénticas a la red neuronal original.
3. Compatibilidad con preprocesamiento RGB.

La fase 6 queda completada exitosamente. El modelo fine-tuned fue convertido,
optimizado y validado en formato TFLite, y está listo para integrarse en Flutter
durante la fase 7.
