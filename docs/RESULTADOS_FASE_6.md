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

## Conclusión

El modelo `modelo_papas.tflite` cumple con todos los criterios de aceptación para uso en celulares:
1. Tamaño menor a 25 MB.
2. Predicciones virtualmente idénticas a la red neuronal original.
3. Compatibilidad con preprocesamiento RGB.

La Fase 6 se da por completada y el modelo resultante está listo para ser integrado en la aplicación Flutter en la Fase 7.
