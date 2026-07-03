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

## Limitacion de reproducibilidad aceptada

El archivo `modelo_papas_finetuned.h5` exacto que produjo el TFLite no fue
conservado. Por ese motivo no es posible repetir la conversion ni comprobar de
nuevo la equivalencia H5-TFLite. El script queda disponible para futuros modelos
que sí conserven ambos artefactos:

```bash
python -m scripts.verify_tflite --image ruta/hoja1.jpg --image ruta/hoja2.jpg
```

La ejecucion falla si cambia la clase o si cualquier probabilidad difiere mas
de 0.01. El equipo acepta la limitacion porque el TFLite versionado carga
correctamente y es el artefacto requerido por Flutter. ONNX queda fuera de
alcance porque la aplicacion Android de la fase 7 consumira TFLite.

## Conclusión

Los resultados registrados indican que `modelo_papas.tflite` satisface estos
criterios preliminares para uso en celulares:
1. Tamaño menor a 25 MB.
2. Predicciones virtualmente idénticas a la red neuronal original.
3. Compatibilidad con preprocesamiento RGB.

La fase 6 queda completada con la limitacion de reproducibilidad documentada. El
modelo TFLite esta listo para integrarse en Flutter durante la fase 7.
