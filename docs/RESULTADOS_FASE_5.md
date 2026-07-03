# Resultados de la Fase 5: Fine-tuning

Esta fase evaluó el impacto de descongelar los últimos bloques convolucionales de la arquitectura base (`ResNet50`) y reentrenar el modelo con una tasa de aprendizaje muy reducida (`1e-5`) durante 10 épocas adicionales. 
El objetivo era permitir que los detectores de características visuales profundas se adaptaran específicamente a enfermedades en hojas de papa, buscando mejorar la sólida línea base que ya habíamos establecido en la Fase 2.

## Comparación de Rendimiento (Set de Prueba - Laboratorio)

Al comparar las métricas frente al mismo set oculto (`test`) de 323 imágenes, observamos mejoras en todos los aspectos:

| Métrica | Línea Base (Fase 2) | Fine-tuned (Fase 5) | Mejora |
|---|---:|---:|---:|
| Accuracy global | 97.83 % | **99.38 %** | **+ 1.55 %** |
| Macro F1 | 96.65 % | **98.39 %** | **+ 1.74 %** |

## Matriz de Confusión Final

Filas = Clase Real, Columnas = Predicción
Orden: `[Tizón temprano, Tizón tardío, Hoja sana]`

```text
[150,   0,   0]
[  0, 148,   2]
[  0,   0,  23]
```

## Desempeño por Clase

| Clase | Precision | Recall | F1 Score | Soporte |
|---|---:|---:|---:|---:|
| Tizón temprano | 100.0 % | 100.0 % | 100.0 % | 150 |
| Tizón tardío | 100.0 % | 98.67 % | 99.33 % | 150 |
| Hoja sana | 92.00 % | 100.0 % | 95.83 % | 23 |

## Conclusiones para cerrar la fase

El proceso de fine-tuning fue un **éxito rotundo**. El modelo logró la perfección (100% de exactitud) al diagnosticar el Tizón Temprano. Para la clase sana (que era la más difícil por tener solo 23 ejemplos en validación), el modelo alcanzó un 100% de Recall (no se le escapó ninguna hoja sana real).
El único error residual de todo el modelo es confundir 2 hojas de Tizón tardío como si fueran sanas.

Con estos resultados, el modelo `modelo_papas_finetuned.h5` se convierte en nuestro mejor candidato para producción, dando la **Fase 5 por completada al 100%**.
