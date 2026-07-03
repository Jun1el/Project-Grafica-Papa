# Plan de desarrollo del proyecto

Este documento coordina el trabajo del equipo para avanzar sin mezclar
entrenamiento, evaluacion y despliegue. El sistema es una herramienta educativa;
no sustituye un diagnostico agronomico.

## Estado de las fases

| Fase | Estado | Resultado |
|---|---|---|
| 1. Consolidacion | Completada | Modulos reproducibles de datos, entrenamiento, segmentacion e inferencia |
| 2. Evaluacion confiable | Completada | Split 70/15/15, metricas por clase, matrices y comparacion Watershed |
| 3. Demo Hugging Face | Completada | Interfaz Gradio (`app.py`) para inferencia, sin entrenamiento |
| 4. Imagenes de campo | Completada | Evaluacion aceptada con caída de métricas documentada (domain shift) |
| 5. Fine-tuning | Completada | Macro F1 de 98.39 % alcanzado descongelando top layers de ResNet50 |
| 6. TFLite/ONNX | Completada | Modelo exportado a TFLite (22.8 MB) y validado para Android |
| 7. Flutter Android | Pendiente | Aplicacion offline con camara e inferencia local |

## Fase 2 completada

La ejecucion final en Colab con GPU fue completada el 1 de julio de 2026. El
resultado aprobado y las decisiones tecnicas se encuentran en
[`RESULTADOS_FASE_2.md`](RESULTADOS_FASE_2.md).

## Flujo reproducible de la fase 2 en Colab

No se debe evaluar el modelo historico como si perteneciera al nuevo conjunto de
prueba. Primero hay que reentrenar usando el manifiesto 70/15/15.

1. Clonar o actualizar el repositorio en Colab y activar GPU.
2. Instalar `requirements.txt` y descargar el dataset en `dataset_papa/`.
3. Crear una sola vez el reparto reproducible:

   ```bash
   python -m scripts.prepare_split
   ```

4. Entrenar con `train` y `validation`; este paso no utiliza `test`:

   ```bash
   python -m scripts.train
   ```

5. Evaluar `validation` y decidir el umbral de resultado no concluyente. El
   valor inicial es 0.70, pero debe justificarse con los resultados:

   ```bash
   python -m scripts.evaluate --split validation --threshold 0.70
   ```

6. Congelar el umbral elegido y ejecutar `test` una sola vez:

   ```bash
   python -m scripts.evaluate --split test --threshold 0.70
   ```

7. Conservar el modelo reentrenado, el manifiesto y los reportes de la misma
   ejecucion. Registrar fecha, responsable, commit y configuracion de Colab.

El modelo nuevo se llama `modelo_papas_resnet50_split70_15_15.h5`; el modelo
historico no se sobrescribe. Los reportes aparecen en `outputs/evaluation/`.

## Criterios para cerrar la fase 2

- El entrenamiento finaliza sin usar las imagenes de `test`.
- Se reportan accuracy, macro F1, precision, recall y F1 de cada clase.
- Se revisa especialmente el recall de `Potato___healthy`, por ser minoritaria.
- Se conservan las matrices de confusion con y sin Watershed.
- Watershed solo pasa al flujo principal si mejora resultados de forma clara.
- Las predicciones bajo el umbral se muestran como no concluyentes.
- El resultado final queda asociado al commit exacto usado en Colab.

## Fases posteriores

### Fase 3: Hugging Face Spaces

Crear una demo Gradio que cargue el modelo aprobado, valide JPG/PNG, muestre
clase, confianza, probabilidades y advertencia educativa. No guardar imagenes de
usuarios ni entrenar dentro de Spaces.

### Fase 4: imagenes de campo

Recolectar o conseguir imagenes autorizadas con luces, fondos, camaras y etapas
de enfermedad variadas. Mantener un conjunto de campo separado y verificar las
etiquetas con apoyo agronomico cuando sea posible.

El conjunto de campo vive en `dataset_campo/` (ignorado por Git) con una
subcarpeta por clase. La evaluacion no usa el manifiesto 70/15/15; se ejecuta
con:

```bash
python -m scripts.evaluate_field
```

El reporte se guarda en `outputs/evaluation/field_report.json` con matriz de
confusion, metricas por clase, cobertura del umbral y `macro_f1_present` (promedio
solo de las clases con muestras). El avance parcial esta en
[`RESULTADOS_FASE_4.md`](RESULTADOS_FASE_4.md).

Criterios para cerrar la fase 4:

- Las tres clases tienen imagenes de campo, incluida `Potato___healthy`.
- Se descartan duplicados frente al dataset de laboratorio (mismo criterio
  SHA-256 de `data.py`) para evitar fugas.
- Se reporta la caida de accuracy y macro F1 frente al test de laboratorio.
- Se documenta el origen y la licencia de cada fuente de imagenes.

### Fase 5: fine-tuning

Descongelar solo bloques superiores de ResNet50, usar una tasa de aprendizaje
menor y comparar contra la linea base mediante macro F1 y metricas por clase.

### Fase 6: TFLite y ONNX (Completada)
- **Objetivo**: Modelos ligeros validados contra TensorFlow.
- **Tareas**: Exportar primero TFLite para Android. Comparar etiquetas y probabilidades con el modelo TensorFlow, y medir tamaño, memoria y latencia antes de aceptar la conversión.

### Fase 7: Flutter Android

Implementar captura o seleccion de imagen, preprocesamiento e inferencia TFLite
sin conexion. Solicitar solo permisos necesarios y probar dispositivos de gama
baja y media antes de considerar iOS.

## Reglas de colaboracion

- Crear una rama por fase o cambio concreto; no trabajar directamente en `main`.
- Usar commits convencionales como `feat:`, `fix:`, `docs:` y `test:`.
- No versionar credenciales, datasets descargados, entornos virtuales ni fotos
  privadas.
- Ejecutar las pruebas antes de solicitar merge.
- No cambiar el orden de clases definido en `papa_disease/config.py`.
- No informar una metrica como final sin indicar modelo, split y commit.
