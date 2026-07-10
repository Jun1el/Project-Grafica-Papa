_**UNIVERSIDAD NACIONAL DE INGENIERÍA  |  COMPUTACIÓN GRÁFICA**_ 

## **UNIVERSIDAD NACIONAL DE INGENIERÍA** 

Curso de Computación Gráfica 

## **PAPA SANA** 

_Clasificación de enfermedades en hojas de papa mediante visión computacional y aprendizaje profundo_ 

## **INFORME TÉCNICO DEL PROYECTO** 

**Integrantes** Edy Serrano Andrés La Torre Diego Orrego Lima, Perú — 2026 

Papa Sana  •  Informe técnico  •  Página 1 

_**UNIVERSIDAD NACIONAL DE INGENIERÍA  |  COMPUTACIÓN GRÁFICA**_ 

## **Resumen** 

Papa Sana es un sistema académico de visión computacional que clasifica fotografías de hojas de papa en tres categorías: tizón temprano, tizón tardío y hoja sana. El desarrollo integra preparación reproducible de datos,  transferencia  de  aprendizaje  con  ResNet50,  evaluación  cuantitativa,  experimentación  con segmentación HSV y Watershed, fine-tuning, conversión optimizada a TensorFlow Lite y una aplicación Android desarrollada en Flutter para inferencia completamente offline. 

El conjunto principal contiene 2 152 imágenes: 1 000 de tizón temprano, 1 000 de tizón tardío y 152 de hoja sana. El modelo base alcanzó 97.83 % de accuracy y 96.65 % de macro F1 en el conjunto de prueba. Tras el fine-tuning, obtuvo 99.38 % de accuracy y 98.39 % de macro F1 sobre el mismo test de 323 imágenes. La conversión a TFLite redujo el artefacto de 200.71 MB a 22.83 MB (88.6 %) y conservó la clase predicha con diferencias de probabilidad inferiores a 0.0035 en la validación documentada. 

**Resultado central. El proyecto demuestra una cadena completa desde datos hasta una aplicación móvil funcional. A la vez, documenta una caída a 37.87 % de accuracy en imágenes de campo, evidencia de cambio de dominio y razón para limitar el uso a fines educativos.** 

## **Palabras clave** 

Visión  computacional;  redes  neuronales  convolucionales;  ResNet50;  transferencia  de  aprendizaje; segmentación de imágenes; Watershed; TensorFlow Lite; Flutter; inferencia offline. 

## **Contenido** 

- 1. Introducción, motivación y formulación del problema 

- 2. Objetivos, alcance y restricciones 

- 3. Fundamentos técnicos y relación con Computación Gráfica 

- 4. Arquitectura y estructura del repositorio 

- 5. Desarrollo completo por fases 

- 6. Evaluación, resultados y discusión 

- 7. Aplicación móvil y despliegue offline 

- 8. Calidad, seguridad, limitaciones y trabajo futuro 

- 9. Conclusiones, contribuciones del equipo y referencias 

Papa Sana  •  Informe técnico  •  Página 2 

_**UNIVERSIDAD NACIONAL DE INGENIERÍA  |  COMPUTACIÓN GRÁFICA**_ 

## **1. Introducción** 

## **1.1 Contexto y motivación** 

La papa es un cultivo de importancia alimentaria y económica. Enfermedades foliares como el tizón temprano y el tizón tardío producen patrones visuales que pueden analizarse mediante técnicas de procesamiento digital de imágenes. Sin embargo, la identificación manual exige experiencia, puede variar entre observadores y no siempre está disponible cuando el agricultor necesita una orientación inicial. 

La  motivación  del  proyecto  es  estudiar,  desde  la  perspectiva  de  Computación  Gráfica  y  visión computacional,  cómo  una  imagen  digital  puede  transformarse  en  una  representación  útil  para clasificación. La solución no pretende sustituir a un especialista: ofrece una demostración educativa reproducible, rápida y ejecutable sin conexión a Internet. 

## **1.2 Formulación del problema** 

¿Cómo diseñar un flujo reproducible que reciba una fotografía de una hoja, preserve un contrato de entrada consistente, extraiga características visuales mediante una CNN y entregue una predicción comprensible en un teléfono Android, manteniendo privacidad y un tamaño de modelo compatible con uso offline? 

## **1.3 Justificación** 

- Justificación técnica: integra adquisición, decodificación, transformación geométrica, representación RGB, clasificación y despliegue móvil. 

- Justificación académica: permite comparar segmentación clásica con aprendizaje profundo y analizar métricas más allá de accuracy. 

- Justificación práctica: el formato TFLite elimina dependencia de una API, reduce latencia de red y mantiene las fotografías en el dispositivo. 

- Justificación ética: incorpora un umbral de incertidumbre y una advertencia explícita de uso educativo. 

## **2. Objetivos y alcance** 

## **2.1 Objetivo general** 

Desarrollar y validar un sistema de visión computacional capaz de clasificar hojas de papa en tizón temprano, tizón tardío u hoja sana, y desplegar el modelo en una aplicación Android offline mediante TensorFlow Lite. 

## **2.2 Objetivos específicos** 

- Preparar un conjunto de datos estratificado, reproducible y libre de duplicados exactos. 

- Construir un clasificador basado en ResNet50 con transferencia de aprendizaje y aumento de datos. 

- Evaluar accuracy, macro F1, métricas por clase, matrices de confusión y cobertura del umbral. 

- Experimentar con segmentación HSV y Watershed y medir su impacto real sobre la clasificación. 

- Aplicar fine-tuning a los bloques superiores para adaptar características profundas al dominio de hojas de papa. 

- Convertir y validar el modelo H5 en formato TFLite optimizado para Android. 

- Implementar una interfaz Flutter que procese cámara o galería y ejecute inferencia local. 

Papa Sana  •  Informe técnico  •  Página 3 

_**UNIVERSIDAD NACIONAL DE INGENIERÍA  |  COMPUTACIÓN GRÁFICA**_ 

## **2.3 Alcance y exclusiones** 

El alcance comprende clasificación de imagen completa en tres clases y una aplicación Android. No comprende segmentación semántica de lesiones, detección de múltiples hojas, historial clínico, cuentas, base de datos, diagnóstico agronómico definitivo ni versión iOS. Hugging Face se conserva como demostración web separada y no participa en la inferencia móvil. 

Papa Sana  •  Informe técnico  •  Página 4 

_**UNIVERSIDAD NACIONAL DE INGENIERÍA  |  COMPUTACIÓN GRÁFICA**_ 

## **3. Fundamentos técnicos** 

## **3.1 Imagen digital y preprocesamiento** 

Cada entrada se decodifica como una matriz RGB y se redimensiona a 224 × 224 píxeles. En el artefacto entrenado, la transformación preprocess_input de ResNet50 está integrada en el modelo; por ello, tanto la ruta Python como la móvil entregan valores float32 derivados de RGB en el rango 0–255. Mantener este contrato evita discrepancias entre entrenamiento y despliegue. 

## **3.2 Redes neuronales convolucionales y ResNet50** 

Una CNN aprende filtros espaciales que responden a bordes, texturas, formas y patrones de mayor abstracción. ResNet50 incorpora conexiones residuales de la forma y = F(x) + x. Estas rutas facilitan el flujo del gradiente en redes profundas y permiten reutilizar pesos aprendidos en ImageNet. El proyecto elimina la cabeza original, aplica Global Average Pooling y produce tres probabilidades mediante softmax. 

## **3.3 Transferencia de aprendizaje y fine-tuning** 

En la primera etapa, la base ResNet50 permanece congelada y solo se entrena la cabeza de clasificación. Posteriormente, se descongelan las últimas 30 capas y se recompila con Adam y una tasa de aprendizaje de 1×10⁻⁵. Esta estrategia ajusta características de alto nivel al dominio de enfermedades foliares sin modificar agresivamente los detectores generales aprendidos previamente. 

## **3.4 Segmentación HSV, morfología y Watershed** 

La ruta experimental convierte BGR a HSV, umbraliza tonos verdes, aplica apertura y cierre morfológico con un kernel 3×3, estima fondo por dilatación y primer plano por transformada de distancia. Los componentes  conectados  generan  marcadores  para  Watershed,  que  interpreta  la  imagen  como  una superficie topográfica y separa regiones a partir de cuencas. 

**Hallazgo experimental. Watershed no mejoró el clasificador porque este fue entrenado con imágenes completas. Al eliminar contexto y alterar texturas, el accuracy de test cayó de 97.83 % a 66.25 %. La técnica se conserva como demostración del curso, no como preprocesamiento productivo.** 

## **3.5 Cuantización y compresión del modelo** 

La cuantización dinámica de TensorFlow Lite reduce la representación numérica de parámetros internos y optimiza operaciones para inferencia. No debe confundirse con JPEG: JPEG comprime la imagen y puede introducir artefactos visuales; la cuantización TFLite comprime la representación de la red. En este proyecto, la interfaz de entrada y salida permanece float32. 

Papa Sana  •  Informe técnico  •  Página 5 

_**UNIVERSIDAD NACIONAL DE INGENIERÍA  |  COMPUTACIÓN GRÁFICA**_ 

## **4. Arquitectura del sistema** 

_Figura 1. Flujo técnico integral desde el dataset hasta la aplicación Android._ 

## **4.1 Componentes principales** 

|**Componente**|**Responsabilidad técnica**|
|---|---|
|papa_disease/data.py|Partición estratificada, hashes SHA-256 y construcción de tf.data.Dataset.|
|papa_disease/training.py|ResNet50, aumento de datos, entrenamiento base y fine-tuning.|
|papa_disease/segmentation.py|Segmentación experimental HSV, morfología y Watershed.|
|papa_disease/inference.py|Carga segura, validación de entrada y predicción H5/TFLite.|
|papa_disease/evaluation.py|Métricas, cobertura, matrices y comparación de variantes.|
|scripts/|Entradas reproducibles para preparar, entrenar, evaluar, convertir y verificar.|
|app.py|Demostración web Gradio local o en Hugging Face Spaces.|
|mobile/|Aplicación Flutter Android e integración del modelo TFLite offline.|
|tests/|Pruebas deterministas de contratos, métricas y casos de error.|



## **4.2 Contrato de inferencia** 

|**Elemento**|**Contrato**|
|---|---|
|Formatos|JPG, JPEG o PNG; máximo 10 MB y 25 megapíxeles.|
|Entrada|Tensor [1, 224, 224, 3], RGB, float32.|
|Salida|Tensor [1, 3], float32, probabilidades softmax.|
|Orden|Early_blight, Late_blight, healthy; orden inmutable.|
|Incertidumbre|Confianza máxima menor a 0.70: resultado no concluyente.|



Papa Sana  •  Informe técnico  •  Página 6 

_**UNIVERSIDAD NACIONAL DE INGENIERÍA  |  COMPUTACIÓN GRÁFICA**_ 

Privacidad móvil No se guardan ni envían fotografías; procesamiento local. 

## **5. Desarrollo del proyecto por fases** 

## **5.1 Fase 1 — Consolidación** 

El notebook exploratorio se reorganizó en módulos con responsabilidades separadas. Esta decisión permitió  ejecutar  entrenamiento,  inferencia,  segmentación  y  evaluación  mediante  scripts,  y  facilitó pruebas unitarias sin depender de una única sesión de Colab. 

## **5.2 Fase 2 — Evaluación confiable** 

Se creó un manifiesto estratificado 70/15/15 con semilla 123. Cada archivo recibe un SHA-256; si dos imágenes tienen contenido idéntico, la preparación se detiene para evitar fuga entre particiones. El conjunto de test se mantiene cerrado hasta congelar el umbral sobre validación. 

|**Partición**|**Fracción**|**Uso**|
|---|---|---|
|Entrenamiento|70 %|Ajuste de pesos y aumento de datos.|
|Validación|15 %|Selección del modelo, early stopping y umbral.|
|Prueba|15 %|Evaluación final una vez cerrado el desarrollo.|



## **5.3 Fase 3 — Demo web** 

La interfaz Gradio permite cargar una imagen y visualizar clase, confianza y probabilidades. Puede ejecutarse localmente o en Hugging Face Spaces. Su función es compartir una demostración accesible por navegador; no es dependencia de la aplicación Android. 

## **5.4 Fase 4 — Imágenes de campo** 

Se evaluaron 338 imágenes externas provenientes de PlantDoc e iNaturalist. El objetivo fue medir robustez fuera del fondo y encuadre controlados del dataset de laboratorio. La caída observada confirmó un cambio de dominio severo, especialmente para hojas sanas fotografiadas en escenas amplias. 

Papa Sana  •  Informe técnico  •  Página 7 

_**UNIVERSIDAD NACIONAL DE INGENIERÍA  |  COMPUTACIÓN GRÁFICA**_ 

## **5.5 Fase 5 — Fine-tuning** 

El modelo base se reconstruyó, cargó sus pesos por nombre y descongeló las últimas 30 capas de ResNet50. El entrenamiento adicional utilizó hasta 10 épocas, tasa de aprendizaje 1×10⁻⁵ y early stopping con paciencia de tres épocas, restaurando los mejores pesos según val_loss. 

## **5.6 Fase 6 — Conversión H5 a TFLite** 

El script export_tflite.py carga el modelo fine-tuned, crea un TFLiteConverter desde Keras y activa tf.lite.Optimize.DEFAULT. El resultado se escribe primero como archivo temporal y luego se reemplaza de forma atómica. El H5 de 200.71 MB se conserva fuera de Git; el TFLite de 22.83 MB es el artefacto versionado para Android. 

La verificación ejecuta H5 y TFLite sobre las mismas imágenes, realiza calentamiento, mide cinco corridas por defecto con perf_counter y compara la mediana de latencia. La prueba falla si cambia la clase o si la diferencia máxima de probabilidad supera 0.01. 

## **5.7 Fase 7 — Flutter Android** 

La aplicación Papa Sana incorpora una copia del modelo en mobile/assets/models/. image_picker obtiene una fotografía desde cámara o galería; el paquete image decodifica, aplica la orientación EXIF y redimensiona; tflite_flutter ejecuta el intérprete. El preprocesamiento y la inferencia se trasladan a isolates para evitar bloquear la interfaz. 

|para evitar bloquear la interfaz.||
|---|---|
|**Estado de fase**|**Evidencia**|
|Implementación terminada|Interfaz, validación, modelo local, cámara/galería y APK debug.|
|Verificación de software|flutter analyze sin problemas; 4 pruebas Flutter aprobadas; compilación Python<br>aprobada.|
|Registro pendiente|La documentación del repositorio aún solicita registrar prueba final en celular físico,<br>versión Android y latencia.|



Papa Sana  •  Informe técnico  •  Página 8 

_**UNIVERSIDAD NACIONAL DE INGENIERÍA  |  COMPUTACIÓN GRÁFICA**_ 

## **6. Evaluación y resultados** 

_Figura 2. Comparación entre laboratorio, fine-tuning y evaluación de campo._ 

## **6.1 Modelo base y efecto de Watershed** 

|**Entrada**|**Split**|**Accuracy**|**Macro F1**|**Cobertura ≥ 0.70**|
|---|---|---|---|---|
|Original|Validación|96.28 %|95.45 %|86.38 %|
|Original|Test|97.83 %|96.65 %|90.40 %|
|Watershed|Validación|68.73 %|48.92 %|74.61 %|
|Watershed|Test|66.25 %|46.67 %|71.52 %|



## **6.2 Fine-tuning** 

|**6.2 Fine-tuning**||||
|---|---|---|---|
|**Métrica**|**Base**|**Fine-tuned**|**Variación**|
|Accuracy|97.83 %|99.38 %|+1.55 puntos|
|Macro F1|96.65 %|98.39 %|+1.74 puntos|



La matriz de confusión final fue [[150, 0, 0], [0, 148, 2], [0, 0, 23]]. El resultado perfecto en tizón temprano debe interpretarse dentro del test de laboratorio, no como garantía de desempeño universal. 

Papa Sana  •  Informe técnico  •  Página 9 

_**UNIVERSIDAD NACIONAL DE INGENIERÍA  |  COMPUTACIÓN GRÁFICA**_ 

## **6.3 Evaluación de campo y cambio de dominio** 

|**Métrica**|**Campo (338 imágenes)**|**Test de laboratorio**|
|---|---|---|
|Accuracy|37.87 %|97.83 %|
|Macro F1|31.18 %|96.65 %|
|Cobertura ≥ 0.70|67.16 %|90.40 %|



La clase hoja sana obtuvo recall de 0.8 % en campo: solo una de 120 imágenes fue identificada correctamente. Muchas fotografías de iNaturalist mostraban plantas completas, flores, tubérculos o escenas agrícolas, mientras que el entrenamiento contenía primeros planos de hojas. La diferencia de composición y fondo explica parte del deterioro. 

**Interpretación responsable. La alta métrica de laboratorio mide desempeño sobre una distribución similar a la de entrenamiento. La evaluación de campo revela que el modelo todavía no es apto para decisiones agronómicas reales sin recolección, curación y reentrenamiento con datos representativos.** 

## **6.4 Optimización TFLite** 

|**Indicador**|**H5 fine-tuned**|**TFLite**|**Resultado**|
|---|---|---|---|
|Tamaño|200.71 MB|22.83 MB|−88.6 %|
|Latencia CPU|186.6 ms|108.0 ms|−42.1 %|
|Consistencia|Referencia|Misma clase|Δ prob. < 0.0035|



La latencia fue medida en CPU de escritorio x86 y solo sirve como comparación controlada entre formatos. No debe presentarse como latencia universal del teléfono; el rendimiento móvil depende del procesador, memoria, hilos y posibles delegados. 

## **7. Aplicación móvil** 

## **7.1 Flujo de ejecución** 

- El usuario acepta la advertencia educativa y selecciona cámara o galería. 

- La aplicación valida extensión, firma del archivo, tamaño máximo y número de píxeles. 

- La imagen se decodifica, corrige según EXIF y redimensiona a 224 × 224. 

- Los píxeles RGB se copian a un Float32List con forma [1,224,224,3]. 

- IsolateInterpreter ejecuta TFLite sin bloquear la interfaz principal. 

- La salida [1,3] se asocia al orden fijo de clases y se muestra con confianza. 

- Si la confianza máxima es menor a 0.70, el resultado se etiqueta no concluyente. 

Papa Sana  •  Informe técnico  •  Página 10 

_**UNIVERSIDAD NACIONAL DE INGENIERÍA  |  COMPUTACIÓN GRÁFICA**_ 

## **7.2 Decisiones de diseño** 

Se configuraron dos hilos para el intérprete y se usó compute para el preprocesamiento. La aplicación no crea historial ni transmite imágenes. Esta arquitectura reduce dependencia de red, evita costos de API y mejora privacidad. Hugging Face permanece útil para una demo web compartible, pero no interviene en el flujo móvil. 

## **8. Calidad, seguridad y reproducibilidad** 

## **8.1 Estrategia de pruebas** 

Las pruebas unitarias cubren lógica de partición, orden de clases, métricas, validación de contratos, finetuning  y  comparación  TFLite.  El  proyecto  utiliza  unittest  en  Python  y  flutter_test  en  móvil.  La compilación estática con compileall y flutter analyze detecta errores sintácticos y problemas de calidad antes del merge. 

## **8.2 Integridad del modelo** 

- El orden de salida se centraliza en config.py y no debe modificarse después del entrenamiento. 

- El conjunto test no participa en el ajuste de pesos ni en la selección del umbral. 

- Los reportes deben indicar modelo, split, umbral y commit utilizados. 

- Los H5 externos se consideran no confiables; solo se carga el artefacto controlado del equipo. 

- Dataset, credenciales, fotografías privadas, entornos y reportes generados permanecen fuera de Git. 

## **8.3 Limitaciones** 

- Desbalance de clases: solo 152 imágenes sanas frente a 1 000 por enfermedad. 

- Cambio de dominio entre fondos controlados y escenas reales de campo. 

- Clasificación global: no localiza lesiones ni cuantifica severidad. 

- Umbral 0.70 útil para abstención, pero no calibrado como probabilidad clínica. 

- La validación móvil documentada todavía debe completar el registro de dispositivo físico y latencia. 

## **8.4 Trabajo futuro** 

- Construir un dataset de campo curado y balanceado con primeros planos, iluminación y dispositivos diversos. 

- Aplicar validación cruzada o múltiples semillas para estimar variabilidad. 

- Evaluar calibración de probabilidades y seleccionar umbrales por clase. 

- Comparar cuantización entera post-entrenamiento y cuantización consciente del entrenamiento. 

- Incorporar mapas de activación de clase para explicar regiones relevantes. 

- Explorar segmentación semántica de lesiones solo si existe anotación por píxel. 

Papa Sana  •  Informe técnico  •  Página 11 

_**UNIVERSIDAD NACIONAL DE INGENIERÍA  |  COMPUTACIÓN GRÁFICA**_ 

## **9. Organización y contribución del equipo** 

El trabajo se organizó en fases y ramas enfocadas para separar datos, evaluación, optimización y aplicación. Esta estructura permite que los tres integrantes expliquen aspectos técnicos complementarios y mantengan trazabilidad de decisiones. 

|**Integrante**|**Eje técnico sugerido para exposición**|
|---|---|
|Edy Serrano|Problema, dataset, preparación reproducible, arquitectura ResNet50 y entrenamiento base.|
|Andrés La Torre|Procesamiento de imágenes, Watershed, evaluación, cambio de dominio y fine-tuning.|
|Diego Orrego|Conversión H5→TFLite, verificación de equivalencia, Flutter offline, demo y conclusiones.|



## **10. Conclusiones** 

- Se implementó una cadena completa y reproducible de clasificación visual, desde datos hasta Android. 

- ResNet50 con transferencia de aprendizaje y fine-tuning alcanzó 99.38 % de accuracy y 98.39 % de macro F1 en el test de laboratorio. 

- La experimentación con Watershed fue valiosa académicamente, aunque su uso previo al clasificador degradó el rendimiento. 

- TFLite redujo el modelo 88.6 % y preservó las decisiones observadas, habilitando inferencia móvil offline. 

- La caída a 37.87 % en campo es el resultado crítico del proyecto: evidencia que precisión de laboratorio no equivale a robustez real. 

- La aplicación protege la privacidad al procesar imágenes localmente, pero debe mantenerse como herramienta educativa hasta mejorar la generalización. 

## **Referencias** 

1. He, K., Zhang, X., Ren, S. y Sun, J. (2016). Deep Residual Learning for Image Recognition. IEEE CVPR. 

2. TensorFlow. TensorFlow Lite: conversión y optimización de modelos para dispositivos móviles. 

3. Keras. ResNet and ResNetV2: arquitecturas preentrenadas y preprocess_input. 

4. OpenCV. Image segmentation with Distance Transform and Watershed Algorithm. 

5. Flutter y TensorFlow Lite Flutter. Documentación de integración para Android. 

6. Potato Disease Dataset, Kaggle, licencia CC0 1.0. 

7. PlantDoc Dataset, imágenes de enfermedades vegetales, licencia CC BY 4.0. 

8. Documentación interna del repositorio: PLAN_PROYECTO.md y RESULTADOS_FASE_2/4/5/6/7.md, consultados en julio de 2026. 

## **Anexo A. Comandos de reproducción** 

python -m scripts.prepare_split python -m scripts.train 

python -m scripts.evaluate --split validation --threshold 0.70 python -m scripts.evaluate --split test --threshold 0.70 python -m scripts.finetune 

Papa Sana  •  Informe técnico  •  Página 12 

_**UNIVERSIDAD NACIONAL DE INGENIERÍA  |  COMPUTACIÓN GRÁFICA**_ 

python -m scripts.export_tflite 

python -m scripts.verify_tflite --image ruta/hoja.jpg 

python -m unittest discover -s tests -v python -m compileall papa_disease scripts tests cd mobile && flutter analyze && flutter test flutter run -d <identificador-dispositivo> 

## **Anexo B. Matrices de confusión** 

|**Modelo / dominio**|**Matriz (filas reales, columnas predichas)**|
|---|---|
|Base, test|[[147, 3, 0], [1, 147, 2], [0, 1, 22]]|
|Fine-tuned, test|[[150, 0, 0], [0, 148, 2], [0, 0, 23]]|
|Campo|[[48, 66, 0], [20, 79, 5], [31, 88, 1]]|



Papa Sana  •  Informe técnico  •  Página 13 

