from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models, callbacks
import tensorflow as tf

# 1. Cargar el modelo base preentrenado ResNet50
# include_top=False realizara que le quite la clasificación original (las 1000 clases de ImageNet: perro, gato, avión, auto, etc.)
# weights='imagenet' el modelo ya aprendio con millones de imágenes: bordes, formas, texturas, objetos y no entrenara desde cero
base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Congela todas las capas del modelo base para que sus pesos no cambien durante el entrenamiento
base_model.trainable = False

# 2. Construir la arquitectura
model = models.Sequential([
    data_augmentation,

    # Lambda aplica una función 'preprocess_input' que alinea
    # los colores de las fotos para que encajen matemáticamente con lo que ResNet espera
    layers.Lambda(preprocess_input),

    # El cerebro convolucional
    base_model,

    # Convertimos los mapas de características 2D a un vector 1D
    layers.GlobalAveragePooling2D(),

    # Apagamos neuronas al azar al 30% (Dropout) para obligar a que la red no memorice las hojas de papas
    layers.Dropout(0.3),

    # La salida real: capa densa del tamaño de 'class_names' (3: saludable, temprana, tardía)
    # Softmax da como salida porcentajes del 0 al 1
    layers.Dense(len(class_names), activation='softmax')
])

# 3. Compilación del modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Aplicamos el optimizador Adam
    loss='sparse_categorical_crossentropy', # Función de pérdida (error)
    metrics=['accuracy'] # Precisión
)

# 4. Configurar Early Stopping (Script Avanzado)
# Si en 3 épocas el 'val_loss' (error en nuevos datos) no baja, detiene todo y evita un sobreajuste
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 5. Ejecutar el entrenamiento
print("Iniciando entrenamiento con arquitectura ResNet50...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=15, # Limite maximo de rondas, pero tomamos en cuenta que Early Stopping detendrá antes si no hay mejora
    callbacks=[early_stopping]
)

# 6. Guardar el modelo
# .h5 es el formato en que TensorFlow guarda todo el bloque neuronal y pesos
model.save('modelo_papas_resnet50.h5')
print("Modelo ResNet guardado exitosamente.")
