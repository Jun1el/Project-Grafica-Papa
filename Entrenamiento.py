# Definición y Entrenamiento de la CNN usando ResNet50 (Añadir en Colab)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models, callbacks
import tensorflow as tf

# 1. Cargar el modelo base preentrenado ResNet50 (El que pidió el profesor)
base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Congelar los pesos base para hacer Transfer Learning
base_model.trainable = False 

# 2. Construir la arquitectura final
model = models.Sequential([
    data_augmentation, # La capa de Data Augmentation creada en celdas anteriores
    layers.Lambda(preprocess_input), # CRÍTICO: Preprocesamiento exacto que requiere ResNet
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3), # Regularización aumentada al 30% por la profundidad de ResNet
    layers.Dense(len(class_names), activation='softmax') # Capa de salida multiclasa
])

# 3. Compilación del modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Configurar Early Stopping (Script Avanzado)
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
    epochs=15,
    callbacks=[early_stopping]
)

# 6. Guardar el modelo exportado
model.save('modelo_papas_resnet50.h5')
print("Modelo ResNet guardado exitosamente.")