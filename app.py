import gradio as gr
from papa_disease.inference import load_model, predict
from papa_disease.config import RETRAINED_MODEL_PATH, CLASS_NAMES_ES

print("Cargando el modelo... (Esto puede tomar unos segundos)")
# Cargamos el modelo globalmente UNA SOLA VEZ usando la ruta del modelo reentrenado
model = load_model(RETRAINED_MODEL_PATH)
print("Modelo cargado exitosamente.")

def predict_disease(image):
    """
    Recibe una imagen (numpy array RGB provisto por Gradio),
    realiza la predicción y devuelve las probabilidades y el aviso.
    """
    if image is None:
        return None, "Por favor sube una imagen."
    
    try:
        # Gradio ya nos entrega un array numpy en RGB, que es exactamente
        # lo que espera nuestra funcion predict()
        result = predict(model, image)
        
        # Formateamos las probabilidades usando los nombres en español
        probabilities = result["probabilities"]
        formatted_probs = {
            CLASS_NAMES_ES[class_name]: float(prob)
            for class_name, prob in probabilities.items()
        }
        
        return formatted_probs, result["disclaimer"]
    except Exception as e:
        # En caso de que la imagen sea muy pequeña o tenga algún problema
        return None, f"Error al procesar la imagen: {str(e)}"

# Creamos la interfaz
demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="numpy", label="Sube una imagen (JPG, JPEG o PNG)"),
    outputs=[
        gr.Label(num_top_classes=3, label="Predicción"),
        gr.Textbox(label="Aviso", interactive=False)
    ],
    title="Clasificador de Enfermedades en Hojas de Papa",
    description=(
        "Sube una imagen de una hoja de papa para clasificar si tiene "
        "**Tizón temprano**, **Tizón tardío** o si es una **Hoja sana**."
    ),
    # Desactivamos el flagging para no guardar las imágenes subidas por los usuarios
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()
