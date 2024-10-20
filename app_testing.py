import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image

# La app funciona perfecto.

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path='modelNella.tflite')
interpreter.allocate_tensors()

repo_path = os.path.dirname(os.path.abspath(__file__))

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Definir nombres de clases
class_names = ['burj_khalifa', 'chichen_itza', 'christ the reedemer', 'eiffel_tower', 
               'great_wall_of_china', 'machu_pichu', 'pyramids_of_giza', 'roman_colosseum', 
               'statue_of_liberty', 'stonehenge', 'taj_mahal', 'venezuela_angel_falls']

# Definir umbral de confianza (por ejemplo, 80%)
confidence_threshold = 0.6050201058387756

# Preprocesar la imagen subida
def preprocess_image(image):
    size = (224, 224)  # Tamaño esperado por el modelo
    image = image.resize(size)  # Usar PIL para redimensionar
    img_array = np.array(image)  # Convertir la imagen a un array NumPy
    img_array = img_array / 255.0  # Normalizar la imagen
    img_array = np.expand_dims(img_array, axis=0)  # Expandir a un batch de tamaño 1
    img_array = img_array.astype(np.float32)  # Asegurar que sea float32
    return img_array

# Función para buscar la carpeta de la imagen
def find_image_folder(image_name, base_dir=repo_path):
    for root, dirs, files in os.walk(base_dir):
        if image_name in files:
            return os.path.basename(root)
    return "Unknown"

# Inicializar textos antes de cargar y procesar imágenes
def reset_texts():
    st.session_state['result_text'] = ""
    st.session_state['prediction_text'] = ""
    st.session_state['real_class_text'] = ""
    st.session_state['correct_predictions_text'] = ""

predicciones_correctas = 0
numero_de_imagen = 0
# Función para procesar una imagen y actualizar textos en pantalla
def process_image_and_update_display(image, image_name):
    global predicciones_correctas
    global numero_de_imagen
  
    reset_texts()  # Reiniciar textos antes de procesar la imagen
    st.image(image, caption=image_name)

    # Preprocesar la imagen
    img_array = preprocess_image(image)

    # Hacer predicciones
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    max_probabilidad = np.max(predictions)

    # Verificar si la probabilidad supera el umbral de confianza
    if max_probabilidad < confidence_threshold:
        st.session_state['result_text'] = f"No se pudo clasificar la imagen. La probabilidad maxima fue de {max_probabilidad:.2f}"
        predicted_class = " "
    else:
        # Obtener la clase con mayor probabilidad
        predicted_class = class_names[np.argmax(predictions)]
        st.session_state['prediction_text'] = f"Prediction: {predicted_class} con una probabilidad de {max_probabilidad:.2f}"

    # Buscar la carpeta de la imagen
    real_class = find_image_folder(image_name, base_dir=repo_path)
    st.session_state['real_class_text'] = f"Real class: {real_class}"
    st.session_state['prediction_text'] = f"Prediction: {predicted_class}"

    # Inicializar el contador en session_state si no existe
    if 'correct_predictions' not in st.session_state:
        st.session_state.correct_predictions = 0

    # Incrementar el contador si la clase real es igual a la clase predicha
    if real_class == predicted_class:
        global predicciones_correctas
        predicciones_correctas += 1
      
    numero_de_imagen += 1

    # Mostrar los textos y el contador actualizado
    st.write(st.session_state['result_text'])
    st.write(st.session_state['prediction_text'])
    st.write(st.session_state['real_class_text'])
    st.write(f"Correct Predictions: {st.session_state.correct_predictions}")

# Título de la aplicación
st.title("Clasificación de imágenes de maravillas del mundo")
st.write("Este sitio web fue creado para la materia Modelizado de Sistemas de IA de la Tecnicatura Superior en Ciencias de Datos e Inteligencia Artificial del IFTS 18. La idea es que subas una imagen de una de las maravillas del mundo y el modelo la clasificará.")

# Subir archivo
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Procesar y mostrar imágenes de un directorio
image_directory = "./test"
for root, dirs, files in os.walk(image_directory):
    for image_name in files: 
        image_path = os.path.join(root, image_name)
        image = Image.open(image_path)
        process_image_and_update_display(image, image_name)

st.write(f"La cantidad de imagenes analizadas es: {numero_de_imagen} y se hicieron {predicciones_correctas} predicciones correctas.")
