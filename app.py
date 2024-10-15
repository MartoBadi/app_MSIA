import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path='modelLautaro.tflite')
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Definir nombres de clases
class_names = ['burj_khalifa', 'chichen_itza', 'christ the reedemer', 'eiffel_tower',
               'great_wall_of_china', 'machu_pichu', 'pyramids_of_giza', 'roman_colosseum',
               'statue_of_liberty', 'stonehenge', 'taj_mahal', 'venezuela_angel_falls']

# Preprocesar la imagen subida
def preprocess_image(image):
    size = (150, 150)  # Tamaño esperado por el modelo
    image = image.resize(size)  # Usar PIL para redimensionar
    img_array = np.array(image)  # Convertir la imagen a un array NumPy
    img_array = img_array / 255.0  # Normalizar la imagen
    img_array = np.expand_dims(img_array, axis=0)  # Expandir a un batch de tamaño 1
    img_array = img_array.astype(np.float32)  # Asegurar que sea float32
    return img_array

# Título de la aplicación
st.title("Clasificación de imágenes de maravillas del mundo")
st.write("Sube una imagen de una de las maravillas del mundo y el modelo la clasificará.")

# Subir archivo
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Cargar la imagen usando PIL
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada', use_column_width=True)
    st.write("Clasificando...")
    
    # Preprocesar la imagen
    img_array = preprocess_image(image)
    
    # Hacer predicciones
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # Obtener la clase con mayor probabilidad
    predicted_class = class_names[np.argmax(predictions)]
    st.write(f"Prediction: {predicted_class}")
