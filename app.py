import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path='modelNella.tflite')
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Definir nombres de clases
class_names = ['Burj Khalifa', 'Chichén Itzá', 'Cristo Redentor', 'Torre Eiffel',
               'Gran muralla china', 'Machu Picchu', 'Gran Pirámide de Guiza', 'Coliseo Romano',
               'Estatua de la Libertad', 'Stonehenge', 'Taj Mahal', 'Salto Ángel']

# Definir umbral de confianza (por ejemplo, 80%)
confidence_threshold = 0.7

# Preprocesar la imagen subida
def preprocess_image(image):
    size = (224, 224)  # Tamaño esperado por el modelo
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
    #predicted_class = class_names[np.argmax(predictions)]
    #st.write(f"Prediction: {predicted_class}")

    max_probabilidad = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    # Verificar si la probabilidad supera el umbral de confianza
    if max_probabilidad < confidence_threshold:
        st.write(f"No se pudo clasificar la imagen. La probabilidad maxima fue de {max_probabilidad:.2f}")
    else:
        st.write(f"Prediction: {predicted_class} con una probabilidad de {max_probabilidad:.2f}")
