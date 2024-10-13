import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Definir nombres de clases
class_names = ['burj_khalifa', 'chichen_itza', 'christ the reedemer', 'eiffel_tower', 'great_wall_of_china', 'machu_pichu', 'pyramids_of_giza', 'roman_colosseum', 'statue_of_liberty', 'stonehenge', 'taj_mahal', 'venezuela_angel_falls']

def preprocess_image(image):
    # Redimensionar la imagen al tamaño esperado por el modelo
    image = image.resize((150, 150))
    
    # Convertir la imagen a un array de numpy
    image = np.array(image)
    
    # Normalizar la imagen (asegúrate de que esto coincida con el notebook)
    image = image / 255.0
    
    # Expandir las dimensiones para que coincida con el formato esperado por el modelo
    image = np.expand_dims(image, axis=0)
    
    # Asegurarse de que la imagen sea de tipo FLOAT32
    image = image.astype(np.float32)
    
    return image

st.title("Clasificación de imágenes de maravillas del mundo")
st.write("Este sitio web fue creado para la materia Modelizado de Sistemas de IA de la Tecnicatura Superior en Ciencias de Datos e Inteligencia Artificial del IFTS 18. La idea es que subas una imagen de uno de las siguientes maravillas del mundo: burj_khalifa, chichen_itza, christ the reedemer, eiffel_tower, great_wall_of_china, machu_pichu, pyramids_of_giza, roman_colosseum, statue_of_liberty, stonehenge, taj_mahal, venezuela_angel_falls y el modelo te dirá qué maravilla aparece en la imagen. ¡Diviértete!")

# Subir archivo
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leer la imagen usando Matplotlib
    image = mpimg.imread(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Preprocesar la imagen
    img_array = preprocess_image(image)
    
    # Hacer predicciones
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # Obtener la clase con mayor probabilidad
    predicted_class = class_names[np.argmax(predictions)]
    st.write(f"Prediction: {predicted_class}")
  
