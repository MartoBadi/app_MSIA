import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Definir nombres de clases
class_names = ['burj_khalifa', 'chichen_itza', 'christ the reedemer', 'eiffel_tower', 'great_wall_of_china', 'machu_pichu', 'pyramids_of_giza', 'roman_colosseum', 'statue_of_liberty', 'stonehenge', 'taj_mahal', 'venezuela_angel_falls']

# Preprocesar la imagen subida
def preprocess_image(image):
    image = np.resize(image,(150, 150))
    image = np.array(image)
    img_array = image / 255.0  # Normalizar la imagen
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32) 
    return img_array

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
  
