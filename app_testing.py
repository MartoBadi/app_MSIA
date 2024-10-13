import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

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
    size = (150, 150)
    image = np.array(image)
    image = np.resize(image, (size[0], size[1], 3))  # Redimensionar la imagen
    img_array = image / 255.0  # Normalizar la imagen
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32)  # Asegurarse de que la imagen sea de tipo FLOAT32
    return img_array

# Función para evaluar el conjunto de prueba
def evaluate_test_set(test_dir):
    correct_predictions = 0
    total_predictions = 0

    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image = mpimg.imread(image_path)
                img_array = preprocess_image(image)
                
                # Hacer predicciones
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])
                
                # Obtener la clase con mayor probabilidad
                predicted_class = class_names[np.argmax(predictions)]
                
                # Comparar con la etiqueta real
                if predicted_class == class_name:
                    correct_predictions += 1
                total_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

# Directorio del conjunto de prueba
test_dir = '/test'

# Evaluar el conjunto de prueba
accuracy = evaluate_test_set(test_dir)
st.write(f"Test set accuracy: {accuracy:.2f}")
