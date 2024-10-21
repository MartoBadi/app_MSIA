El archivo requirements.txt tiene las librerías que son necesarias para correr la aplicación. Para entrenar el modelo de esta aplicación se uso el dataset "Wonders of the World Image Dataset"
de kaggle: https://www.kaggle.com/datasets/balabaskar/wonders-of-the-world-image-classification. El archivo MDSI_clasificacion_imagenesNella.ipynb tiene el notebook con el que se preprocesan los datos que están en el directorio "imagenes dataset" del repositorio (antes del preprocesamiento se usan la función create_directories   para crear los directorios train, val y test en la ruta './', es decir, donde están el notebook, la aplicación, etc., y la función split_data que divide los datos del directorio "imagenes dataset" copiándo el 75% en la carpeta train, el 15% en la carpeta val, y el 15% restante en la carpeta test. El lector puede probar esto ejecutando el notebook en la nube o clonándolo en su máquina). 

El directorio "imagenes dataset" contiene doce subdirectorios, cada uno con las imágenes de cada categoría del dataset, así es como están los datos en kaggle). En el notebook también se entrena y guarda el modelo model.keras.Este modelo está integrado en la aplicación app.py hecha con la librería de python Streamlit. La aplicación se puede probar en: https://app-grupo2-msia.streamlit.app/.  

El archivo app_testing.py contiene una versión de la aplicación que se utilizó para testear app.py con un caso de prueba por cada imagen del set de prueba, con lo que se chequeó que la aplicación funciona correctamente. Se puede ver la aplicación de prueba en este link:  https://testing-app-grupo2.streamlit.app.
 
