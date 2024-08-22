import os

data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

epocas = 20
longitud, altura = 150, 150
batch_size = 32
pasos = len(os.listdir(data_entrenamiento))
validation_steps = len(os.listdir(data_validacion))
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)

# Contar el número de carpetas en el directorio de entrenamiento para definir el número de clases
clases = len([folder for folder in os.listdir(data_entrenamiento) if os.path.isdir(os.path.join(data_entrenamiento, folder))])

lr = 0.0004
