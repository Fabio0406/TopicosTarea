import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from config import data_entrenamiento, data_validacion, longitud, altura, batch_size

class DataLoader:
    def __init__(self):
        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        self.test_datagen = ImageDataGenerator(rescale=1. / 255)

    def load_data(self):
        train_generator = self.train_datagen.flow_from_directory(
            data_entrenamiento,
            target_size=(altura, longitud),
            batch_size=batch_size,
            class_mode='categorical')

        validation_generator = self.test_datagen.flow_from_directory(
            data_validacion,
            target_size=(altura, longitud),
            batch_size=batch_size,
            class_mode='categorical')

        # Contar el número total de imágenes en el directorio de entrenamiento
        total_train_images = sum([len(files) for _, _, files in os.listdir(data_entrenamiento)])
        total_val_images = sum([len(files) for _, _, files in os.listdir(data_validacion)])

        steps_per_epoch = total_train_images // batch_size
        validation_steps = total_val_images // batch_size

        return train_generator, validation_generator, steps_per_epoch, validation_steps
