# model.py
import sys
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from config import longitud, altura, filtrosConv1, filtrosConv2, tamano_filtro1, tamano_filtro2, tamano_pool, clases, lr

class CNNModel:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Convolution2D(filtrosConv1, tamano_filtro1, padding="same", input_shape=(longitud, altura, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=tamano_pool))
        
        self.model.add(Convolution2D(filtrosConv2, tamano_filtro2, padding="same", activation='relu'))
        self.model.add(MaxPooling2D(pool_size=tamano_pool))
        
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(clases, activation='softmax'))

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(learning_rate=lr),
                           metrics=['accuracy'])

    def train(self, train_generator, validation_generator, steps_per_epoch, validation_steps, epochs):
        self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps)

    def save_model(self, model_path='./modelo/modelo.keras', weights_path='./modelo/pesos.weights.h5'):
        if not os.path.exists('./modelo/'):
            os.mkdir('./modelo/')
        self.model.save(model_path)
        self.model.save_weights(weights_path)
