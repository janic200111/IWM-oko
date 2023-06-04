sufix="_1stHO.png"

import os
import random
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Model , load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
import cv2
import matplotlib.pyplot as plt

def read_image(file_path):
    image = Image.open(file_path)
    # Przykładowe przetwarzanie obrazu, jeśli jest to wymagane
    # Przykład: przeskalowanie obrazu do rozmiaru 224x224
    image = image.resize((256, 256))
    # Przetwarzanie obrazu do postaci tablicy numpy
    image_array = np.array(image)
    return image_array

def load_data(file_list, test_ratio=0.2):
    random.shuffle(file_list)
    num_test = int(len(file_list) * test_ratio)

    X_test = []
    Y_test = []
    X_train = []
    Y_train = []

    for i, file_name in enumerate(file_list):
        file_path = os.path.join('Images/', file_name)
        image_array_X = read_image(file_path)
        index = file_path.index('.')
        image_array_Y =read_image(file_path[:index] + sufix)
        if i < num_test:
            X_test.append(image_array_X)
            Y_test.append(image_array_Y)
        else:
            X_train.append(image_array_X)
            Y_train.append(image_array_Y)

    # Konwersja list na tablice numpy
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_test, Y_test, X_train, Y_train

def create_model(input_shape):
    inputs = Input(input_shape)

    # Część kodująca
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Część dekodująca
    up2 = UpSampling2D(size=(2, 2))(pool2)
    merge2 = concatenate([conv2, up2])
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    merge1 = concatenate([conv1, up1])
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    # Warstwa wyjściowa
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv4)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
def predict_from_model(path,model):
    image = read_image(path)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    prediction = np.squeeze(prediction, axis=0)
    binary_mask = (prediction > 0.5).astype(np.uint8)
    plt.imshow(binary_mask, cmap='gray')
    plt.show()

# Przykładowa lista nazw plików
file_list = ['Image_01L.jpg','Image_02L.jpg','Image_03L.jpg','Image_04L.jpg','Image_05L.jpg','Image_06L.jpg',
             'Image_07L.jpg','Image_08L.jpg','Image_09L.jpg','Image_10L.jpg','Image_11L.jpg','Image_12L.jpg',
             'Image_13L.jpg','Image_01R.jpg','Image_02R.jpg','Image_03R.jpg','Image_04R.jpg','Image_05R.jpg',
             'Image_06R.jpg','Image_07R.jpg','Image_08R.jpg','Image_09R.jpg','Image_10R.jpg','Image_11R.jpg',
             'Image_12R.jpg','Image_13R.jpg','Image_14R.jpg','Image_14L.jpg']

# Wczytanie danych
X_test, Y_test, X_train, Y_train = load_data(file_list)

model = create_model((256, 256, 3))
model.fit(X_train, Y_train, batch_size=32, epochs=10)
model.save('path_to_my_model.h5')
loss, accuracy = model.evaluate(X_test, Y_test)
print(loss, accuracy)

predict_from_model("kubus.jpg",model)
predict_from_model("test.jpg",model)

#print(X_train.shape)

#image = Image.fromarray(X_test[1])

#image.save('obraz.png')
