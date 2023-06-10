sufix="_1stHO.png"
sactivation = 'elu'

import os
import random
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Model , load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate,BatchNormalization
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import sys

epochs = 10
val_ratio = 0.1

def read_image(file_path, isn = False):
    image = Image.open(file_path)
    # Przykładowe przetwarzanie obrazu, jeśli jest to wymagane
    # Przykład: przeskalowanie obrazu do rozmiaru 224x224
    image = image.resize((256, 256))
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Przetwarzanie obrazu do postaci tablicy numpy
    if isn:
        image = Image.eval(image, lambda x: 255 - x)
    #plt.imshow(image)
    #plt.show()
    image_array = np.array(image)
    image_array = image_array/255
    return image_array

from sklearn.model_selection import train_test_split

# ...

def load_data(file_list, test_ratio=0.2, val_ratio=0.1):
    random.shuffle(file_list)
    num_test = int(len(file_list) * test_ratio)
    num_val = int(len(file_list) * val_ratio)

    X_test = []
    Y_test = []
    X_train = []
    Y_train = []

    for i, file_name in enumerate(file_list):
        file_path = os.path.join('Images/', file_name)
        for angle in range(2):
            image_array_X = read_image(file_path, angle * 90)
            index = file_path.index('.')
            image_array_Y = read_image(file_path[:index] + sufix, angle * 90)
            if i < num_test:
                X_test.append(image_array_X)
                Y_test.append(image_array_Y)
            else:
                X_train.append(image_array_X)
                Y_train.append(image_array_Y)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_ratio, random_state=42)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)

    return X_test, Y_test, X_train, Y_train, X_val, Y_val


def create_model(input_shape):
    inputs = Input(input_shape)

    # Część kodująca
    conv1 = Conv2D(128, (3, 3), activation=sactivation, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(128, (3, 3), activation=sactivation, padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(256, (3, 3), activation=sactivation, padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(256, (3, 3), activation=sactivation, padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Część dekodująca
    up2 = UpSampling2D(size=(2, 2))(pool2)
    merge2 = concatenate([conv2, up2])
    conv3 = Conv2D(256, (3, 3), activation=sactivation, padding='same')(merge2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), activation=sactivation, padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    merge1 = concatenate([conv1, up1])
    conv4 = Conv2D(128, (3, 3), activation=sactivation, padding='same')(merge1)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, (3, 3), activation=sactivation, padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)

    # Warstwa wyjściowa
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv4)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def predict_from_model(path,model,lev):
    image = read_image(path)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    prediction = np.squeeze(prediction, axis=0)
    binary_mask = (prediction > lev).astype(np.uint8)
    binary_mask = binary_mask*255
    plt.imshow(binary_mask, cmap='gray')
    plt.show()

# Przykładowa lista nazw plików
if sys.argv[1] == '1':
    file_list = ['Image_01L.jpg','Image_02L.jpg','Image_03L.jpg','Image_04L.jpg','Image_05L.jpg','Image_06L.jpg',
             'Image_07L.jpg','Image_08L.jpg','Image_09L.jpg','Image_10L.jpg','Image_11L.jpg','Image_12L.jpg',
             'Image_13L.jpg','Image_01R.jpg','Image_02R.jpg','Image_03R.jpg','Image_04R.jpg','Image_05R.jpg',
             'Image_06R.jpg','Image_07R.jpg','Image_08R.jpg','Image_09R.jpg','Image_10R.jpg','Image_11R.jpg',
             'Image_12R.jpg','Image_13R.jpg','Image_14R.jpg','Image_14L.jpg']

    # Wczytanie danych
    X_test, Y_test, X_train, Y_train, X_val, Y_val = load_data(file_list, val_ratio=val_ratio)
    model = create_model((256, 256, 3))
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=1000, epochs=epochs)
    model.save('model.h5')

    # Wykresy funkcji straty
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
else:
    model = tf.keras.models.load_model("model.h5")


#predict_from_model("Images\Image_01L.jpg",model,0.4)
for i in range(1,10):
    predict_from_model("test.jpg",model,i/10)
#print(X_train.shape)

#image = Image.fromarray(X_test[1])

#image.save('obraz.png')

