sactivation = 'elu'
sufix="_1stHO.png"

import os
import random
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Model , load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.optimizers import Adam

import cv2
import matplotlib.pyplot as plt
import sys

epochs = 20
PART_SIZE=256
lr =0.001

def dice_loss(y_true, y_pred):
    smooth = 1e-5
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice_score
    return dice_loss

def iou(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou_score = intersection / union
    return iou_score

def read_image(file_path, rotation=0, is_mask=False):
    image = Image.open(file_path)
    image = image.resize((256, 256))
    image = image.rotate(rotation)
    image_array = np.array(image)
    if is_mask:  # If the image is a mask
        image_array = (image_array > 0).astype(np.float32)
    else:  # If the image is not a mask
        image_array = image_array/255
    return image_array

def load_data(file_list, test_ratio=0.3, patch_size=(PART_SIZE, PART_SIZE)):
    random.shuffle(file_list)
    num_test = int(len(file_list) * test_ratio)

    X_test = []
    Y_test = []
    X_train = []
    Y_train = []

    for i, file_name in enumerate(file_list):
        file_path = os.path.join('Images/', file_name)
        for angle in range(1):
            image_array_X = read_image(file_path, angle * 90)
            index = file_path.index('.')
            image_array_Y = read_image(file_path[:index] + sufix, angle * 90,1)
            
            # Podział obrazów na fragmenty
            for x in range(0, image_array_X.shape[0]-patch_size[0]+1, patch_size[0]):
                for y in range(0, image_array_X.shape[1]-patch_size[1]+1, patch_size[1]):
                    patch_X = image_array_X[x:x+patch_size[0], y:y+patch_size[1], :]
                    patch_Y = image_array_Y[x:x+patch_size[0], y:y+patch_size[1]]
                    if i < num_test:
                        X_test.append(patch_X)
                        Y_test.append(patch_Y)
                    else:
                        X_train.append(patch_X)
                        Y_train.append(patch_Y)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_test, Y_test, X_train, Y_train


from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

class ClippedMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        y_pred = tf.clip_by_value(y_pred, 0, 1)  # Ensure values are between 0 and 1
        return super().update_state(y_true, y_pred, sample_weight)

def create_model(input_shape):
    inputs = Input(input_shape)

    # Encoding part
    conv1 = Conv2D(64, (3, 3), activation=sactivation, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3, 3), activation=sactivation, padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.5)(pool1)

    conv2 = Conv2D(128, (3, 3), activation=sactivation, padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, (3, 3), activation=sactivation, padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(256, (3, 3), activation=sactivation, padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), activation=sactivation, padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    center = Conv2D(512, (3, 3), activation=sactivation, padding='same')(pool3)
    center = BatchNormalization()(center)
    center = Conv2D(512, (3, 3), activation=sactivation, padding='same')(center)
    center = BatchNormalization()(center)

    # Decoding part
    up3 = UpSampling2D(size=(2, 2))(center)
    merge3 = concatenate([conv3, up3])
    conv4 = Conv2D(256, (3, 3), activation=sactivation, padding='same')(merge3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation=sactivation, padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    merge2 = concatenate([conv2, up2])
    conv5 = Conv2D(128, (3, 3), activation=sactivation, padding='same')(merge2)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation=sactivation, padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up1 = UpSampling2D(size=(2, 2))(conv5)
    merge1 = concatenate([conv1, up1])
    conv6 = Conv2D(64, (3, 3), activation=sactivation, padding='same')(merge1)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation=sactivation, padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation=sactivation)(conv6)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision()])

    return model




def predict_from_model(path, model, lev):
    image = read_image(path)
    
    # Rozdziel obraz na fragmenty
    patches = []
    for x in range(0, image.shape[0] - PART_SIZE + 1, PART_SIZE):
        for y in range(0, image.shape[1] - PART_SIZE + 1, PART_SIZE):
            patch = image[x:x+PART_SIZE, y:y+PART_SIZE, :]
            patches.append(patch)
    
    # Przewiduj dla każdego fragmentu
    patches = np.array(patches)
    predictions = model.predict(patches)
    
    # Przygotuj do połączenia fragmentów
    predicted_image = np.zeros((image.shape[0], image.shape[1]))
    
    i = 0
    for x in range(0, image.shape[0] - PART_SIZE + 1, PART_SIZE):
        for y in range(0, image.shape[1] - PART_SIZE + 1, PART_SIZE):
            predicted_patch = predictions[i]
            predicted_patch = np.squeeze(predicted_patch, axis=-1)
            binary_mask = (predicted_patch > lev).astype(np.uint8) * 255
            predicted_image[x:x+PART_SIZE, y:y+PART_SIZE] = binary_mask
            i += 1
            
    # Wyświetl obraz
    plt.imshow(predicted_image, cmap='gray')
    plt.show()

# Przykładowa lista nazw plików
if sys.argv[1] == '1':
    file_list = ['Image_01L.jpg','Image_02L.jpg','Image_03L.jpg','Image_04L.jpg','Image_05L.jpg','Image_06L.jpg',
             'Image_07L.jpg','Image_08L.jpg','Image_09L.jpg','Image_10L.jpg','Image_11L.jpg','Image_12L.jpg',
             'Image_13L.jpg','Image_01R.jpg','Image_02R.jpg','Image_03R.jpg','Image_04R.jpg','Image_05R.jpg',
             'Image_06R.jpg','Image_07R.jpg','Image_08R.jpg','Image_09R.jpg','Image_10R.jpg','Image_11R.jpg',
             'Image_12R.jpg','Image_13R.jpg','Image_14R.jpg','Image_14L.jpg']

    # Wczytanie danych
    X_test, Y_test, X_train, Y_train = load_data(file_list, patch_size=(PART_SIZE, PART_SIZE))
    model = create_model((PART_SIZE, PART_SIZE, 3))
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=200, epochs=epochs)
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

for i in range(1, 10):
    predict_from_model("test.jpg", model, i/10)