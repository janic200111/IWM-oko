import os
import random
import numpy as np
from PIL import Image 

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, UpSampling2D, Concatenate, Input, Dropout
from tensorflow.keras.models import Model,load_model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Binarizer

import matplotlib.pyplot as plt
import sys

epochs = 100
PART_SIZE=256
lr =0.001
batches=32
sactivation = 'sigmoid'
sufix="_1stHO.png"
data_test_ratio=0.3
num_of_rotations=4

def read_image(file_path, rotation=0, is_mask=False):
    image = Image.open(file_path)
    image = image.resize((PART_SIZE, PART_SIZE))
    image = image.rotate(rotation)
    image_array = np.array(image)
    if is_mask:  
        image_array = (image_array > 0).astype(np.float32)
    else:  
        image_array = image_array/255.0
    return image_array

def load_data(file_list, test_ratio=data_test_ratio, patch_size=(PART_SIZE, PART_SIZE)):
    random.shuffle(file_list)
    num_test = int(len(file_list) * test_ratio)

    X_test = []
    Y_test = []
    X_train = []
    Y_train = []

    for i, file_name in enumerate(file_list):
        file_path = os.path.join('Images/', file_name)
        for angle in range(num_of_rotations):
            image_array_X = read_image(file_path, angle * 90)
            index = file_path.index('.')
            image_array_Y = read_image(file_path[:index] + sufix, angle * 90,1)
            
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

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, activation=sactivation, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(num_filters, 3, activation=sactivation, padding="same")(x)
    x = BatchNormalization()(x)

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    p = Dropout(0.5)(p)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = UpSampling2D((2, 2))(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def create_model(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    d3 = decoder_block(p3, s3, 256)
    d2 = decoder_block(p2, s2, 128)
    d3 = decoder_block(d2, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d3)

    model = Model(inputs, outputs, name="UNET")

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
            optimizer=Adam(lr), 
            metrics=[
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.MeanIoU(num_classes=2),  
                'accuracy'
            ])
    return model

def calculate_metrics(y_true, y_pred):
    # Assuming y_true and y_pred are 1D arrays of the same length of binary values
    binarizer = Binarizer()

    y_true_binary = binarizer.transform(y_true.reshape(-1, 1))
    y_pred_binary = binarizer.transform(y_pred.reshape(-1, 1))

    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)  # also known as recall
    specificity = tn / (tn + fp)

    return accuracy, sensitivity, specificity

def predict_from_model(path, model, lev,ground_truth_path):
    image = read_image(path)
    
    patches = []
    for x in range(0, image.shape[0] - PART_SIZE + 1, PART_SIZE):
        for y in range(0, image.shape[1] - PART_SIZE + 1, PART_SIZE):
            patch = image[x:x+PART_SIZE, y:y+PART_SIZE, :]
            patches.append(patch)
    
    patches = np.array(patches)
    predictions = model.predict(patches)
    
    predicted_image = np.zeros((image.shape[0], image.shape[1]))
    
    i = 0
    for x in range(0, image.shape[0] - PART_SIZE + 1, PART_SIZE):
        for y in range(0, image.shape[1] - PART_SIZE + 1, PART_SIZE):
            predicted_patch = predictions[i]
            predicted_patch = np.squeeze(predicted_patch, axis=-1)
            binary_mask = (predicted_patch > lev).astype(np.uint8) * 255
            predicted_image[x:x+PART_SIZE, y:y+PART_SIZE] = binary_mask
            i += 1
    
    ground_truth_image = Image.open(ground_truth_path)
    ground_truth_image = ground_truth_image.resize((PART_SIZE, PART_SIZE))
    ground_truth_array = np.array(ground_truth_image)
    accuracy, sensitivity, specificity = calculate_metrics(ground_truth_array.flatten(), predicted_image.flatten())
    print(f"Accuracy: {accuracy:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}")
            
    plt.imshow(predicted_image, cmap='gray')
    plt.show()

if sys.argv[1] == '1':
    file_list = ['Image_01L.jpg','Image_02L.jpg','Image_03L.jpg','Image_04L.jpg','Image_05L.jpg','Image_06L.jpg',
             'Image_07L.jpg','Image_08L.jpg','Image_09L.jpg','Image_10L.jpg','Image_11L.jpg','Image_12L.jpg',
             'Image_13L.jpg','Image_01R.jpg','Image_02R.jpg','Image_03R.jpg','Image_04R.jpg','Image_05R.jpg',
             'Image_06R.jpg','Image_07R.jpg','Image_08R.jpg','Image_09R.jpg','Image_10R.jpg','Image_11R.jpg',
             'Image_12R.jpg','Image_13R.jpg','Image_14R.jpg','Image_14L.jpg']

    X_test, Y_test, X_train, Y_train = load_data(file_list, patch_size=(PART_SIZE, PART_SIZE))
    model = create_model((PART_SIZE, PART_SIZE, 3))
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batches, epochs=epochs)
    model.save('model_1.h5')

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
else:
    model = load_model('model.h5')

for i in range(1, 10):
    predict_from_model("Images\Image_14L.jpg", model, i/10,"Images\Image_14R_1stHO.png")