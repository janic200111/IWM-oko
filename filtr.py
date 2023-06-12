import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi
from skimage.io import imread
from skimage import exposure
from sklearn.metrics import confusion_matrix, accuracy_score

# Definiowanie list do przechowywania wyników
accuracies = []
sensitivities = []
specificities = []

#for i in range(1,40):
    #for j in range(1,30):
# Wczytanie obrazu
image = imread('Images/Image_01L.jpg', as_gray=True)
reference_image = imread('Images/Image_01L_1stHO.png', as_gray=True)

# Rozmycie Gaussowskie
image = cv2.GaussianBlur(image, (5, 5), 0)

# Korekcja gamma 
gamma_corrected = exposure.adjust_gamma(image, 3.9)

# Równoważenie histogramu
hist_equalized = exposure.equalize_hist(gamma_corrected)

# Zastosowanie filtru Frangi’ego
filtered_image = frangi(hist_equalized)

# Końcowa obróbka obrazu
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(filtered_image, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

threshold_value = 0.0015
_, binary_image = cv2.threshold(closing, threshold_value, 1, cv2.THRESH_BINARY)

tn, fp, fn, tp = confusion_matrix(reference_image.flatten(), binary_image.flatten()).ravel()

# Obliczanie miar
accuracy = accuracy_score(reference_image.flatten(), binary_image.flatten())
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"Accuracy: {accuracy}")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
# Dodawanie wyników do list
accuracies.append(accuracy)
sensitivities.append(sensitivity)
specificities.append(specificity)
"""
# Tworznie wykresów
plt.figure(figsize=(12,8))

plt.plot(accuracies, label='Accuracy')
plt.plot(sensitivities, label='Sensitivity')
plt.plot(specificities, label='Specificity')

plt.legend(loc='lower right')
plt.title('Performance metrics over iterations')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.show()


        # Wyświetlenie oryginalnego obrazu i wyniku
        fig, ax = plt.subplots(3, 2, figsize=(15,22))

        ax[0,0].imshow(image, cmap=plt.cm.gray)
        ax[0,0].set_title('Oryginalny obraz')

        ax[0,1].imshow(gamma_corrected, cmap=plt.cm.gray)
        ax[0,1].set_title('Obraz po korekcji gamma')

        ax[1,0].imshow(hist_equalized, cmap=plt.cm.gray)
        ax[1,0].set_title('Obraz po równoważeniu histogramu')

        ax[1,1].imshow(filtered_image, cmap=plt.cm.gray)
        ax[1,1].set_title('Obraz po zastosowaniu filtru Frangi’ego')

        ax[2,0].imshow(closing, cmap=plt.cm.gray)
        ax[2,0].set_title('Obraz po operacji zamknięcia')
"""
plt.imshow(binary_image, cmap='gray')
plt.show()

