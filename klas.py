from PIL import Image
import random
import numpy as np
from sklearn.svm import SVC 
WHITE=(255,255,255)
BLACK=(0,0,0)

def procces_part(image):
    image_array = np.array(image)
    image_array =image_array/255
    #image_array= image_array.reshape(3,n*m)
    print(image_array)
    return WHITE


nazwa_pliku = "kubus.jpg"
obraz = Image.open(nazwa_pliku)

n=5 # co ile na szerokosci
m=5 # co ile na wysokosci
upper = 0
while upper<obraz.height:
    left = 0
    if upper+m<obraz.height:
        lowwer=upper+m
    else:
        lowwer=obraz.height
    while left<obraz.width:
        if left + n <= obraz.width:
            right=left+n
        else:
            right=obraz.width
        #print(left, upper, right, lowwer)
        subimage = obraz.crop((left, upper, right, lowwer))
        procces_part(subimage)
        left+=n
    upper+=m

