#this will be image analysis of the fashion mnist dataset.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist

#tensorflow makes it easy to create ML models that can run in any environment. 

#Setting a grid on seaborn to plot the images
sns.set_style('whitegrid')

print ("Starting app ...")

#loading the data - populating values
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#checking the shape of the data
#print ("Shape of x_train: ", x_train.shape)
#print ("Shape of y_train: ", y_train.shape)
#print ("Shape of x_test: ", x_test.shape)
#print ("Shape of y_test: ", y_test.shape)

#we want the function to recieve the value (image) and decide what is being analysed
def analyse_product_images (images, labels, target_label, product_name):
    print(target_label)

    boolean_mask = (labels == target_label)
    print (boolean_mask)

    subset_images = images[boolean_mask]

    avg_image = np.mean(subset_images, axis = 0)

    plt.figure(figsize = (5,5))
    plt.imshow(avg_image, cmap = 'viridis')
    plt.title(f'Average image: {product_name} (Label {target_label})', fontsize = 14)
    plt.axis('off')
    plt.show()


#xtrain = total images for the product (e.g. tshirt, shoes)
# #Ytrain = label(e.g 0 for t-shirt or 9 for ankle boots)

#Analysing T-shirt to see if the average uploaded t-shirt images remotely matches what a T-shirt should look like.
analyse_product_images(x_train,y_train, target_label = 0, product_name = 'T-shirt')
analyse_product_images(x_train,y_train, target_label = 9, product_name = 'Ankle boots')
analyse_product_images(x_train,y_train, target_label = 1, product_name = 'Trouser')
