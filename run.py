# This will be image analysis of the Fashion MNIST dataset.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.Keras.datasets import fashion_mnist

# TensorFlow makes it easy to create ML models that can run in different environments.

# Setting a grid style on seaborn to plot the images
sns.set_style('whitegrid')

print("Starting app ...")

# Loading the data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Checking the shape of the data
print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of x_test: ", x_test.shape)
print("Shape of y_test: ", y_test.shape)

# Class names for Fashion MNIST labels
class_names = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# This function displays a grid of individual sample images from each class.
# It helps us see examples of the actual images before looking at average images.
def show_sample_images(images, labels, class_names, samples_per_class=5):
    plt.figure(figsize=(samples_per_class * 2, len(class_names) * 2))

    for class_label, class_name in class_names.items():
        # Find all image positions where the label matches the current class
        class_indices = np.where(labels == class_label)[0]

        # Select the first few images from this class
        selected_indices = class_indices[:samples_per_class]

        for i, image_index in enumerate(selected_indices):
            plot_number = class_label * samples_per_class + i + 1

            plt.subplot(len(class_names), samples_per_class, plot_number)
            plt.imshow(images[image_index], cmap="gray")
            plt.axis("off")

            # Add the class name to the first image in each row
            if i == 0:
                plt.ylabel(class_name, fontsize=10)

    plt.suptitle("Sample images from each Fashion MNIST class", fontsize=16)
    plt.tight_layout()
    plt.show()


# Show a grid of individual images from each class
show_sample_images(x_train, y_train, class_names, samples_per_class=5)


# This function receives the images and labels, then calculates
# the average image for one selected product class.
def analyse_product_images(images, labels, target_label, product_name):
    print("Target label:", target_label)

    # Create a True/False mask for the selected label
    boolean_mask = (labels == target_label)
    print(boolean_mask)

    # Select only the images that match the target label
    subset_images = images[boolean_mask]

    # Calculate the average image for the selected class
    avg_image = np.mean(subset_images, axis=0)

    # Display the average image
    plt.figure(figsize=(5, 5))
    plt.imshow(avg_image, cmap='viridis')
    plt.title(f'Average image: {product_name} (Label {target_label})', fontsize=14)
    plt.axis('off')
    plt.show()


# x_train = total training images for the products
# y_train = labels, for example 0 for T-shirt/top or 9 for ankle boot

# Analysing different product categories to see whether the average image
# roughly matches what that product should look like.
analyse_product_images(x_train, y_train, target_label=0, product_name='T-shirt/top')
analyse_product_images(x_train, y_train, target_label=9, product_name='Ankle boot')
analyse_product_images(x_train, y_train, target_label=1, product_name='Trouser')
analyse_product_images(x_train, y_train, target_label=2, product_name='Pullover')
analyse_product_images(x_train, y_train, target_label=3, product_name='Dress')
analyse_product_images(x_train, y_train, target_label=4, product_name='Coat')
analyse_product_images(x_train, y_train, target_label=5, product_name='Sandal')