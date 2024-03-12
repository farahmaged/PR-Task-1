"""
__author__ = "Farah"
__version__ = "0.0.0"
PR task 1
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# Feature extraction using centroids
def extract_features(image, num_blocks):
    block_size = 28 // num_blocks  # Calculate the size of each block based on the number of blocks
    features = []  # Initialize an empty list to store the extracted features

    for i in range(num_blocks):
        for j in range(num_blocks):
            # Define the coordinates for the current block
            start_row, end_row = i * block_size, (i + 1) * block_size
            start_col, end_col = j * block_size, (j + 1) * block_size

            # Extract the block from the image
            block = image[start_row:end_row, start_col:end_col]

            # Calculate the mean value of pixel intensities in the block
            block_mean = np.mean(block)

            # Append the mean value to the list of features
            features.append(block_mean)

    return features


# Split the data
train_image, test_image, train_lab, test_lab = train_test_split(train_images, train_labels, test_size=0.3)

# Extract features for each image
num_blocks = 9
train_features = [extract_features(image, num_blocks) for image in
                  train_image]  # List of feature vectors corresponding to the training set
test_features = [extract_features(image, num_blocks) for image in
                 test_image]  # List of feature vectors corresponding to the testing set

# Flatten the features for the KNN classifier
train_features_flat, test_features_flat = np.array(train_features).reshape((len(train_features), -1)), np.array(
    test_features).reshape((len(test_features), -1))


# Initialize a KNN classifier with n_neighbors=3 considering the three nearest neighbors when making predictions

classifier = KNeighborsClassifier(n_neighbors=3).fit(train_features_flat, train_lab)

# Make predictions on the feature vectors of the test set, and calculate the accuracy of the classifier's predictions by comparing them to the true labels 
accuracy = accuracy_score(test_lab, classifier.predict(test_features_flat))
print("Accuracy: " + str(round(accuracy * 100, 2)) + "%")
