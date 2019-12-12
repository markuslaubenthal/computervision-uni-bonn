import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import cv2
import sklearn
import sklearn.decomposition
import random
import matplotlib.pylab as plt


def main():
    random.seed(0)
    np.random.seed(0)

    # Loading the LFW dataset
    lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw.images.shape
    X = lfw.data
    n_pixels = X.shape[1]
    y = lfw.target  # y is the id of the person in the image

    # splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # Compute the PCA
    # TODO
    n_components = 100
    pca = sklearn.decomposition.PCA(n_components=n_components)
    pca.fit(X)

    print(pca.singular_values_)
    print(pca.components_)

    cov_matrix = pca.get_covariance()

    e1 = pca.components_[0]
    dingdong = np.dot(e1.T, np.dot(cov_matrix, e1))

    

    # Visualize Eigen Faces
    # TODO

    # Compute reconstruction error
    # TODO

    # Perform face detection
    # TODO

    # Perform face recognition
    # TODO

if __name__ == '__main__':
    main()
