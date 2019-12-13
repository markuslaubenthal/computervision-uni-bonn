import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import cv2
import sklearn
import sklearn.decomposition
import random
import matplotlib.pylab as plt


def display(img):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    # ax.plot(pointss[:,0],pointss[:,1],c='red')
    # ax.plot(points[:,0],points[:,1],c='green')
    plt.show()

def main():
    random.seed(0)
    np.random.seed(0)


    # Loading the LFW dataset
    lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw.images.shape
    X = lfw.data
    n_pixels = X.shape[1]
    y = lfw.target  # y is the id of the person in the image


    test_images_name = ['boris.jpg', 'merkel.jpg', 'obama.jpg', 'putin.jpg', 'trump.jpg']
    test_images = []
    for img in test_images_name:
        img = cv2.imread('./data/exercise1/detect/face/' + img, cv2.IMREAD_GRAYSCALE)
        img = (cv2.resize(img, (w, h)).flatten())
        test_images.append(img)
    test_images = np.array(test_images).astype(np.float32)

    object_images_name = ['cat.jpg', 'dog.jpg', 'flag.jpg', 'flower.jpg', 'monkey.jpg']
    object_images = []
    for img in object_images_name:
        img = cv2.imread('./data/exercise1/detect/other/' + img, cv2.IMREAD_GRAYSCALE)
        img = (cv2.resize(img, (w, h)).flatten())
        object_images.append(img)
    object_images = np.array(object_images)

    # test_images = object_images

    # splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # Compute the PCA
    # TODO
    n_components = 100
    pca = sklearn.decomposition.PCA(n_components=n_components)
    pca.fit(X)

    # Visualize Eigen Faces
    # TODO
    fig = plt.figure()
    for i, eigenvector in enumerate(pca.components_[:10]):
        ax = plt.subplot(4, 3, i + 1)
        ax.imshow((eigenvector).reshape(h,w), cmap='gray')

    plt.show()

    # Compute reconstruction error
    # TODO

    eigenvectors = pca.components_
    mean = pca.mean_

    def calculate_coefficients(image, mean, eigenvectors):
        diff = image - mean
        coefficients = eigenvectors.dot(diff)
        return coefficients

    coefficients = calculate_coefficients(test_images[0], mean, eigenvectors)

    display((pca.mean_ + (eigenvectors.T * coefficients).sum(axis=1)).reshape(h,w))


    # Perform face detection
    # TODO


    # Perform face recognition
    # TODO

if __name__ == '__main__':
    main()
