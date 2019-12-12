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
    test_images = np.array(test_images)

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
    cov_matrix = pca.get_covariance()
    fig = plt.figure()
    for i, eigenvector in enumerate(pca.components_[:10]):
        ax = plt.subplot(4, 3, i + 1)
        ax.imshow((eigenvector).reshape(h,w), cmap='gray')

    plt.show()

    # Compute reconstruction error
    # TODO
    def calculate_coefficients(images):
        image_pca = pca.transform(images)
        return image_pca

    def calculate_coefficients_for_single_image(image):
        return pca.transform(image.reshape(1, image.shape[0]))

    # X_train_pca, X_test_pca = calculate_coefficients(X_train, X_test)
    eigenvectors = pca.components_
    coefficients = calculate_coefficients(test_images)
    obj_coefficients = calculate_coefficients(object_images)

    images_err = (test_images -
        (pca.mean_[:,None] + eigenvectors.T.dot(coefficients.T)).T
    )
    objects_err = (object_images -
        (pca.mean_[:,None] + eigenvectors.T.dot(coefficients.T)).T
    )

    error_per_image = np.linalg.norm(images_err, axis = 1)
    error_per_image_obj = np.linalg.norm(objects_err, axis = 1)

    # display(test_images[0].reshape(h,w))
    # display((pca.mean_[None, :] + coefficients.dot(eigenvectors)) [0].reshape(h,w))
    # display(test_images[1].reshape(h,w))
    # display((pca.mean_ + coefficients.dot(eigenvectors))[1].reshape(h,w))
    # display(test_images[2].reshape(h,w))
    # display((pca.mean_ + coefficients.dot(eigenvectors))[2].reshape(h,w))
    # display(test_images[3].reshape(h,w))
    # display((pca.mean_ + coefficients.dot(eigenvectors))[3].reshape(h,w))
    # display(test_images[4].reshape(h,w))
    # display((pca.mean_ + coefficients.dot(eigenvectors))[4].reshape(h,w))


    print(error_per_image)
    print(error_per_image_obj)
    # Perform face detection
    # TODO
    # Select random image
    randint = np.random.randint(2)
    image = None
    test_label = None
    if(randint == 0):
        index = np.random.randint(object_images.shape[0])
        image = object_images[index]
        error = error_per_image_obj[index]
        test_label = 'object'
    else:
        index = np.random.randint(test_images.shape[0])
        image = test_images[index]
        error = error_per_image[index]
        test_label = 'face'

    label = None
    print("The prediction is: ", end='')
    if(error > 2000):
        label = "object"
    else:
        label = "face"
    print(label)
    print("the label is:", test_label)

    display(image.reshape(h,w))

    # Perform face recognition
    # TODO

if __name__ == '__main__':
    main()
