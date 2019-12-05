import numpy as np
import matplotlib.pyplot as plt
import cv2

def display(img, points):
    plt.imshow(img, cmap='gray')
    plt.scatter(points[:,0],points[:,1])
    plt.show()

def importImage():
    image = cv2.imread('data/hand.jpg')
    image = image / image.max()
    return image.astype(np.float32)

def importLandmarks():
    with open('data/hand_landmarks.txt') as f:
        lines = [line.rstrip('\n')[1:-1] for line in f]
        landmarks = [np.fromstring(line, dtype=int, sep=',') for line in lines]
        return np.array(landmarks)

def getEdges(i):
    return cv2.Canny((i*255).astype(np.uint8), 80, 100)

def derive(i):
    x_kernel = np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]])
    y_kernel = np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]])
    dx = cv2.filter2D(i, -1, x_kernel)
    dy = cv2.filter2D(i, -1, y_kernel)
    return dx, dy

def getDistanceTransform(i):
    return cv2.distanceTransform(cv2.bitwise_not(i), cv2.DIST_L1, 3)


def task_1(i, w_, e, dx, dy, dt):
    # Find estimate landmarks
    #display(i,w_)
    w_temp = np.transpose(w_[:,::-1])
    w = (w_temp[0], w_temp[1])
    print((dt[w] / np.sqrt(dx[w] ** 2 + dy[w] ** 2)) * dx[w] * dy[w])
    print(w)
    f_w = w - (dt[w] / np.sqrt(dx[w] ** 2 + dy[w] ** 2)) * dx[w] * dy[w]
    print(f_w)
    f_w = np.transpose(f_w)[:,::-1]
    not_NAN_index = np.where(np.isnan(f_w).any(axis=1) == False)
    w_[not_NAN_index] = f_w[not_NAN_index]
    #display(i,w_)
    # Least squares method to find psi




i = importImage()
w = importLandmarks()
e = getEdges(i)
dt = getDistanceTransform(e)
dx, dy = derive(dt)
task_1(i, w, e, dx, dy, dt)
