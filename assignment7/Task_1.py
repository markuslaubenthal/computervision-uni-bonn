import numpy as np
import matplotlib.pyplot as plt
import cv2

def display(img, points, pointss):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    ax.plot(pointss[:,0],pointss[:,1],c='red')
    ax.plot(points[:,0],points[:,1],c='green')
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

def findEstimates(w_, dx, dy, dt):
    w_temp = np.transpose(w_[:,::-1])
    w = (w_temp[0], w_temp[1])
    f_w = w - (dt[w] / np.sqrt(dx[w] ** 2 + dy[w] ** 2)) * np.array([dy[w], dx[w]])
    f_w = np.transpose(f_w)[:,::-1]
    not_NAN_index = np.where(np.isnan(f_w).any(axis=1) == False)
    w__ = w_.copy()
    w__[not_NAN_index] = f_w[not_NAN_index]
    return w__

def solveLeastSquares(w, w_estimate):
    A = np.zeros((w.shape[0] * w.shape[1], 6))

    for i,p in enumerate(w):
        A[2*i,0] = p[0]
        A[2*i,1] = p[1]
        A[2*i+1,2] = p[0]
        A[2*i+1,3] = p[1]
        A[2*i,4] = 1
        A[2*i+1,5] = 1

    b = w_estimate.reshape(w_estimate.shape[0] * w_estimate.shape[1])
    x,res,rank,s = np.linalg.lstsq(A,b)

    transformationMat = np.array([[x[0], x[1]],[x[2], x[3]]])
    translationVec = np.array([x[4], x[5]])
    return transformationMat, translationVec

def task_1(i, w, e, dx, dy, dt):
    v = w.copy()
    for xxxx in range(6):
        w_estimate = findEstimates(w, dx, dy, dt)
        transformationMat, translationVec = solveLeastSquares(w, w_estimate)
        print(transformationMat, translationVec)
        w = (transformationMat @ w.transpose()).transpose() + translationVec
        w = w.astype(np.int32)
        w = np.array([p for p in w if (p[1] >= 0 and p[1] < i.shape[0] and p[0] >= 0 and p[0] < i.shape[1])])

    display(dt, w, v)
    display(i, w, v)


i = importImage()
w = importLandmarks()
e = getEdges(i)
dt = getDistanceTransform(e)
dx, dy = derive(dt)
task_1(i, w, e, dx, dy, dt)
