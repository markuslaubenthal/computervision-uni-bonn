import numpy as np
import matplotlib.pyplot as plt
import cv2

def display(points,pointss):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    points = np.transpose(pointsTo2D(points))
    pointss = np.transpose(pointsTo2D(pointss))
    ax.plot(points[0,:],points[1,:],c='green')
    ax.plot(pointss[0,:],pointss[1,:],c='red')
    plt.show()

def pointsTo2D(points):
    return points.reshape(points.shape[0] // 2, 2, order='F')

def pointsTo1D(points):
    return points.flatten(order='F')


def importTrainingData():
    with open('./data/hands_aligned_train.txt.new') as f:
        lines = [line.rstrip('\n') for line in f]
        landmarks = [np.fromstring(line, dtype=int, sep=' ') for line in lines[1:]]
    return np.transpose(np.array(landmarks))

def importTestData():
    with open('./data/hands_aligned_test.txt.new') as f:
        lines = [line.rstrip('\n') for line in f]
        landmarks = [np.fromstring(line, dtype=int, sep=' ') for line in lines[1:]]
    return np.transpose(np.array(landmarks))[0]

def calcMeanShape(data):
    return data.sum(axis=0) / data.shape[0]

def calcEigenValues(data, mue):
    w = data - mue.reshape(1, mue.shape[0])
    wwt = np.transpose(w) @ w / mue.shape[0]
    return np.linalg.svd(wwt)[:2]

def findDefiningValues(eig):
    denominator = (eig).sum()
    k = 0
    eigSum = 0
    for i in range(eig.shape[0]):
        eigSum += eig[i]
        representation = eigSum / denominator
        if representation > 0.9:
            k = i
            break
    return k

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

def task_2():
    data = importTrainingData()
    mue = calcMeanShape(data)
    u, eig = calcEigenValues(data,mue)
    k = findDefiningValues(eig)

    lostVar = eig[k:].sum() / (eig.shape[0] - k)
    phi = u[:,:k] * np.sqrt(eig[:k] - lostVar).reshape(1,k)
    h = np.array([-0.4,-0.2,0.0,0.2,0.4])
    w = mue + (phi * h.reshape(1,k)).sum(axis=1)
    return mue, k, phi

def task_3(mue, k, phi):
    data = importTestData()
    h = np.zeros(k)
    i = 0
    w = mue + (phi * h.reshape(1,k)).sum(axis=1)
    while i < 9:
        w = mue + (phi * h.reshape(1,k)).sum(axis=1)
        error = np.sqrt(np.sum((w - data) ** 2) / w.shape[0])
        display(data,w)
        print(error)
        w2d, data2d = pointsTo2D(w), pointsTo2D(data)
        transformationMat, translationVec = solveLeastSquares(w2d, data2d)
        transformationMat_inv = np.linalg.inv(transformationMat)
        data_inv = np.transpose(transformationMat_inv @ np.transpose(data2d - translationVec.reshape(1,2)))
        data_inv = pointsTo1D(data_inv)
        h = np.transpose(phi).dot((data_inv - mue) / np.sqrt(data_inv.dot(mue)))
        i += 1

mue, k, phi = task_2()

task_3(mue, k, phi)




















#
