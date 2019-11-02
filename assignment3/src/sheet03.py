import numpy as np
import cv2 as cv
import random


def display(img):
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

##############################################
#     Task 1        ##########################
##############################################


def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread('../images/shapes.png')
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    '''
    ...
    your code ...
    ...
    '''
    edges = cv.Canny(img,50,150,apertureSize = 3)
    # display(edges)
    houghLines = cv.HoughLines(edges,0.7,np.pi/180,55)
    for x in houghLines:
        rho,theta = x[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    display(img)

def myHoughLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Your implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs and the accumulator
    """
    distSteps = int(np.linalg.norm(img_edges.shape) / d_resolution)
    degSteps = int(np.pi / theta_step_sz)
    print(distSteps)
    print(degSteps)
    accumulator = np.zeros((degSteps, distSteps))
    with np.nditer(img_edges, flags=['multi_index']) as it:
        for px in it:
            if px > 3:
                for deg in range(degSteps):
                    p = (it.multi_index[0] * np.cos(deg)) + (it.multi_index[1] * np.sin(deg))
                    if(-distSteps/2 <= p and p < distSteps/2):
                        accumulator[deg, int(p + distSteps/2) ] += 1

    detected_lines = []
    with np.nditer(accumulator, flags=['multi_index']) as it2:
        for votes in it2:
            if votes >= threshold:
                detected_lines.append(it2.multi_index)

    return detected_lines, accumulator


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/shapes.png')
    edges = cv.Canny(img,50,150,apertureSize = 3)
    houghLines, accumulator= myHoughLines(edges,0.7,np.pi/180,55)

    display(accumulator/np.max(accumulator))

    for theta,rho in houghLines:
        rho = rho - accumulator.shape[1] / 2
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 10000*(-b))
        y1 = int(y0 + 10000*(a))
        x2 = int(x0 - 10000*(-b))
        y2 = int(y0 - 10000*(a))
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    display(img)

    #detected_lines, accumulator = myHoughLines(edges, 1, 2, 50)
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 2        ##########################
##############################################


def task_2():
    print("Task 2 ...")
    img = cv.imread('../images/line.png')
    img_gray = None # convert the image into grayscale
    edges = None # detect the edges
    theta_res = None # set the resolution of theta
    d_res = None # set the distance resolution
    #_, accumulator = myHoughLines(edges, d_res, theta_res, 50)
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 3        ##########################
##############################################


def myKmeans(data, k):
    """
    Your implementation of k-means algorithm
    :param data: list of data points to cluster
    :param k: number of clusters
    :return: centers and list of indices that store the cluster index for each data point
    """
    centers = np.zeros((k, data.shape[1]))
    index = np.zeros(data.shape[0], dtype=int)
    clusters = [[] for i in range(k)]

    # initialize centers using some random points from data
    # ....
    for i in range(data.shape[1]):
        subset = data[:,i]
        r = np.random.uniform(
            np.min(subset),
            np.max(subset),
            len(clusters))
        centers[:,i] = r

    print(centers)
    convergence = False
    iterationNo = 0
    while not convergence:
        # assign each point to the cluster of closest center
        # ...

        # update clusters' centers and check for convergence
        # ...

        iterationNo += 1
        print('iterationNo = ', iterationNo)

    return index, centers


def task_3_a():
    print("Task 3 (a) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''
    data = img.reshape((img.shape[0] * img.shape[1], 3))
    myKmeans(data, 5)

def task_3_b():
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 4        ##########################
##############################################


def task_4_a():
    print("Task 4 (a) ...")
    # construct the D matrix
    D = np.array([
    [2.2, 0,   0,   0,	0,	0,	0,	 0],
    [0,	  2.1, 0,   0,	0,	0,	0,	 0],
    [0,   0,   2.6,	0,	0,	0,	0,   0],
    [0,	  0,   0,   3,	0,	0,	0,	 0],
    [0,	  0,   0,	0,	3,	0,	0,	 0],
    [0,   0,   0,	0,	0,	3,	0,	 0],
    [0,	  0,   0,	0,	0,	0,	3.3, 0],
    [0,	  0,   0,	0,	0,	0,	0,	 2],
    ])
    # construct the W matrix
    W = np.array([
    [0,	  1,   0.2, 1,	0,	0,	0,	 0],
    [1,	  0,   0.1, 0,	1,	0,	0,	 0],
    [0.2, 0.1, 0,	1,	0,	1,	0.3, 0],
    [1,	  0,   1,   0,	0,	1,	0,	 0],
    [0,	  1,   0,	0,	0,	0,	1,	 1],
    [0,   0,   1,	1,	0,	0,	1,	 0],
    [0,	  0,   0.3,	0,	1,	1,	0,	 1],
    [0,	  0,   0,	0,	1,	0,	1,	 0],
    ])

    Dpowinv = np.sqrt(np.linalg.inv(D))

    A = np.dot(np.dot(Dpowinv,D-W), Dpowinv)

    State, EigValues, EigVectors = cv.eigen(A)

    print('Task 4 (b) ...')
    print('eigen vector to smalles eigen value:')
    print(np.round(EigVectors[0:,-1:],2))
    print('C1 = {A,B,E,G,H}, C1 = {C,D,F}')
    volC1 = 2.2 + 2.1 + 3 + 3.3 + 2
    volC2 = 2.6 + 3 + 3
    cut = 1 + 0.2 + 0.1 + 0.3 + 1
    normCut = cut/volC1 + cut/volC2
    print('the NormCut is ', normCut)

##############################################
##############################################
##############################################

if __name__ == "__main__":
    task_1_a()
    task_1_b()
    #task_2()
    #task_3_a()
    #task_3_b()
    #task_3_c()
    #task_4_a()
