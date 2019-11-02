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
    accumulator = np.zeros((int(180 / theta_step_sz), int(np.linalg.norm(img_edges.shape) / d_resolution)))
    detected_lines = []
    '''
    ...
    your code ...
    ...
    '''
    return detected_lines, accumulator


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/shapes.png')
    img_gray = None # convert the image into grayscale
    edges = None # detect the edges
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
    # clusters = [[] for i in range(k)]

    # initialize centers using some random points from data
    # ....
    for i in range(k):
        random_index = np.random.randint(0, data.shape[0])
        centers[i] = data[random_index]

    convergence = False
    iterationNo = 0
    while not convergence:
        # Create empty Clusters
        clusters = [[] for i in range(k)]
        # assign each point to the cluster of closest center
        # ...
        for d_index, datapoint in enumerate(data):
            # Calculate euclidean distance for one point to all the clusters
            euclidean_distance = np.linalg.norm(datapoint - centers, axis=1)
            # Select the cluster with the minimum distance to the point
            cluster_index = np.argmin(euclidean_distance)
            # Assign the point to the cluster
            clusters[cluster_index].append(datapoint)
            index[d_index] = cluster_index


        # update clusters' centers and check for convergence
        # ...
        new_centers = centers.copy()
        # For every cluster calculate its mean and set it as new center
        for c_index, cluster in enumerate(clusters):
            new_centers[c_index] = np.mean(cluster, axis=0)
        # Check if something changed. If yes, continue. If no, convergence = True
        if(np.sum(centers - new_centers) == 0):
            convergence = True
        # Update Centers for new iteration
        centers = new_centers

        iterationNo += 1
        print('iterationNo = ', iterationNo)
        if(iterationNo == 30): convergence = True

    return index, centers


def task_3_a():
    print("Task 3 (a) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''
    # Reshape Image to Intensity Image as list of pixels
    data = img.copy().reshape((img.shape[0] * img.shape[1], 3))
    data = np.mean(data, axis=1).reshape(data.shape[0],1)

    for k in [2,4,6]:
        # calculate kmeans for k=2,4,6
        kmeans = myKmeans(data, k)
        indices = kmeans[0]
        img_copy = data.copy().astype(np.uint8)
        # Iterate over clusters and replace pixels with Cluster Value
        for cluster_id, cluster in enumerate(kmeans[1]):
            cluster_indices = np.where(indices == cluster_id)
            img_copy[cluster_indices] = cluster[0:3]
        #Display the image
        display(img_copy.reshape(img.shape[0], img.shape[1], 1))

def task_3_b():
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''
    # Reshape to a list of RGB pixels
    data = img.copy().reshape((img.shape[0] * img.shape[1], 3))
    for k in [2,4,6]:
        # Calculate kmeans for k = 2,4,6
        kmeans = myKmeans(data, k)
        indices = kmeans[0]
        img_copy = img.reshape(img.shape[0] * img.shape[1], 3)
        # Iterate over clusters and replace pixels with Cluster Value
        for cluster_id, cluster in enumerate(kmeans[1]):
            cluster_indices = np.where(indices == cluster_id)
            img_copy[cluster_indices] = cluster[0:3]
        display(img_copy.reshape(img.shape[0], img.shape[1], 3))


def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''
    data = img.copy()
    # Create new Dataset and expand every pixel from RGB to RGBXY
    data = np.zeros((img.shape[0] * img.shape[1], 5))
    # Set RGB Values for every data point
    data[:,0:3] = img.reshape(img.shape[0] * img.shape[1], 3)

    # Set X,Y Coordinates for every data point
    for px_index, px_val in enumerate(data):
        data[px_index,3:5] = np.array([px_index // img.shape[1] + 1, px_index % img.shape[1] + 1])
        # Scale Image Coordinates to RGB Interval [0,255]
        # Also keep the x:y ratio
        data[px_index,3] = data[px_index,3] / np.maximum(img.shape[1], img.shape[0]) * 255
        data[px_index,4] = data[px_index,4] / np.maximum(img.shape[1], img.shape[0]) * 255

    for k in [2,4,6]:
        # Calculate kmeans for k = 2,4,6
        kmeans = myKmeans(data, k)
        indices = kmeans[0]
        img_copy = img.reshape(img.shape[0] * img.shape[1], 3)
        # Iterate over clusters and replace pixels with Cluster Value
        for cluster_id, cluster in enumerate(kmeans[1]):
            cluster_indices = np.where(indices == cluster_id)
            img_copy[cluster_indices] = cluster[0:3]
        display(img_copy.reshape(img.shape[0], img.shape[1], 3))


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
    #task_1_a()
    #task_1_b()
    #task_2()
    #task_3_a()
    #task_3_b()
    #task_3_c()
    # task_4_a()
