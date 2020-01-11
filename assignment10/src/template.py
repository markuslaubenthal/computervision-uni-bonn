import cv2
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt

NUM_IMAGES=14
NUM_Boards = NUM_IMAGES
image_prefix = "../images/"
image_suffix = ".png"
images_files_list = [osp.join(image_prefix, f) for f in os.listdir(image_prefix)
                     if osp.isfile(osp.join(image_prefix, f)) and f.endswith(image_suffix)]
board_w = 10
board_h = 7
board_size = (board_w, board_h)
board_n = board_w * board_h
img_shape = (0,0)
obj = []
for ptIdx in range(0, board_n):
    obj.append(np.array([[ptIdx/board_w, ptIdx%board_w, 0.0]],np.float32))
obj = np.vstack(obj)

images = []
images_gray = []
imagepoints = []

def task1():
    #implement your solution
    objectpoints = []
    imagepoints = []

    objp = np.zeros((board_h*board_w,3), np.float32)
    objp[:,:2] = np.mgrid[0:board_h,0:board_w].T.reshape(-1,2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for i, img in enumerate(images):
        gray = images_gray[i]
        ret, corners = cv2.findChessboardCorners(gray, (board_h,board_w),None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners, (11,11),(-1,-1), criteria)
            imagepoints.append(corners2)
            objectpoints.append(objp)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (board_h, board_w), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(1)
    return imagepoints, objectpoints

def task2(imagePoints, objectPoints):
    #implement your solution
    shape = images_gray[0].shape[::-1]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, shape,None,None)
    print("DistortionMatrix Shape: ", mtx.shape)
    return mtx, dist, rvecs, tvecs

def task3(imagePoints, objectPoints, CM, D, rvecs, tvecs):
    #implement your solution
    pass

def task4(CM, D):
    #implement your solution
    pass

def task5(CM, rvecs, tvecs):
    #implement your solution
    pass

def main():
    #Showing images
    for img_file in images_files_list:
        print(img_file)
        img = cv2.imread(img_file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)
        images_gray.append(img_gray)
        # cv2.imshow("Task1", img)
        # cv2.waitKey(10)

    imagePoints, objectPoints = task1() #Calling Task 1

    CM, D, rvecs, tvecs = task2(imagePoints, objectPoints) #Calling Task 2

    task3(imagePoints, objectPoints, CM, D, rvecs, tvecs)  # Calling Task 3

    task4(CM, D) # Calling Task 4

    task5(CM, rvecs, tvecs) # Calling Task 5

    print("FINISH!")

main()
