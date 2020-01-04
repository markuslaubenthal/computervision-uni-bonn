# Authors: Markus Laubenthal, Lennard Alms

import numpy as np
import os
import time
import cv2 as cv
import matplotlib.pyplot as plt


def load_FLO_file(filename):
    assert os.path.isfile(filename), 'file does not exist: ' + filename
    flo_file = open(filename,'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    assert magic == 202021.25,  'Magic number incorrect. .flo file is invalid'
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    data = np.fromfile(flo_file, np.float32, count=2*w[0]*h[0])
    flow = np.resize(data, (int(h[0]), int(w[0]), 2))
    flo_file.close()
    return flow

def display(img):
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def unit_vector(vector):
    return vector / np.linalg.norm(vector, axis = 2)[:,:,None]

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = np.squeeze(unit_vector(v2))
    if(np.count_nonzero(v1) == 0):
        return 0
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))) / 360 * 179

class OpticalFlow:
    def __init__(self):
        # Parameters for Lucas_Kanade_flow()
        self.EIGEN_THRESHOLD = 0.01 # use as threshold for determining if the optical flow is valid when performing Lucas-Kanade
        self.WINDOW_SIZE = [25, 25] # the number of points taken in the neighborhood of each pixel

        # Parameters for Horn_Schunck_flow()
        self.EPSILON= 0.002 # the stopping criterion for the difference when performing the Horn-Schuck algorithm
        self.MAX_ITERS = 1000 # maximum number of iterations allowed until convergence of the Horn-Schuck algorithm
        self.ALPHA = 1.0 # smoothness term

        # Parameter for flow_map_to_bgr()
        self.UNKNOWN_FLOW_THRESH = 1000

        self.prev = None
        self.next = None

    def next_frame(self, img):
        self.prev = self.next
        self.next = img

        if self.prev is None:
            return False

        frames = np.float32(np.array([self.prev, self.next]))
        frames /= 255.0

        #calculate image gradient
        self.Ix = cv.Sobel(frames[0], cv.CV_32F, 1, 0, 3)
        self.Iy = cv.Sobel(frames[0], cv.CV_32F, 0, 1, 3)
        self.It = frames[1]-frames[0]

        return True

    #***********************************************************************************
    # function for converting flow map to to BGR image for visualisation
    # return bgr image
    def flow_map_to_bgr(self, flow):
        flow_bgr = None

        hsv_colorspace = np.ones((flow.shape[0], flow.shape[1], 3)).astype(np.float32) * 255
        flow_bgr = np.zeros((flow.shape[0], flow.shape[1], 3)).astype(np.uint8)

        hsv_colorspace[:,:,0] = angle_between(flow, np.array([1,0])[None,None,:])
        hsv_colorspace[:,:,2] = np.linalg.norm(flow, axis = 2)
        hsv_colorspace[:,:,2] = np.clip(hsv_colorspace[:,:,2] / np.max(hsv_colorspace[:,:,2]) * 255, 0, 255)

        # Since pyplot takes RGB we decided to convert to RGB instead of BGR
        flow_bgr = cv.cvtColor(hsv_colorspace.astype(np.uint8), cv.COLOR_HSV2RGB)
        return flow_bgr

    #***********************************************************************************
    # implement Lucas-Kanade Optical Flow
    # returns the Optical flow based on the Lucas-Kanade algorithm and visualisation result
    def Lucas_Kanade_flow(self):
        flow = None
        Ix = self.Ix
        Iy = self.Iy
        It = self.It

        #Precompute Variables
        Ixx = Ix ** 2
        Ixy = Ix * Iy
        Iyy = Iy ** 2
        Ixt = Ix * It
        Iyt = Iy * It

        # Filter the images with a 25x25 convolution
        conv_filter = np.ones((25,25))
        Ixx_sum = cv.filter2D(Ixx, -1, conv_filter)
        Ixy_sum = cv.filter2D(Ixy, -1, conv_filter)
        Iyy_sum = cv.filter2D(Iyy, -1, conv_filter)
        Ixt_sum = cv.filter2D(Ixt, -1, conv_filter)
        Iyt_sum = cv.filter2D(Iyt, -1, conv_filter)

        flow = np.zeros((Ix.shape[0], Ix.shape[1], 2))

        # solve Least squares for every pixel
        for y in range(Ix.shape[0]):
            for x in range(Ix.shape[1]):
                lhs = np.array([[Ixx_sum[y,x], Ixy_sum[y,x]], [Ixy_sum[y,x], Iyy_sum[y,x]]])
                rhs = -np.array([Ixt_sum[y,x], Iyt_sum[y,x]])
                res, r, rank, s = np.linalg.lstsq(lhs, rhs, rcond=None)
                flow[y][x] = res
        flow_bgr = self.flow_map_to_bgr(flow)
        return flow, flow_bgr

    #***********************************************************************************
    # implement Horn-Schunck Optical Flow
    # returns the Optical flow based on the Horn-Schunck algorithm and visualisation result
    def Horn_Schunck_flow(self):
        flow = None
        alpha = 1

        # Precomputation of Image
        Ix = self.Ix
        Ixx = Ix ** 2
        Iy = self.Iy
        Iyy = Iy ** 2
        It = self.It

        # Init u and v
        u = np.zeros((Ix.shape[0], Ix.shape[1]))
        v = np.zeros((Ix.shape[0], Ix.shape[1]))

        # Laplacian filter
        laplacian_conv = np.array([
            [0,1,0],
            [1,-4,1],
            [0,1,0]
        ]) * (1/4)

        cont = True
        while cont:
            # get the laplacian filtered image of previous u and v
            delta_u = cv.filter2D(u, -1, laplacian_conv)
            delta_v = cv.filter2D(v, -1, laplacian_conv)
            # calculate u prime and v prime
            u_prime = u + delta_u
            v_prime = v + delta_v

            # Formula from exercise Sheet
            numerator_same = (Ix * u_prime + Iy * v_prime + It)
            numerator_u = Ix * numerator_same
            numerator_v = Iy * numerator_same
            denominator = alpha ** 2 + Ixx + Iyy

            # Update step
            u_next = u_prime - numerator_u / denominator
            v_next = v_prime - numerator_v / denominator

            # Calculate difference
            diff_u = np.absolute(u - u_next)
            diff_v = np.absolute(v - v_next)
            sum = diff_u.sum() + diff_v.sum()

            # Stop if update step becomes very small
            if(sum < 0.02):
                cont = False

            u = u_next
            v = v_next

        flow = np.zeros((Ix.shape[0], Ix.shape[1], 2))
        flow[:,:,0] = u
        flow[:,:,1] = v



        flow_bgr = self.flow_map_to_bgr(flow)
        return flow, flow_bgr

    #***********************************************************************************
    #calculate the angular error here
    # return average angular error and per point error map
    def calculate_angular_error(self, estimated_flow, groundtruth_flow):
        aae = None
        aae_per_point = None
        # get height and width
        h = estimated_flow.shape[0]
        w = estimated_flow.shape[1]
        # rename and reshape into 2d array
        uv = estimated_flow.reshape((h * w, 2))
        ucvc = groundtruth_flow.reshape((h * w, 2))
        # Unscramble variables for calculation
        u = uv[:,0]
        v = uv[:,1]
        uc = ucvc[:,0]
        vc = ucvc[:,1]

        # Implementation of the formula in slides
        numerator = uc*u + vc*v + 1
        denominator = (uc**2 + vc**2 + 1)*(u**2+v**2+1)
        denominator = np.sqrt(denominator)
        aae_per_point = np.arccos(numerator / denominator)
        aae = (1/aae_per_point.shape[0]) * np.sum(aae_per_point)
        aae_per_point = aae_per_point.reshape((h,w))
        
        return aae, aae_per_point


if __name__ == "__main__":

    data_list = [
        'data/frame_0001.png',
        'data/frame_0002.png',
        'data/frame_0007.png',
    ]

    gt_list = [
        './data/frame_0001.flo',
        './data/frame_0002.flo',
        './data/frame_0007.flo',
    ]

    Op = OpticalFlow()

    for (i, (frame_filename, gt_filemane)) in enumerate(zip(data_list, gt_list)):
        groundtruth_flow = load_FLO_file(gt_filemane)
        img = cv.cvtColor(cv.imread(frame_filename), cv.COLOR_BGR2GRAY)
        if not Op.next_frame(img):
            continue

        flow_lucas_kanade, flow_lucas_kanade_bgr = Op.Lucas_Kanade_flow()
        aae_lucas_kanade, aae_lucas_kanade_per_point = Op.calculate_angular_error(flow_lucas_kanade, groundtruth_flow)
        print('Average Angular error for Luacas-Kanade Optical Flow: %.4f' %(aae_lucas_kanade))

        flow_horn_schunck, flow_horn_schunck_bgr = Op.Horn_Schunck_flow()
        aae_horn_schunk, aae_horn_schunk_per_point = Op.calculate_angular_error(flow_horn_schunck, groundtruth_flow)
        print('Average Angular error for Horn-Schunck Optical Flow: %.4f' %(aae_horn_schunk))

        flow_bgr_gt = Op.flow_map_to_bgr(groundtruth_flow)

        fig = plt.figure(figsize=(img.shape))

        # Display
        fig.add_subplot(2, 3, 1)
        plt.imshow(flow_bgr_gt)
        fig.add_subplot(2, 3, 2)
        plt.imshow(flow_lucas_kanade_bgr)
        fig.add_subplot(2, 3, 3)
        plt.imshow(aae_lucas_kanade_per_point)
        fig.add_subplot(2, 3, 4)
        plt.imshow(flow_bgr_gt)
        fig.add_subplot(2, 3, 5)
        plt.imshow(flow_horn_schunck_bgr)
        fig.add_subplot(2, 3, 6)
        plt.imshow(aae_horn_schunk_per_point)
        plt.show()

        print("*"*20)
