import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import time
import sys


def display(img):
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def plot(img):
    plt.imshow(img, cmap="gray")
    plt.show()

if __name__ == '__main__':
    img_path = sys.argv[1]


#    =========================================================================
#    ==================== Task 1 =================================
#    =========================================================================
    print('Task 1:');
    # Intelgral Image
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img_integral = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for k in range(img.shape[1]):
            top, left, topleft = 0,0,0
            if(k - 1 >= 0):
                left = img_integral[i][k-1]
            if(i - 1 >= 0):
                top = img_integral[i-1][k]
            if(i -1 >= 0 and k-1 >= 0):
                topleft = img_integral[i-1][k-1]
            img_integral[i][k] = left + top - topleft + np.mean(img[i][k])

    output = (img_integral / np.max(img_integral) * 255).astype(np.uint8)

    # b)
    sum = 0
    n_pixels = img.shape[0] * img.shape[1]
    for i in range(img.shape[0]):
        for k in range(img.shape[1]):
            sum = sum + np.mean(img[i][k])
    mean = sum / n_pixels
    print(mean)

    cv_integral = cv.integral(img)
    mean = cv_integral[cv_integral.shape[0] -1][cv_integral.shape[1] -1] / n_pixels
    print(mean)

    mean = img_integral[img_integral.shape[0] -1][img_integral.shape[1] -1] / n_pixels
    print(mean)

    # c)
    rand_y = np.random.randint(img.shape[0] - 101, size=10)
    rand_x = np.random.randint(img.shape[1] - 101, size=10)
    patch_size = 100
    for index in range(10):
        y = rand_y[index]
        x = rand_x[index]
        mean = img_integral[y][x] + img_integral[y + 100][x + 100] - img_integral[y][x + 100] - img_integral[y + 100][x]
        mean = mean / patch_size ** 2
    for index in range(10):
        y = rand_y[index]
        x = rand_x[index]
        mean = img_integral[y][x] + img_integral[y + 100][x + 100] - img_integral[y][x + 100] - img_integral[y + 100][x]
        mean = mean / patch_size ** 2
    for index in range(10):
        y = rand_y[index]
        x = rand_x[index]
        mean = img_integral[y][x] + img_integral[y + 100][x + 100] - img_integral[y][x + 100] - img_integral[y + 100][x]
        mean = mean / patch_size ** 2

#    =========================================================================
#    ==================== Task 2 =================================
#    =========================================================================
    print('Task 2:');
    img_hist_eq = cv.equalizeHist(img)
    img_copy = img.copy()

    def calculateHistogram(img):
        histogram = np.zeros(256)
        for i in range(img.shape[0]):
            for k in range(img.shape[1]):
                pixelvalue = img[i][k]
                histogram[pixelvalue] += 1
        return histogram

    def calculateHistogramIntegral(histogram):
        hist_integral = np.zeros(256).astype(np.uint32)
        for i in range(256):
            prev = 0
            if(i - 1 >= 0):
                prev = hist_integral[i-1]
            hist_integral[i] = histogram[i] + prev
        return hist_integral

    histogram = calculateHistogram(img)
    hist_integral = calculateHistogramIntegral(histogram)

    hist_integral_normalized = (hist_integral / np.max(hist_integral) * 255).astype(np.uint8)
    for i in range(img.shape[0]):
        for k in range(img.shape[1]):
            img_copy[i][k] = hist_integral_normalized[img[i][k]]

    hist_eq_custom = calculateHistogram(img_copy)
    new_integral = calculateHistogramIntegral(hist_eq_custom)


    img_copy = img_copy.astype(np.int32)
    img_hist_eq = img_hist_eq.astype(np.int32)
    difference = img_hist_eq - img_copy


#    =========================================================================
#    ==================== Task 4 =================================
#    =========================================================================
    print('Task 4:');
    img_cvg = cv.GaussianBlur(img, (0,0), 2 * np.math.sqrt(2))
    sigma = 2 * np.math.sqrt(2)
    kernel_size = np.math.ceil(sigma * 3)
    kernel = np.zeros((kernel_size, kernel_size))

    def pdf(x, sigma):
        return (1.0 / np.math.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-x ** 2 / (2 * sigma ** 2))

    kernel_x = np.zeros(9)
    for i in range(kernel_size):
        kernel_x[i] = pdf(i - 4, sigma)

    kernel_x = kernel_x / kernel_x.sum()
    kernel_x = kernel_x.reshape((kernel_size, 1))
    kernel_y = np.transpose(kernel_x)
    kernel = np.matmul(kernel_x, kernel_y)
    img_custom_filter= cv.filter2D(img, -1, kernel)

    img_sepfilter = cv.sepFilter2D(img, -1, kernel_x, kernel_y)

    # display(img_cvg)
    # display(img_sepfilter)
    # display(img_custom_filter)



#    =========================================================================
#    ==================== Task 5 =================================
#    =========================================================================
    print('Task 5:');
    img_2gaussianfilters = cv.GaussianBlur(img, (0,0), 2)
    img_2gaussianfilters = cv.GaussianBlur(img_2gaussianfilters, (0,0), 2)
    img_1gaussianfilter = cv.GaussianBlur(img, (0,0), 2 * np.math.sqrt(2))

    # display(img_2gaussianfilters)
    # display(img_1gaussianfilter)
    img_2gaussianfilters = img_2gaussianfilters.astype(np.int32)
    img_1gaussianfilter = img_1gaussianfilter.astype(np.int32)
    difference = np.absolute(img_2gaussianfilters - img_1gaussianfilter)
    diff_max = np.max(difference)
    print(diff_max)


#    =========================================================================
#    ==================== Task 6 =================================
#    =========================================================================
    print('Task 6:');
    img_cpy = img.copy()
    # Salt and Pepper
    for i in range(img.shape[0]):
        for k in range(img.shape[1]):
            random = np.random.rand()
            if(random < 0.3): img_cpy[i][k] = 0
            if(random < 0.15): img_cpy[i][k] = 255
    display(img_cpy)
    best_score_gauss = 99999999
    winner_image_gauss = img_cpy.copy()
    winner_size_gauss = -1
    best_score_median = 99999999
    winner_image_median = img_cpy.copy()
    winner_size_median = -1
    best_score_bilateral = 99999999
    winner_image_bilateral = img_cpy.copy()
    winner_size_bilateral = -1
    img_cpy = img_cpy
    # Finde beste dings
    for i in np.array([1,3,5,7,9]):
        print(i)
        filtered_gauss = cv.GaussianBlur(img_cpy, (i,i), 0)
        filtered_median = cv.medianBlur(img_cpy, i)
        filtered_bilateral = cv.bilateralFilter(img_cpy, i, 100, 100)
        #Berechne absolute difference mit mean grey difference
        difference_gauss = np.absolute(img.astype(np.int32) - filtered_gauss.astype(np.int32))
        difference_median = np.absolute(img.astype(np.int32) - filtered_median.astype(np.int32))
        difference_bilateral = np.absolute(img.astype(np.int32) - filtered_bilateral.astype(np.int32))
        mean_gauss = np.mean(difference_gauss)
        mean_median = np.mean(difference_median)
        mean_bilateral = np.mean(difference_bilateral)
        if mean_gauss < best_score_gauss:
            best_score_gauss = mean_gauss
            winner_image_gauss = filtered_gauss.copy()
            winner_size_gauss = i
        if mean_median < best_score_median:
            best_score_median = mean_median
            winner_image_median = filtered_median.copy()
            winner_size_median = i
        if mean_bilateral < best_score_bilateral:
            best_score_bilateral = mean_bilateral
            winner_image_bilateral = filtered_bilateral.copy()
            winner_size_bilateral = i

    print(best_score_gauss)
    print(winner_size_gauss)
    print(best_score_median)
    print(winner_size_median)
    print(best_score_bilateral)
    print(winner_size_bilateral)
    # display(winner_image_gauss)
    # display(winner_image_median)
    # display(winner_image_bilateral)






#    =========================================================================
#    ==================== Task 7 =================================
#    =========================================================================
    print('Task 7:');





#    =========================================================================
#    ==================== Task 8 =================================
#    =========================================================================
    print('Task 8:');
