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

    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

#    =========================================================================
#    ==================== Task 1 =================================
#    =========================================================================

    print('Task 1:');

    # a)

    # This function computes the integral image dynamically in linear time - left to right and top to bottom

    def calc_integral_image(img):
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
                img_integral[i][k] = left + top - topleft + img[i][k]
        return img_integral

    # Calculate the integral image from the source

    img_integral = calc_integral_image(img)

    # Normalize the integral image so it can be displayed as 0-255 grayscale.

    output = (img_integral / np.max(img_integral) * 255).astype(np.uint8)

    display(output)

    # b) Compute the mean intensity of the image in the three ways.

    def mean_by_summation(img):
        sum = 0
        for px in np.nditer(img):
            sum = sum + px
        mean = sum / np.size(img)
        return mean

    def mean_by_cv_integral(img):
        cv_integral = cv.integral(img)
        mean = cv_integral[cv_integral.shape[0] -1][cv_integral.shape[1] -1] / np.size(img)
        return mean

    def mean_by_own_integral(img):
        img_integral = calc_integral_image(img)
        mean = img_integral[img_integral.shape[0] -1][img_integral.shape[1] -1] / np.size(img)
        return mean

    print('mean three ways:')
    print(mean_by_summation(img))
    print(mean_by_cv_integral(img))
    print(mean_by_own_integral(img))

    # c) create random patches

    rand_x = np.random.randint(img.shape[0] - 101, size=10)
    rand_y = np.random.randint(img.shape[1] - 101, size=10)
    patch_size = 100

    # calculate using the summation method

    start = time.time()

    means = []

    for index in range(10): # For each patch top left corner x,y
        y = rand_y[index]
        x = rand_x[index]
        means.append(mean_by_summation(img[x : x + 100, y : y + 100]))

    end = time.time()

    print('time summation method:')
    print(end - start)

    # calculate using the cv integral image method

    start = time.time()

    means = []

    img_integral = cv.integral(img)

    for index in range(10): # For each patch top left corner x,y
        y = rand_y[index]
        x = rand_x[index]
        mean = img_integral[x][y] + img_integral[x + 100][y + 100] - img_integral[x][y + 100] - img_integral[x + 100][y]
        mean = mean / 10000
        means.append(mean)

    end = time.time()

    print('time cv integral image method:')
    print(end - start)

    # calculate using the own integral image method

    start = time.time()

    means = []

    img_integral = calc_integral_image(img)

    for index in range(10): # For each patch top left corner x,y
        y = rand_y[index]
        x = rand_x[index]
        mean = img_integral[x][y] + img_integral[x + 100][y + 100] - img_integral[x][y + 100] - img_integral[x + 100][y]
        mean = mean / 10000
        means.append(mean)

    end = time.time()

    print('time own integral image method:')
    print(end - start)

#    =========================================================================
#    ==================== Task 2 =================================
#    =========================================================================

    print('Task 2:');

    # a) Using the cv function for histogram equalization.

    img_hist_eq_cv = cv.equalizeHist(img)

    # b) Using an own histogram equalization function.

    img_hist_eq_own = img.copy()

    display(img_hist_eq_own)

    # Define a function to calculate the histogram.

    def calculateHistogram(img):
        histogram = np.zeros(256)
        for i in range(img.shape[0]):
            for k in range(img.shape[1]):
                pixelvalue = img[i][k]
                histogram[pixelvalue] += 1
        return histogram

    # Define a funciton to calculate the descrete intrgral of the histogram.

    def calculateHistogramIntegral(histogram):
        hist_integral = np.zeros(256).astype(np.uint32)
        for i in range(256):
            prev = 0
            if(i - 1 >= 0):
                prev = hist_integral[i-1]
            hist_integral[i] = histogram[i] + prev
        return hist_integral

    # Calculate the integral of the histogram of the original image

    histogram = calculateHistogram(img)
    hist_integral = calculateHistogramIntegral(histogram)

    # Normalize the range of the integral in Y-direction to 0-255.

    hist_integral_normalized = (hist_integral / np.max(hist_integral) * 255).astype(np.uint8)

    # For each pixel lookup the new value in the equalized histogram.

    with np.nditer(img_hist_eq_own, op_flags=['readwrite']) as iter:
        for px in iter:
            px = hist_integral_normalized[px]

    display(img_hist_eq_own)

    # Calculate the maximum pixel difference form both equalized images.

    max_diff = np.max(np.absolute(img_hist_eq_cv.astype(np.int32) - img_hist_eq_own.astype(np.int32)))

    print(max_diff)


#    =========================================================================
#    ==================== Task 4 =================================
#    =========================================================================

    print('Task 4:');

    # Set sigma

    sigma = 2 * np.math.sqrt(2)

    # a) Filter the image using the cv gaussian blur and sigma.

    img_cvg = cv.GaussianBlur(img, (0,0), sigma)

    display(img_cvg)

    # b) Filter the image using an own gaussian convolution.

    # Define a propability density function.

    def pdf(x, sigma):
        return (1.0 / np.math.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-x ** 2 / (2 * sigma ** 2))

    # Choose the kernel size according to the rule of thumb 6 * sigma
    # to approximate 99.6% of the values form the gaussian.

    kernel_size = np.math.ceil(sigma * 3 * 2)

    # Calculate the 2D kernel by multiplying a 1D gaussian kernel with itself transposed.

    kernel = np.zeros((kernel_size, kernel_size))

    # Calculate 1D kernel and normalize it.

    kernel_x = np.zeros(kernel_size)
    negative_part = np.math.floor(kernel_size / 2)

    for i in range(kernel_size):
        kernel_x[i] = pdf(i - negative_part, sigma)

    kernel_x = kernel_x / kernel_x.sum()

    # Multiply the original 1D kernel with itself transposed.

    kernel_x = kernel_x.reshape((kernel_size, 1))
    kernel_y = np.transpose(kernel_x)
    kernel = np.matmul(kernel_x, kernel_y)

    # Filter the image with the obtained 2D kernel

    img_custom_filter = cv.filter2D(img, -1, kernel)

    display(img_custom_filter)

    # c) Filter the image using two 1D kernels.

    img_sepfilter = cv.sepFilter2D(img, -1, kernel_x, kernel_y)

    display(img_sepfilter)

    # Calculate the maximum pixel error for all pairs of filterd images.

    max_err_cv_2d = np.max(np.absolute(img_cvg.astype(np.int32) - img_custom_filter.astype(np.int32)))
    max_err_cv_1d = np.max(np.absolute(img_cvg.astype(np.int32) - img_sepfilter.astype(np.int32)))
    max_err_2d_1d = np.max(np.absolute(img_custom_filter.astype(np.int32) - img_sepfilter.astype(np.int32)))

    print('maximum err cv - 2d')
    print(max_err_cv_2d)
    print('maximum err cv - 1dx1d')
    print(max_err_cv_1d)
    print('maximum err 1dx1d - 2d')
    print(max_err_2d_1d)



#    =========================================================================
#    ==================== Task 5 =================================
#    =========================================================================

    print('Task 5:');

    # a)

    img_2gaussianfilters = cv.GaussianBlur(img, (0,0), 2)

    # b)

    img_2gaussianfilters = cv.GaussianBlur(img_2gaussianfilters, (0,0), 2)
    img_1gaussianfilter = cv.GaussianBlur(img, (0,0), 2 * np.math.sqrt(2))

    display(img_2gaussianfilters)
    display(img_1gaussianfilter)

    # Calculate the maximum pixel difference.

    max_err = np.max(np.absolute(img_2gaussianfilters.astype(np.int32) - img_1gaussianfilter.astype(np.int32)))

    print(max_err)

#    =========================================================================
#    ==================== Task 6 =================================
#    =========================================================================
    print('Task 6:');






#    =========================================================================
#    ==================== Task 7 =================================
#    =========================================================================

    print('Task 7:')

    img_noisy = img.copy()

    # Add salt and pepper noise randomly.

    for i in range(img.shape[0]):
        for k in range(img.shape[1]):
            random = np.random.rand()
            if(random < 0.3): img_noisy[i][k] = 0
            if(random < 0.15): img_noisy[i][k] = 255

    display(img_noisy)

    # Initialize winner sizes, images and scores for a)b) and c)

    best_score_gauss = 99999999
    winner_image_gauss = img_noisy.copy()
    winner_size_gauss = -1
    best_score_median = 99999999
    winner_image_median = img_noisy.copy()
    winner_size_median = -1
    best_score_bilateral = 99999999
    winner_image_bilateral = img_noisy.copy()
    winner_size_bilateral = -1

    # For each kernel size filter the images and compare it to the current best filter size.

    for i in np.array([1,3,5,7,9]):
        filtered_gauss = cv.GaussianBlur(img_noisy, (i,i), 0)
        filtered_median = cv.medianBlur(img_noisy, i)
        filtered_bilateral = cv.bilateralFilter(img_noisy, i, 100, 100)

        # Calculate the difference between the orignial image and the filtered images.

        difference_gauss = np.absolute(img.astype(np.int32) - filtered_gauss.astype(np.int32))
        difference_median = np.absolute(img.astype(np.int32) - filtered_median.astype(np.int32))
        difference_bilateral = np.absolute(img.astype(np.int32) - filtered_bilateral.astype(np.int32))

        # Calculate the mean error.

        mean_gauss = np.mean(difference_gauss)
        mean_median = np.mean(difference_median)
        mean_bilateral = np.mean(difference_bilateral)

        # Update the current best filter size.

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

    display(winner_image_gauss)
    display(winner_image_median)
    display(winner_image_bilateral)

#    =========================================================================
#    ==================== Task 8 =================================
#    =========================================================================

    print('Task 8:');

    # Initialize both kernels.

    k1 = np.array([[0.0113, 0.0838, 0.0113],
                        [0.0838, 0.6193, 0.0838],
                        [0.0113, 0.0838, 0.0113]])
    k2 = np.array([[-0.8984, 0.1472, 1.1410],
                        [-1.9075, 0.1566, 2.1359],
                        [-0.8659, 0.0573, 1.0337]])

    # a) Filter the image.

    img_2D_k1 = cv.filter2D(img, -1, k1)
    img_2D_k2 = cv.filter2D(img, -1, k2)

    display(img_2D_k1)
    display(img_2D_k2)

    # b) Decompose the filter kernels k1 and k2.

    ew_k1, u_k1, vT_k1 = cv.SVDecomp(k1)
    ew_k2, u_k2, vT_k2 = cv.SVDecomp(k2)

    # Extract 1D kernels from the decomposition and normalize them.

    X_1D_k1 = (1/vT_k1[0].sum()) * vT_k1[0]
    Y_1D_k1 = (1/u_k1[:,0].sum()) * u_k1[:,0]

    X_1D_k2 = (1/vT_k2[0].sum()) * vT_k2[0]
    Y_1D_k2 = (1/u_k2[:,0].sum()) * u_k2[:,0]

    # Filter the images with both 1D kernels.

    img_1D_k1 = cv.sepFilter2D(img, -1, X_1D_k1, Y_1D_k1)
    img_1D_k2 = cv.sepFilter2D(img, -1, X_1D_k2, Y_1D_k2)

    display(img_1D_k1)
    display(img_1D_k2)

    # c) Calculate the maximum pixel error.

    k1_max_error = np.max(np.absolute(img_2D_k1.astype(np.int32) - img_1D_k1.astype(np.int32)))
    k2_max_error = np.max(np.absolute(img_2D_k2.astype(np.int32) - img_1D_k2.astype(np.int32)))

    print(k1_max_error)
    print(k2_max_error)
























    #
