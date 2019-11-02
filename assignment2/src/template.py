# authors: Markus Laubenthal, Lennard Alms

import cv2
import numpy as np
import time


## Entfernen
import matplotlib.pyplot as plt

def display(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_convolution_using_fourier_transform(image, kernel):

    # Use the fast fourier transformation on the image
    ft_img = np.fft.fft2(image)

    # Use the fast fourier transformation on the kernel
    # and expand the size to match the size of the image
    ft_kernel = np.fft.fft2(kernel, (image.shape[0], image.shape[1]))

    # Multiply the image and the kernel in the frequency domain
    # and use the inverse of the fourier transformation to get back to the image domain
    result = np.fft.ifft2(ft_img * ft_kernel)

    return np.abs(result).astype(np.uint8)


def task1():

    # Read the image
    image = cv2.imread("../data/einstein.jpeg", 0)

    # Calculate the 7x7 gaussian kernel by multiplying a 1D kernel with itself transposed
    # Sigma = 1
    kernel = cv2.getGaussianKernel(7,1)
    kernel = kernel.dot(kernel.reshape(1, 7))

    # Filter the image with the 7x7 gaussian kernel
    conv_result = cv2.filter2D(image, -1, kernel)

    # Use fourier transformation to apply the kernel
    fft_result = get_convolution_using_fourier_transform(image, kernel)

    # Calculate and print the mean absolute difference between the two results
    mean_abs_diff = np.mean(np.abs(np.subtract(conv_result.astype(np.int32), fft_result.astype(np.int32))))
    print(mean_abs_diff)

def ncc(normalized_template, normalized_patch):

    # Normalized cross correlation implementation for given template and patch like on the slides
    top = np.sum(normalized_template * normalized_patch)
    bot = np.sum(np.power(normalized_template, 2))
    bot *= np.sum(np.power(normalized_patch, 2))
    bot = np.sqrt(bot)

    # Error handling
    if(bot == 0.0): return 0

    return top / bot


def normalized_cross_correlation(image, template, interesting_px = None):
    # Initialize the otput array
    h = np.zeros(image.shape)

    # Calculate a padding size and add the padding to the image
    # This is needed to be able to calculate the correlation in the corners of the image
    padd_shape = np.floor(np.max(template.shape) / 2).astype(np.int32)
    image_padding = np.pad(image, padd_shape)

    # Calculate the error of the padding since the template might not be a square
    padd_shape1 = np.floor(np.min(template.shape) / 2).astype(np.int32)
    padd_diff = padd_shape - padd_shape1


    p_height = template.shape[0]
    p_width = template.shape[1]

    # Substract the template-average from the template
    normalized_template = template - np.mean(template)

    # This if branching is for task 3
    if(interesting_px is None):

        # Task 2 always goes here
        # Iterate over the pixels
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):

                # Select the patch from the original image with padding.
                # The patch is centered around (m,n)
                patch = image_padding[row + padd_diff: row + p_height + padd_diff, col : col + p_width]

                # Substract the patch-average from the patch
                normalized_patch = patch - np.mean(patch)

                # Calculate the normalized cross correlation on a specific point in the image
                h[row][col] = ncc(normalized_patch, normalized_template)
    else:

        # Task 3 goes here after first iteration
        # Iterate over all interesting pixel-coodinates
        for row,col in interesting_px:

            # Select the patch from the original image with padding.
            # The patch is centered around (m,n)
            patch = image_padding[row + padd_diff: row + p_height + padd_diff,
                                  col : col + p_width]

            # Substract the patch-average from the patch
            normalized_patch = patch - np.mean(patch)

            # Error handling
            try:

                # Calculate the normalized cross correlation on a specific point in the image
                h[row][col] = ncc(normalized_patch, normalized_template)

            except:
                err = 1
    return h


def draw_rectangle(template,image,result_ncc):

    # Calculate the rectangle offset from topleft to center
    offsetx = int(template.shape[1] / 2)
    offsety = int(template.shape[0] / 2)

    image_cpy = image.copy()

    # Iterate over all pixels
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):

            # Draw rectangle arround pixel if the threshold is met.
            if(result_ncc[y][x] > 0.7):
                image_cpy = cv2.rectangle(
                    image_cpy,
                    (x - offsetx, y - offsety),
                    (x + offsetx, y + offsety),
                    (255, 255, 255, 0.4),
                1)

    return image_cpy

def task2():

    # Load the image and the template
    image = cv2.imread("../data/lena.png", 0)
    template = cv2.imread("../data/eye.png", 0)

    # Template Matching with cross correlation and threshold 0.7

    # Calculate the normalized cross correlation
    result_ncc = normalized_cross_correlation(image, template)

    # Draw rectangles arround the found results
    img_cpy = draw_rectangle(template, image, result_ncc)

    # Show the results
    display(img_cpy)


def build_gaussian_pyramid_opencv(image, num_levels):
    pyramid = []
    pyramid.append(image)
    for x in range(num_levels):
        pyramid.append(cv2.pyrDown(pyramid[x]))
    return pyramid

def build_gaussian_pyramid(image, num_levels, sigma):
    pyramid = []
    pyramid.append(image)
    kernel = cv2.getGaussianKernel(3,sigma)
    kernel = kernel.dot(kernel.reshape(1, 3))
    sample_img = image.copy()
    for x in range(num_levels):
        sample_img = cv2.filter2D(sample_img, -1, kernel)
        sample_img = sample_img[::2,::2]
        pyramid.append(sample_img.copy())
    return pyramid


def template_matching_multiple_scales(pyramid, template):
    return None


def task3():

    # Load the image and the template
    image = cv2.imread("../data/traffic.jpg", 0)
    template = cv2.imread("../data/traffic-template.png", 0)

    # Use own function to calculate the gaussian pyramid 4 levels deep
    cv_pyramid_image = build_gaussian_pyramid_opencv(image, 4)

    # Use cv function to calculate the gaussian pyramid 4 levels deep
    my_pyramid_image = build_gaussian_pyramid(image, 4, 1)

    # Calculate the mean absolute difference between both pyramids
    for x in range(5):
        mean_abs_diff = np.mean(
                            np.abs(
                                np.subtract(
                                    cv_pyramid_image[x].astype(np.int32),
                                    my_pyramid_image[x].astype(np.int32))))
        print('Mean abs difference: level ', x, ' ', mean_abs_diff)

    # Use own normalized cross correlation implementation to find the template in the image
    start = time.time()
    normalized_cross_correlation(image, template)
    end = time.time()
    print('time for ncc without pyramid: ' , (end - start))

    # Use gaussian pyramid for normalized cross correlation to find the template
    start = time.time()

    # Calculate the gaussian pyramid for the image
    my_pyramid_image = build_gaussian_pyramid(image, 4, 1)

    # Calculate the gaussian pyramid for the template
    my_pyramid_template = build_gaussian_pyramid(template, 4, 1)

    # Initialize
    interesting_px = None
    ncc_result = None

    # Iterate over the levels from lowest to highest 4->0
    for level in range(5)[::-1]:
        if(interesting_px is None):
            # First iteration goes always here
            # Calculate the ncc for the pyramid and the template at the lowest level
            ncc_result = normalized_cross_correlation(my_pyramid_image[level],
                                     my_pyramid_template[level])
        else:
            # All other iteration go here
            # Calculate the ncc for the pyramid and the template at the given level
            # and make use of the intesting pixels calculated in the lower level
            ncc_result = normalized_cross_correlation(my_pyramid_image[level],
                                     my_pyramid_template[level],
                                     interesting_px)

        # Calculate the intesting pixels in this level
        candidates = np.where(ncc_result > 0.7)

        # Derive the intesting pixels in the next level by upscaling the coordinates
        interesting_px = []
        for y,x in np.transpose(candidates):
            interesting_px.append((y * 2, x * 2))
            interesting_px.append((y * 2 + 1, x * 2))
            interesting_px.append((y * 2, x * 2 + 1))
            interesting_px.append((y * 2 + 1, x * 2 + 1))
    end = time.time()
    print('time for ncc with pyramid: ', (end - start))

def pdf(x, sigma):
    # Gaussian function
    return (1.0 / np.math.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-x ** 2 / (2 * sigma ** 2))

def pdf_dx(x, sigma):
    # Derivative of gaussian function
    return ((-np.exp(-(x**2))*x )/np.math.sqrt(2*np.pi))

def get_derivative_of_gaussian_kernel(size, sigma):

    # Initialize
    kernel_2d_dy = np.zeros((size, size))
    kernel_2d_dx = np.zeros((size, size))

    kernel_1d_x = np.zeros(size)
    kernel_1d_dx = np.zeros(size)
    negative_part = np.math.floor(size / 2)

    # Calculate 1D kernels.

    for i in range(size):
        kernel_1d_x[i] = pdf(i - negative_part, sigma)
        kernel_1d_dx[i] = pdf_dx(i - negative_part, sigma)

    # Multiply the the 1D gaussian kernel with the 1D derivative of gaussian kernel.
    # Do this in both directions
    kernel_1d_x = kernel_1d_x.reshape((size, 1))
    kernel_1d_dx = kernel_1d_dx.reshape((size, 1))
    kernel_2d_dx = np.matmul(kernel_1d_x, np.transpose(kernel_1d_dx))
    kernel_2d_dy = np.matmul(kernel_1d_dx, np.transpose(kernel_1d_x))

    return kernel_2d_dx, kernel_2d_dy


def task4():

    # Read the image
    image = cv2.imread("../data/einstein.jpeg", 0)

    # Calculate the convolution filters in both directions with sigma 0.6
    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)

    # Convolve with kernel_x
    edges_x = cv2.filter2D(image.astype(np.float32), -1, kernel_x)

    # Convolve with kernel_y
    edges_y = cv2.filter2D(image.astype(np.float32), -1, kernel_y)

    # Normalize the output
    edges_x /= np.max(edges_x)
    edges_y /= np.max(edges_y)

    # Display the Edges.
    display(edges_x)
    display(edges_y)


def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
    return None


def task5():

    # Load the image
    image = cv2.imread("../data/traffic.jpg", 0)

    # Use the canny edge detector from opencv
    edges = cv2.Canny(image, 100, 200)

    display(edges)

    # We did not implement this on our own.

    #dist_transfom_mine = l2_distance_transform_2D(edge_function, positive_inf, negative_inf)

    # compute using opencv
    dist_transfom_cv = cv2.distanceTransform(image, cv2.DIST_L2, 5)

    display(dist_transfom_cv)
    # compare and print mean absolute difference


if __name__ == "__main__":
    print('task1')
    task1()
    print('task2')
    task2()
    print('task3')
    task3()
    print('task4')
    task4()
    print('task5')
    task5()
