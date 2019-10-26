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
    ft_img = np.fft.fft2(image)
    # kernel = cv2.getGaussianKernel2D((440,680),1)  # calculate kernel
    # ft_kernel = np.fft.fft2(kernel)

    ft_kernel = np.fft.fft2(kernel, (image.shape[0], image.shape[1]))
    return np.fft.ifft2(ft_img * ft_kernel)


def task1():
    image = cv2.imread("../data/einstein.jpeg", 0)
    kernel = cv2.getGaussianKernel(7,1)  # calculate kernel
    kernel = kernel.dot(kernel.reshape(1, 7))
    conv_result = cv2.filter2D(image, -1, kernel)

    fft_result = get_convolution_using_fourier_transform(image, kernel)
    fft_result = np.abs(fft_result).astype(np.uint8)


    # compare results
    mean_abs_diff = np.mean(np.abs(np.subtract(conv_result.astype(np.int32), fft_result.astype(np.int32))))
    print(mean_abs_diff)


def sum_square_difference(image, template):

    return None


def ncc(normalized_template, normalized_patch):
    top = np.sum(normalized_template * normalized_patch)
    bot = np.sum(np.power(normalized_template, 2))
    bot *= np.sum(np.power(normalized_patch, 2))
    bot = np.sqrt(bot)
    if(bot == 0.0): return 0
    return top / bot


def normalized_cross_correlation(image, template):
    h = image.copy().astype(np.float64)
    padd_shape = np.floor(np.max(template.shape) / 2).astype(np.int32)
    padd_shape1 = np.floor(np.min(template.shape) / 2).astype(np.int32)
    padd_diff = padd_shape- padd_shape1
    image_padding = np.pad(image, padd_shape)
    display(image_padding)
    p_height = template.shape[0]
    p_width = template.shape[1]

    normalized_template = template - np.mean(template)
    for m in range(image.shape[0]):
        for n in range(image.shape[1]):
            patch = image_padding[m + padd_diff: m + p_height + padd_diff, n : n + p_width]
            normalized_patch = patch - np.mean(patch)
            h[m][n] = ncc(normalized_patch, normalized_template)
    return h

def threshold_image(img, t):
    img_cpy = img - t
    img_cpy = img_cpy.clip(0,1)
    img_cpy[img_cpy > 0] = 1
    return img_cpy


def task2():
    image = cv2.imread("../data/lena.png", 0)
    template = cv2.imread("../data/eye.png", 0)

    # result_ssd = sum_square_difference(image, template)
    result_ncc = normalized_cross_correlation(image, template)
    threshold_img = threshold_image(result_ncc, 0.7)

    display(image)
    display(threshold_img)

    result_cv_sqdiff = None  # calculate using opencv
    result_cv_ncc = None  # calculate using opencv

    # draw rectangle around found location in all four results

    offsetx = int(template.shape[1] / 2)
    offsety = int(template.shape[0] / 2)
    image_cpy = image.copy()
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if(result_ncc[y][x] > 0.7):
                image_cpy = cv2.rectangle(
                    image_cpy,
                    (x - offsetx, y - offsety),
                    (x + offsetx, y + offsety),
                    (255, 255, 255, 0.4),
                1)

    display(image_cpy)
    # show the results


def build_gaussian_pyramid_opencv(image, num_levels):
    return None


def build_gaussian_pyramid(image, num_levels, sigma):
    return None


def template_matching_multiple_scales(pyramid, template):
    return None


def task3():
    image = cv2.imread("../data/traffic.jpg", 0)
    template = cv2.imread("../data/template.jpg", 0)

    cv_pyramid = build_gaussian_pyramid_opencv(image, 8)
    mine_pyramid = build_gaussian_pyramid(image, 8)

    # compare and print mean absolute difference at each level
    result = template_matching_multiple_scales(pyramid, template)

    # show result


def get_derivative_of_gaussian_kernel(size, sigma):
    return None, None


def task4():
    image = cv2.imread("../data/einstein.jpeg", 0)

    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)

    edges_x = None  # convolve with kernel_x
    edges_y = None  # convolve with kernel_y

    magnitude = None  # compute edge magnitude
    direction = None  # compute edge direction

    cv2.imshow("Magnitude", magnitude)
    cv2.imshow("Direction", direction)


def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
    return None


def task5():
    image = cv2.imread("../data/traffic.jpg", 0)

    edges = None  # compute edges
    edge_function = None  # prepare edges for distance transform

    dist_transfom_mine = l2_distance_transform_2D(
        edge_function, positive_inf, negative_inf
    )
    dist_transfom_cv = None  # compute using opencv

    # compare and print mean absolute difference


if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()
    task5()
