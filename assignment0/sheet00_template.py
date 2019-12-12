import cv2 as cv
import numpy as np
import random
import sys
from numpy.random import randint


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':

    # set image path
    img_path = 'bonn.png'

    # 2a: read and display the image

    img = cv.imread(img_path, cv.IMREAD_COLOR)
    display_image('2 - a - Original Image', img)

    # 2b: display the intensity image
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display_image('2 - b - Intensity Image', img_gray)


    # 2c: for loop to perform the operation
    img_cpy = np.copy(img)
    for column_index, column  in enumerate(img):
        for row_index, pixel in enumerate(column):
            half_intensity = np.rint(0.5 * img_gray[column_index][row_index])
            img_cpy[column_index][row_index] = np.array([
                max(0,pixel[0] - half_intensity),
                max(0,pixel[1] - half_intensity),
                max(0,pixel[2] - half_intensity),
            ])
    display_image('2 - c - Reduced Intensity Image', img_cpy)


    # 2d: one-line statement to perfom the operation above
    img_cpy = np.maximum(
        np.zeros((img_gray.shape[0], img_gray.shape[1], 3)),
        (img - np.expand_dims(img_gray * 0.5, axis=2))
    ).astype(np.uint8)
    display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy)

    # 2e: Extract the center patch and place randomly in the image
    patch_size = 16
    patch_x_start = int((img_gray.shape[0] - patch_size) / 2)
    patch_y_start = int((img_gray.shape[1] - patch_size) / 2)
    img_patch = img[patch_x_start:patch_x_start + patch_size, patch_y_start:patch_y_start + patch_size]
    display_image('2 - e - Center Patch', img_patch)

    # Random location of the patch for placement
    rand_coord = np.zeros(2).astype(np.uint8)
    rand_coord[0] = randint(img_gray.shape[0] - patch_size)
    rand_coord[1] = randint(img_gray.shape[1] - patch_size)
    img_cpy[rand_coord[0]:rand_coord[0] + patch_size, rand_coord[1]:rand_coord[1] + patch_size] = img_patch
    display_image('2 - e - Center Patch Placed Random %d, %d' % (rand_coord[0], rand_coord[1]), img_cpy)

    # 2f: Draw random rectangles and ellipses
    img_cpy = img.copy()
    height = img.shape[0]
    width = img.shape[1]
    for i in range(10):
        cv.ellipse(
            img_cpy,
            (randint(width),randint(height)),
            (randint(5, 50), randint(5, 50)),
            randint(360),
            0.0,
            360,
            (randint(255 + 1),randint(255 + 1),randint(255 + 1)),
            -1 # Negative Thickness -> Shape will be filled
        )
        p1 = (randint(width), randint(height))
        p2 = (randint(p1[0], width + 1), randint(p1[1], height + 1))
        cv.rectangle(img_cpy, p1, p2, (randint(255 + 1),randint(255 + 1),randint(255 + 1)), -1)
    display_image('2 - f - Rectangles and Ellipses', img_cpy)

    # destroy all windows
    cv.destroyAllWindows()
