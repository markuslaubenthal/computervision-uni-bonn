# authors: Markus Laubenthal, Lennard Alms

import cv2
import numpy as np
import matplotlib.pylab as plt

def display(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Load the images
    mountain1 = cv2.imread('data/exercise3/mountain1.png', 0)
    mountain2 = cv2.imread('data/exercise3/mountain2.png', 0)
    # extract sift keypoints and descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    sift_mountain1, descriptors_mountain1 = sift.detectAndCompute(mountain1, None)
    sift_mountain2, descriptors_mountain2 = sift.detectAndCompute(mountain2, None)


    # Number of keypoints in image 1
    n_keypoints_1 = descriptors_mountain1.shape[0]

    # Initialize Match List
    my_matches = []

    for m1 in range(n_keypoints_1):
        # Calculate the distances for m1th keypoint of image 1 to all keypoints of image 2
        matches = np.linalg.norm(descriptors_mountain2 - descriptors_mountain1[m1], axis=1)

        # Get the 2 smallest distances
        best_matches_ind = np.argpartition(matches, 2)[:2]

        # Do the ratio test
        ratio = (matches[best_matches_ind[0]] / matches[best_matches_ind[1]])
        if(ratio > 0.4):
            dmatch = cv2.DMatch(m1, best_matches_ind[0], 0, matches[best_matches_ind[0]])
            my_matches.append([dmatch])

    # Sort matches such that the best matches come first
    my_matches = sorted(my_matches, key = lambda x:x[0].distance)
    # display the matches (first 10)
    # img_match = np.zeros((max(mountain1.shape[0], mountain2.shape[0]), mountain1.shape[1] + mountain2.shape[1], 3), dtype=np.uint8)
    img_match = cv2.drawMatchesKnn(mountain1, sift_mountain1, mountain2, sift_mountain2, my_matches[:10], None, flags=2)
    display(img_match)
    pass


if __name__ == '__main__':
    main()
