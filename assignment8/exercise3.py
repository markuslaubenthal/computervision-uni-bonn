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


    n_keypoints_1 = descriptors_mountain1.shape[0]
    n_keypoints_2 = descriptors_mountain2.shape[0]

    my_matches = []
    k1 = []
    k2 = []
    offset = 0
    bfmatcher = cv2.BFMatcher()

    # for m1 in range(n_keypoints_1):
        # keypoint1 = sift_mountain1[m1], descriptors_mountain1[m1]
        # matches = np.zeros((n_keypoints_2, 2))

        # matches = np.linalg.norm(descriptors_mountain2 - keypoint1[1], axis=1)
        # best_matches_ind = np.argpartition(matches, 2)[:2]
        # ratio = (matches[best_matches_ind[0]] / matches[best_matches_ind[1]])
    dmatch = bfmatcher.knnMatch(descriptors_mountain1, descriptors_mountain2, k=2)
    # dmatch = sorted(dmatch, key = lambda x:x[0].distance)
        # dm1 = dmatch[0][0]
        # dm2 = dmatch[0][1]
        # print(dmatch)
    for match in dmatch:
        dm1 = match[0]
        dm2 = match[1]

        if dm1.distance > dm2.distance:
            dm1,dm2 = dm2, dm1

        ratio = (dm1.distance / dm2.distance)

        if(ratio > 0.4):
            my_matches.append(dm1)


            # k1.append(sift_mountain1[m1])
            # k2.append(sift_mountain2[best_matches_ind[0]])
        # else:
        #     offset += 1

        # print(match_indices)


    my_matches = sorted(my_matches, key = lambda x:x.distance)
    # your own implementation of matching

    # k1 = []
    # k2 = []
    # for match in enumerate(match_indices):
    #     k1.append()

    # display the matches
    # matches_indices = np.arange(0, len(k1))
    # print(matches_indices)
    img_match = np.zeros((max(mountain1.shape[0], mountain2.shape[0]), mountain1.shape[1] + mountain2.shape[1], 3), dtype=np.uint8)
    img_match = cv2.drawMatches(mountain1, sift_mountain1, mountain2, sift_mountain2, my_matches[:10], None, flags=2)
    # img_match = np.empty((max(mountain1.shape[0], mountain2.shape[0]), mountain1.shape[1] + mountain2.shape[1], 3), dtype=np.uint8)
    # cv2.drawMatchesKnn(mountain1, k1, mountain2, k2, matches_indices, outImg=img_match, matchColor=None, singlePointColor=(255, 255, 255), flags=2)
    display(img_match)
    pass


if __name__ == '__main__':
    main()
