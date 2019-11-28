#!/usr/bin/python3.5

import numpy as np
import cv2 as cv

'''
    load the image and foreground/background parts
    image: the original image
    background/foreground: numpy array of size (n_pixels, 3) (3 for RGB values), i.e. the data you need to train the GMM
'''

def display(img):
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


image_max = None
def read_image(filename):
    global image_max
    image = cv.imread(filename)
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    image_max = image.max()
    image = image / image.max()
    height, width = image.shape[:2]
    bounding_box = np.zeros(image.shape)
    bounding_box[90:350, 110:250, :] = 1
    bb_width, bb_height = 140, 260
    background = image[bounding_box == 0].reshape((height * width - bb_width * bb_height, 3))
    foreground = image[bounding_box == 1].reshape((bb_width * bb_height, 3))

    return image, foreground, background

def pdf(x,mue,sigma):
    numerator = np.exp(-0.5 *
                    np.sum(
                        (x-mue.reshape(1,mue.shape[0])) ** 2 *
                        (1/sigma.reshape(1,sigma.shape[0]))
                    , axis=1)
                )
    denominator = np.sqrt(((2*np.pi) ** 3) * np.prod(sigma))
    return numerator/denominator

class GMM(object):
    def __init__(self, image, foreground, background):
        self.image = image
        self.foreground = foreground
        self.background = background
        self.data = None
        self.mue = None
        self.sigma = None
        self.lamb = np.ones(1)

    def gaussian_scores(self, data):
        # TODO
        pass


    def fit_single_gaussian(self, data):
        mue = np.mean(data, axis=0)
        var = np.var(data, axis=0)
        return mue, var

        pass


    def estep(self):
        r = np.zeros((self.data.shape[0], self.mue.shape[0]))
        denominator = 0
        for k in range(self.mue.shape[0]):
            denominator += self.lamb[k] * pdf(self.data, self.mue[k], self.sigma[k])
        for k in range(self.mue.shape[0]):
            numerator = self.lamb[k] * pdf(self.data, self.mue[k], self.sigma[k])
            r[:,k] = numerator / denominator
        return r
        pass


    def mstep(self, r):
        lamb_deno = r.sum()
        for k in range(self.mue.shape[0]):
            r_sum = r[:,k].sum()
            self.lamb[k] = r_sum / lamb_deno
            self.mue[k] = np.sum(r[:,k].reshape(r.shape[0], 1) * self.data, axis=0) / r_sum
            self.sigma[k] = np.sum(
                                r[:,k] *
                                np.sum(
                                    np.power(
                                        self.data - self.mue[k].reshape(1,self.mue.shape[1]),
                                        2
                                    )
                                ,axis=1)) \
                            / r_sum
        pass

    def train(self, data, n_iterations = 20, k_splits = 3):
        self.data = data
        for i in range(n_iterations):
            if i < k_splits:
                self.split()
            self.mstep(self.estep())
        print(self.mue)
        print(self.sigma)
        print(self.lamb)

        pass


    def split(self, epsilon = 15):
        mue = np.array([])
        sigma = np.array([])
        lamb = np.array([])
        for i in range(self.mue.shape[0]):
            mue_i = self.mue[i]
            sigma_i = self.sigma[i]
            lamb_i = self.lamb[i]
            mue_1, mue_2 = mue_i + sigma_i * epsilon, mue_i - sigma_i * epsilon
            sigma_1, sigma_2 = sigma_i, sigma_i
            lamb_2, lamb_1 = lamb_i / 2, lamb_i / 2
            mue = np.append(mue, mue_1)
            mue = np.append(mue, mue_2)
            sigma = np.append(sigma, sigma_1)
            sigma = np.append(sigma, sigma_2)
            lamb = np.append(lamb, lamb_1)
            lamb = np.append(lamb, lamb_2)
        mue = mue.reshape(mue.shape[0] // 3, 3)
        sigma = sigma.reshape(sigma.shape[0] // 3, 3)
        self.mue = mue
        self.sigma = sigma
        self.lamb = lamb
        pass


    def probability(self, data):
        # TODO
        pass


    def sample(self):
        # TODO
        pass


    def em_algorithm(self, tau=0.4):
        global image_max
        mue, sigma = self.fit_single_gaussian(self.background)
        self.mue = np.array([mue])
        self.sigma = np.array([sigma])
        self.train(self.background)
        prob = np.zeros(self.image.shape[0] * self.image.shape[1])
        for k in range(self.mue.shape[0]):
            prob += self.lamb[k] * pdf(self.image.reshape(self.image.shape[0] * self.image.shape[1], 3), self.mue[k], self.sigma[k])
        prob = prob.reshape(self.image.shape[0],self.image.shape[1])
        prob = prob / prob.max()
        indices = np.where(prob > tau)
        self.image *= image_max
        self.image = cv.cvtColor(self.image, cv.COLOR_HSV2BGR)
        self.image[indices] = [0,0,0]



        display(self.image.astype(np.uint8))
        # with np.nditer(self.image, op_flags=['readwrite']) as it
        # for px in it:
        #     for k in range(self.mue.shape[0]):


        pass


image, foreground, background = read_image('person.jpg')

'''
TODO: compute p(x|w=background) for each image pixel and manipulate the image such that everything below the threshold is black, display the resulting image
Hint: Slide 64
'''
gmm_background = GMM(image, foreground, background)
gmm_background.em_algorithm()
