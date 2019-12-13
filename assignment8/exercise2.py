import cv2
import numpy as np
import matplotlib.pylab as plt

def display(img,m,points = None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.imshow(img, cmap='gray')
    ax.imshow(img, cmap=m)
    if points is not None:
        ax.scatter(points[1],points[0],s=1,c='red')
    # ax.plot(points[:,0],points[:,1],c='green')
    plt.show()

def non_maxima_supression(response):
    res = response.copy()
    indices = np.nonzero(res)
    for i in range(indices[0].shape[0]):
        y,x = indices[0][i], indices[1][i]
        for y_ in range(-2,2):
            for x_ in range(-2,2):
                if y+y_ > 0 and y+y_ < response.shape[0] and x+x_ > 0 and x+x_ < response.shape[1]:
                    if res[y,x] < res[y+y_,x+x_]:
                        res[y,x] = 0

    res[np.nonzero(res)] = 1
    return res

def main():
    # Load the image
    img = cv2.imread('./data/exercise2/building.jpeg', cv2.IMREAD_GRAYSCALE)
    # Compute Structural Tensor
    w_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]]) / 8
    w = cv2.filter2D(img, -1, w_kernel)
    dxx = cv2.Sobel(w, -1, 2, 0, ksize=3)
    dyy = cv2.Sobel(w, -1, 0, 2, ksize=3)
    dxy = cv2.Sobel(w, -1, 1, 1, ksize=3)

    response = np.array( [  (np.absolute(np.linalg.eig([[dxx.flatten()[i], dxy.flatten()[i]],[dxy.flatten()[i], dyy.flatten()[i]]])[0])).prod() for i in range(dxx.flatten().shape[0]) ]).reshape((dxx.shape[0], dxx.shape[1]))

    heatmap = response / response.max() * 255
    heatmap = cv2.equalizeHist(heatmap.astype(np.uint8))
    display(heatmap, 'jet')

    response_p = np.percentile(response[np.nonzero(response)], 98)
    response[np.where(response < response_p)] = 0
    res = non_maxima_supression(response)

    display(img, 'gray', np.nonzero(res))

    w = (dxx * dyy - dxy ** 2) / (dxx + dyy +0.000001)
    q = (4 * (dxx * dyy - dxy ** 2)) / ((dxx + dyy) **2 +0.000001)

    heatmap = w / w.max() * 255
    heatmap = cv2.equalizeHist(heatmap.astype(np.uint8))
    display(heatmap, 'jet')

    heatmap = q / q.max() * 255
    heatmap = cv2.equalizeHist(heatmap.astype(np.uint8))
    display(heatmap, 'jet')

    w_p = np.percentile(w[np.nonzero(w)], 96)
    w[np.where(w < w_p)] = 0
    w[np.where(w >= w_p)] = 1
    q_p = np.percentile(q[np.nonzero(q)], 96)
    q[np.where(q < q_p)] = 0
    q[np.where(q >= q_p)] = 1

    display(img, 'gray', np.nonzero(w))
    display(img, 'gray', np.nonzero(q))
    display(img, 'gray', np.where(q != w))




    # Forstner Corner Detection

    pass


if __name__ == '__main__':
    main()






























#
