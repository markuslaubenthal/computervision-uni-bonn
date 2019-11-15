import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=False)  # if you do not have latex installed simply uncomment this line + line 75

def display(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_data():
    """ loads the data for this task
    :return:
    """
    fpath = 'images/ball.png'
    radius = 70
    Im = cv2.imread(fpath, 0).astype('float32')/255  # 0 .. 1

    # we resize the image to speed-up the level set method
    Im = cv2.resize(Im, dsize=(0, 0), fx=0.5, fy=0.5)

    height, width = Im.shape

    centre = (width // 2, height // 2)
    Y, X = np.ogrid[:height, :width]
    phi = radius - np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2)

    return Im, phi


def get_contour(phi):
    """ get all points on the contour
    :param phi:
    :return: [(x, y), (x, y), ....]  points on contour
    """
    eps = 1
    A = (phi > -eps) * 1 #1 wenn  oder 0
    B = (phi < eps) * 1 #1 oder 0
    D = (A - B).astype(np.int32)
    D = (D == 0) * 1
    Y, X = np.nonzero(D)
    return np.array([X, Y]).transpose()

# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
def edge_magnitude(Im):

    # Kernels for finding partial derivatives
    mx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    my = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # Partial derivaties of the image
    part_der_x = cv2.filter2D(Im, -1, mx)
    part_der_y = cv2.filter2D(Im, -1, my)

    # Magnitude of the derivative
    magnitude = part_der_x ** 2 + part_der_y ** 2

    return magnitude

def scaled_mean_curvature_motion(phi):

    phi_x_kernel = np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]])
    phi_y_kernel = np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]])
    phi_x = cv2.filter2D(phi, -1, phi_x_kernel)
    phi_y = cv2.filter2D(phi, -1, phi_y_kernel)
    

    phi_xx_kernel = np.array([[0, 0, 0],[1, -2, 1],[0, 0, 0]])
    phi_yy_kernel = np.array([[0,1,0], [0,-2,0], [0,1,0]])

    phi_xx = cv2.filter2D(phi, -1, phi_xx_kernel)
    phi_yy = cv2.filter2D(phi, -1, phi_yy_kernel)

    phi_xy_kernel = np.array([[0.25, 0, -0.25], [0,0,0], [-0.25,0,0.25]])
    phi_xy = cv2.filter2D(phi, -1, phi_xy_kernel)

    eps = 10 ** (-4)
    mean_curv_mot = (phi_xx * phi_y ** 2 -
                     2 * phi_x * phi_y * phi_xy +
                     phi_yy * phi_x ** 2) / (phi_x ** 2 + phi_y ** 2 + eps)

    return mean_curv_mot


def propagation_towards_edges(phi, w):
    # w' * phi'

    w_x_kernel = np.array([[0, 0, 0],[-0.5, 0, 0.5],[0, 0, 0]])
    w_y_kernel = np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]])
    w_x = cv2.filter2D(w, -1, w_x_kernel)
    w_y = cv2.filter2D(w, -1, w_y_kernel)

    # One sided difference
    osdf_kernel_x = np.array([[0, 0, 0],[0,-1,1],[0, 0, 0]])
    osdf_kernel_y = np.array([[0,0,0],[0,-1,0],[0,1,0]])
    osdb_kernel_x = np.array([[0, 0, 0],[-1,1,0],[0, 0, 0]])
    osdb_kernel_y = np.array([[0,-1,0],[0,1,0],[0,0,0]])

    phi_osdf_x = cv2.filter2D(phi, -1, osdf_kernel_x)
    phi_osdf_y = cv2.filter2D(phi, -1, osdf_kernel_y)
    phi_osdb_x = cv2.filter2D(phi, -1, osdb_kernel_x)
    phi_osdb_y = cv2.filter2D(phi, -1, osdb_kernel_y)


    uphill_dir = (np.max(np.max(w_x), 0) * (phi_osdf_x) +
                  np.min(np.min(w_x), 0) * (phi_osdb_x) +
                  np.max(np.max(w_y), 0) * (phi_osdf_y) +
                  np.min(np.min(w_y), 0) * (phi_osdb_y))

    return uphill_dir


def geodesic_active_contour(edge_magn, phi):

    # Geodesic metric
    w = 1 / (edge_magn + 1)

    tau1 = 1 / (4 * np.max(w)) # 0.25 since max(w) = 1
    #tau2 = 0.5

    height, width = phi.shape
    phi_next = phi.copy()

    mean_curv_mot = scaled_mean_curvature_motion(phi)
    prop_towards_edges = propagation_towards_edges(phi, w)

    phi_next = phi + tau1 * w * mean_curv_mot + 0.5 * prop_towards_edges

    return phi_next
# ------------------------


if __name__ == '__main__':

    n_steps = 20000
    plot_every_n_step = 5

    Im, phi = load_data()

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    edges = edge_magnitude(Im)


    ax1.imshow(Im, cmap='gray')
    ax1.set_title('frame 0')

    contour = get_contour(phi)
    if len(contour) > 0:
        ax1.scatter(contour[:, 0], contour[:, 1], color='red', s=1)

    ax2.clear()
    ax2.imshow(phi)
    ax2.set_title(r'$\phi$', fontsize=22)
    plt.pause(0.01)


    for t in range(n_steps):

        phi = geodesic_active_contour(edges, phi)

        if t % plot_every_n_step == 0:
            ax1.clear()
            ax1.imshow(Im, cmap='gray')
            ax1.set_title('frame ' + str(t))

            contour = get_contour(phi)
            if len(contour) > 0:
                ax1.scatter(contour[:, 0], contour[:, 1], color='red', s=1)

            ax2.clear()
            ax2.imshow(phi)
            ax2.set_title(r'$\phi$', fontsize=22)
            plt.pause(0.01)

    plt.show()
