import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import cv2


def plot_snake(ax, V, fill='green', line='red', alpha=1, with_txt=False):
    """ plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    V_plt = np.append(V.reshape(-1), V[0,:]).reshape((-1, 2))
    ax.plot(V_plt[:,0], V_plt[:,1], color=line, alpha=alpha)
    ax.scatter(V[:,0], V[:,1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))


def load_data(fpath, radius):
    """
    :param fpath:
    :param radius:
    :return:
    """
    Im = cv2.imread(fpath, 0)
    h, w = Im.shape
    n = 20  # number of points
    u = lambda i: radius * np.cos(i) + w / 2
    v = lambda i: radius * np.sin(i) + h / 2
    V = np.array(
        [(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][0:-1],
        'int32')

    return Im, V


# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
# your implementation here

# ------------------------


def run(fpath, radius):
    """ run experiment
    :param fpath:
    :param radius:
    :return:
    """
    img, V = load_data(fpath, radius)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 200

    # ------------------------
    # take derivative of the image
    img_dx = cv2.Sobel(img, -1, 1, 0, 7)
    img_dy = cv2.Sobel(img, -1, 0, 1, 7)

    #define helper functions for energy calculation
    dx_i = lambda px: img_dx[px[0],px[1]]
    dy_i = lambda px: img_dy[px[0],px[1]]

    ext_e = lambda px : -(dx_i(px) ** 2 + dy_i(px) ** 2)

    #d2_v = lambda i: V[i+1 % len(V)] - 2 * V[i] + V[i-1 % len(V)]

    def get_elast(V, alpha):
        dist_avg = 0
        for i in range(len(V)):
            dist_avg += np.linalg.norm(V[i]-V[(i+1) % len(V)])
        dist_avg /= len(V)
        print(dist_avg)
        return lambda u,v: alpha * (np.linalg.norm(u-v) - dist_avg) ** 2


    #define alpha
    alpha = 0.001

    # ------------------------

    for t in range(n_steps):
        # ------------------------

        #get function for internal energy (pairwise term)
        int_e = get_elast(V, alpha)

        #calculate unary terms
        unary = []
        for v in V:
            u = []
            for x_offset in range(-1,2):
                for y_offset in range(-1,2):
                    x = v[0] + x_offset
                    y = v[1] + y_offset
                    u.append( {'ext': ext_e([x,y]),
                               'pos': np.array([x,y]),
                               'parent': None,
                               'acc': 0} )

            unary.append(u)

        #dyn prog
        for i in range(1, len(V)):
            for u in unary[i]:
                smallest_energy = float('inf')
                parent = None
                for v in unary[i-1]:
                    int = int_e(u['pos'], v['pos'])
                    energy = v['acc'] + v['ext'] + int
                    if(energy < smallest_energy):
                        smallest_energy = energy
                        parent = v
                u['acc'] = smallest_energy
                u['parent'] = parent

        smallest_node = None
        smallest_energy = float('inf')
        for v in unary[-1]:
            energy = v['acc'] +  v['ext']
            if(energy < smallest_energy):
                smallest_energy = energy
                smallest_node = v

        V_new = []
        V_new.insert(0, smallest_node['pos'])

        while smallest_node['parent'] is not None:
            smallest_node = smallest_node['parent']
            V_new.insert(0, smallest_node['pos'])
            print(smallest_node['ext'])
        print('______')

        V = np.array(V_new)

        # ------------------------

        ax.clear()
        ax.imshow(img, cmap='gray')
        ax.set_title('frame ' + str(t))
        plot_snake(ax, V)
        plt.pause(0.01)


    plt.pause(2)


if __name__ == '__main__':
    run('images/ball.png', radius=120)
    #run('images/coffee.png', radius=98)
