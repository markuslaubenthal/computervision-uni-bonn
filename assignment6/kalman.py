import numpy as np
import matplotlib.pylab as plt

observations = np.load('observations.npy')


def get_observation(t):
    return observations[t]


class KalmanFilter(object):
    def __init__(self, psi, sigma_p, phi, sigma_m, tau):
        self.psi = psi
        self.sigma_p = sigma_p
        self.phi = phi
        self.sigma_m = sigma_m
        self.state = None
        self.convariance = None
        self.tau = tau
        #self.sigma_t = sigma_p

    def init(self, init_state):
        self.state = init_state
        self.covariance = np.identity(4)
        pass

    def track(self, xt):
        #State prediction
        mue_prediction = np.dot(self.psi, self.state)
        print("mue_prediction", mue_prediction)
        #Covariance prediction
        sigma_p_prediction = self.sigma_p + np.dot(np.dot(self.psi, self.covariance), np.transpose(self.psi))
        print("sigma_p_prediction", sigma_p_prediction)
        #Kalman Gain
        kalman_gain1 = np.dot(sigma_p_prediction, np.transpose(self.phi))
        kalman_gain2 = np.dot(np.dot(self.phi, sigma_p_prediction), np.transpose(self.phi))
        kalman_gain3 = np.linalg.inv(self.sigma_m + kalman_gain2)
        kalman_gain = np.dot(kalman_gain1, kalman_gain3)
        print("kalman_gain", kalman_gain)
        #State update
        inner = xt - np.dot(self.phi, mue_prediction)
        mu_t = mue_prediction + np.dot(kalman_gain, inner)
        print("mu_t", mu_t)
        #Covariance update
        sigma_t = np.dot(np.identity(4) - np.dot(kalman_gain, self.phi), sigma_p_prediction)
        print('sigma_t', sigma_t)

        self.covariance = sigma_t
        self.state = mu_t
        pass

    def get_current_location(self):
        return self.state[0:2]
        pass

def perform_tracking(tracker):
    track = []
    for t in range(len(observations)):
        tracker.track(get_observation(t))
        track.append(tracker.get_current_location())

    return track

def main():
    init_state = np.array([0, 1, 0, 0])

    psi = np.array([[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    sp = 0.01
    sigma_p = np.array([[sp, 0, 0, 0],
                        [0, sp, 0, 0],
                        [0, 0, sp * 4, 0],
                        [0, 0, 0, sp * 4]])

    phi = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])
    sm = 0.05
    sigma_m = np.array([[sm, 0],
                        [0, sm]])


    tracker = KalmanFilter(psi, sigma_p, phi, sigma_m, tau=0)
    tracker.init(init_state)

    fixed_lag_smoother = KalmanFilter(psi, sigma_p, phi, sigma_m, tau=5)
    fixed_lag_smoother.init(init_state)

    track = perform_tracking(tracker)
    track_smoothed = perform_tracking(fixed_lag_smoother)

    plt.figure()
    plt.plot([x[0] for x in observations], [x[1] for x in observations])
    plt.plot([x[0] for x in track], [x[1] for x in track])
    #plt.plot([x[0] for x in track_smoothed], [x[1] for x in track_smoothed])

    plt.show()


if __name__ == "__main__":
    main()
