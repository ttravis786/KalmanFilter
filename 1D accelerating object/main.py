import matplotlib.pyplot as plt
import numpy as np
from  kalman_filter import KalmanFilter

def main():
    dt = 2.0
    t = np.arange(0, 100, dt)
    # Define a model track
    real_track = 0.1*((t**2) - t)
    u = 2
    std_acc = 0.25     # we assume that the standard deviation of the acceleration is 0.25 (m/s^2)
    std_meas = 1.2    # and standard deviation of the measurement is 1.2 (m)
    # create KalmanFilter object
    kf = KalmanFilter(dt, u, std_acc, std_meas)
    predictions = []
    measurements = []
    for x in real_track:
        # Mesurement
        z = kf.H * x + np.random.normal(0, 5)
        measurements.append(z.item(0))
        predictions.append(kf.predict()[0])
        kf.update(z.item(0))
    fig = plt.figure()
    fig.suptitle('Example of Kalman filter for tracking a moving object in 1-D', fontsize=20)
    plt.plot(t, measurements, '.', label='Measurements', color='b')
    plt.plot(t, np.array(real_track), label='Real Track', color='y', linewidth=1.5)
    plt.plot(t, np.squeeze(predictions), label='Kalman Filter Prediction', color='r', linewidth=1.5)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Position (m)', fontsize=20)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()