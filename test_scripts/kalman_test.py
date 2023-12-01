from AdvancedKalmanFilter import kalman_filter
from clustering import preprocessor
from Seeding import linear_regression#
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append(r'\\wsl$\Ubuntu\home\tt1020\CombinedSim')
sys.path.append(r'\\wsl$\Ubuntu\home\tt1020\CombinedSim\DetectorSim')
sys.path.append(r'\\wsl$\Ubuntu\home\tt1020\CombinedSim\TrackSim')
import combinedsim

# Get data
output, detector = combinedsim.run_sim({'muon': 5}, 0.5, 0.5)
df = output['detector']
df.time = df.time * 1e9
df = df.sort_values(by='time')
df = df[df.time <= 25]

# Process it
preprocessor_ob = preprocessor.PreProcesser(df)
fig, ax = plt.subplots()
ax.plot(df.real_x, df.real_y, '.', label='truth data')
preprocessed_data = preprocessor_ob.geometric_cluster(t_step=0.3, cut_off=0.7, dist=0.6, merge=True)
ax.plot(preprocessed_data.x, preprocessed_data.y, 'x', label='clustered data')
detector.plot_output_data(fig_ax=(fig, ax), df=df)
x, y = np.array(preprocessed_data.x), np.array(preprocessed_data.y)
points = np.array([x, y]).T

run_linear_regression = False

if run_linear_regression:
    ## preform Linear Regression
    B, B_cov_mat, var = linear_regression.linear_regression(points=points, order=3)
    x_fit = np.linspace(-27, 27, 1000)
    y_fit = linear_regression.linear_func(x_fit, B)
    plt.plot(x_fit, y_fit, label='Linear Regression')
    print(B_cov_mat)


run_kalman = False

if run_kalman:
    ## preform Linear Regression
    B, B_cov_mat, var = linear_regression.linear_regression(points=points[0:6], order=3)
    x_fit = np.linspace(-27, 27, 1000)
    y_fit = linear_regression.linear_func(x_fit, B)
    plt.plot(x_fit, y_fit, label='Initial Linear Regression')
    # Preform Kalman Filter
    kalman_params = kalman_filter.generate_params(B, B_cov_mat)
    kalman_filter_ob = kalman_filter.Kalman_Filter(kalman_params)
    B_n, B_cov_mat_n = kalman_filter_ob.add_points(x, y)
    y_fit_n = linear_regression.linear_func(x_fit, B_n)
    plt.plot(x_fit, y_fit_n, label='Kalman Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    print(B_cov_mat_n)

run_kalman_benchmark = False
a = time.time()
if run_kalman_benchmark:
    for i in range(0, 1000):
        ## preform Linear Regression
        B, B_cov_mat, var = linear_regression.linear_regression(points=points[0:6], order=3)
        # preform kalman filter
        kalman_params = kalman_filter.generate_params(B, B_cov_mat)
        kalman_filter_ob = kalman_filter.Kalman_Filter(kalman_params)
        print(len(y))
        B_n, B_cov_mat_n = kalman_filter_ob.add_points(x, y)
    print(time.time() - a)


run_linear_regression_benchmark = False

a = time.time()
if run_linear_regression_benchmark:
    for i in range(0, 1000):
        ## preform Linear Regression
        W = (np.identity(len(points)) * 1 /(0.5**2))
        B, B_cov_mat, var = linear_regression.linear_regression(points=points, order=3, W=W)
    print(time.time() - a)

run_kalman_points_benchmark = False

a = time.time()
if run_kalman_points_benchmark:
    time_array = []
    points_array = np.logspace(1, 13, base=1.5, num=50)
    for i in points_array:
        a = time.time()
        x_n, y_n = list(points[:,0]), list(points[:,1])
        x_n = x_n * int(i)
        y_n = y_n * int(i)
        points_n = np.array([x_n,y_n]).T
        a = time.time()
        ## preform Linear Regression
        B, B_cov_mat, var = linear_regression.linear_regression(points=points_n[0:6], order=3)
        print(B)
        # preform kalman filter
        kalman_params = kalman_filter.generate_params(B, B_cov_mat)
        kalman_filter_ob = kalman_filter.Kalman_Filter(kalman_params)
        B_n, B_cov_mat_n = kalman_filter_ob.add_points(x_n, y_n)
        print(B_n)
        time_array.append(time.time()- a)
        break
    plt.plot(points_array, time_array,  label='Kalman Filter')

run_linear_regression_points_benchmark=False

a = time.time()
if run_linear_regression_points_benchmark:
    time_array = []
    points_array = np.logspace(1, 13, base=1.5, num=50)
    for i in points_array:
        a = time.time()
        x_n, y_n = list(points[:,0]), list(points[:,1])
        x_n = x_n * int(i)
        y_n = y_n * int(i)
        points_n = np.array([x_n,y_n]).T
        a = time.time()
        ## preform Linear Regression
        W = (np.identity(len(points_n)) * 1 /(0.5**2))
        B, B_cov_mat, var = linear_regression.linear_regression(points=points_n, order=3, W=W)
        time_array.append(time.time()- a)
    plt.plot(points_array, time_array, label='Linear Regression')
    plt.xlabel('Relative n.of points')
    plt.ylabel('n.of points')
    plt.legend()
