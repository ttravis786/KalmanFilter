# linear regression on 3 data points
import numpy as np

def conversion(points=np.array([[1.04,1.1], [2,2.09,], [3, 3.11]]), d_0=np.array([1.04,1.1]), cartesian=True):
    # d_0 will be start point i.e. origin.
    cenetred_points = points - d_0
    if cartesian:
        return cenetred_points
    else:
        d = np.linalg.mag(cenetred_points)
        theta = np.arctan2(cenetred_points[:,1], cenetred_points[:,0])
        return np.array([d, theta]).T

# def convert_parameters(parameters):
#     d_0 =

def linear_func(x, params):
    n = len(params)
    return sum([params[i] * x**(n-i-1) for i in range(0, n)])

def linear_regression(points=np.array([[1.04,1.1], [2,2.09,], [3, 3.11]]), order=3, W=None):
    Y = points[:, 1]
    if W is None:
        W = np.identity(len(Y))
    X = np.array([points[:, 0] ** i for i in range(order, -1, - 1)]).T
    B = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(X.T, W), X)), X.T), W).dot(points[:, 1])
    N = len(Y)
    p = order + 1
    res = Y - X.dot(B)
    var = (1 / (N - p+1)) * np.sum(res**2)
    B_cov_mat = var * np.linalg.inv(np.matmul(X.T, X))
    return B, B_cov_mat, var

