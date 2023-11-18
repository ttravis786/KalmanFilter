import numpy as np

def H_func(x, order=2):
    return np.array(([[x**(order - i) for i in range(0,order+1)]] +
                    [[0 for i in range(0, order+1)] for i in range(order)]))

def generate_params(fit_params, fit_params_cov):
    X = fit_params
    P = fit_params_cov
    F = np.identity(len(X))
    H = H_func
    return F,X,P,H,0,0
    # start at 0.5
    #K = np.identity(order + 1)/2


class Kalman_Filter():
    def __init__(self, params):
        F,X,P,H,B,U = params
        self.F = F
        # y = HX + err therefore for ax**2 +bx + c the matrix contains x**2 x and 1
        self.order = len(X) - 1
        self.H = H
        # X = parameters i.e. a,b,c for ax**2 +bx + c
        self.X = X
        # covariamce
        self.P = P
        # control input noise (unknown)
        #self.B = B

    def update(self, x_cord, y_cord, R_k=None):
        ###predict###
        if R_k is None:
            var = 0.5**2
            R_k = np.zeros((self.order+1, self.order+1))
            R_k[0,0] = var
        # x_k_k-1 = F_k * x_k-1_k-1 + B_k * U_k
        X_k_km1 = np.matmul(self.F, self.X)
        # P_k_k-1 = F_k * P_k-1_k-1 * (F**T)_k  + Q_k
        P_k_km1 = np.matmul(self.F, self.P)
        y_cord = np.array([y_cord] + [0] * self.order)
        ### update ###
        H = self.H(x_cord, order=self.order)
        # innovation residual
        y_k = y_cord - np.matmul(H, X_k_km1)
        # innovation covariance
        S_k = np.matmul(np.matmul(H, P_k_km1), H.T) + R_k
        # optimal Kalman Gain
        inv_S_k = S_k.copy()
        inv_S_k[0,0] = 1/inv_S_k[0,0]
        K = np.matmul(np.matmul(P_k_km1, H.T), inv_S_k)
        # x_k_k
        self.X = X_k_km1 + np.matmul(K, y_k)
        # P_K_k
        self.P = np.matmul((np.identity(self.order+1) - np.matmul(K, H)),
                           P_k_km1)
        y_k_k = y_cord - np.matmul(H, self.X)

    def add_points(self, x, y):
        for x, y in zip(x, y):
            self.update(x, y)
        return self.X, self.P





