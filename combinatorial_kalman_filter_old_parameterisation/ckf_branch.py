import numpy as np
import copy
class CKF_branch():
    def __init__(self, X, P, curr_cell, prev_cell,
                 points, total_points, ckf, master=None, iteration=1):
        self.X = X
        self.P = P
        self.curr_cell = curr_cell
        self.prev_cell = prev_cell
        self.points = points
        self.master = master
        self.children = None
        self.total_points = total_points
        self.ckf = ckf
        self.iteration = iteration


    def add_children(self, children):
        self.children = children#

    def get_children(self):
        return self.children

    def resid_func(self, max_res, x, calculate_uncertainty=True):
    #     if self.total_points < 15:
    #         return 3.0
        if calculate_uncertainty:
            B_cov_mat_n = self.P
            n = len(B_cov_mat_n.diagonal())
            a = (max_res +
                    sum([abs(b_c * x**(n-i-1)) for i, b_c in enumerate(B_cov_mat_n.diagonal() ** 0.5)]))
            #print(f'residual: {a}')
            return a


    def goodness_measure(self):
        B_n, B_cov_mat_n = self.X, self.P
        return sum([b_c/abs(b) for b, b_c in zip(B_n, B_cov_mat_n.diagonal()**0.5)])

    def full_points(self):
        if self.master is None:
            return self.points
        return self.master.full_points() + self.points

    def get_all_children(self):
        if self.children is None:
            return self
        else:
            return [child.get_all_children() for child in self.children]


    def get_best_child(self):
        if self.children is None:
            return self, self.goodness_measure(), self.total_points
        else:
            best_goodness = 1e8
            best_child = None
            best_total_points = 0
            for c in self.children:
                pc, pc_goodness_measure, total_points = c.get_best_child()
                if ((pc_goodness_measure < best_goodness) and
                        (total_points >= self.ckf.min_points)):
                    best_goodness = pc_goodness_measure
                    best_child = pc
                    best_total_points = best_child.total_points
            if best_child is None:
                return self, self.goodness_measure(), self.total_points
            return best_child, best_goodness, best_total_points

    def get_suitable_children(self, error=0.0):
        if self.children is None:
            return [self]
        else:
            full_children = []
            for ch in self.children:
                f_c = ch.get_suitable_children()
                for c in f_c:
                    full_children.append(c)
            return full_children
    def _update_kf(self, x_cord, y_cord):
        ###predict###
        # x_k_k-1 = F_k * x_k-1_k-1 + B_k * U_k
        X_k_km1 = np.matmul(self.ckf.F, self.X)
        # P_k_k-1 = F_k * P_k-1_k-1 * (F**T)_k  + Q_k
        P_k_km1 = np.matmul(self.ckf.F, self.P)
        y_cord = np.array([y_cord] + [0] * self.ckf.fit_order)
        ### update ###
        H = self.ckf.H(x_cord, order=self.ckf.fit_order)
        # innovation residual
        y_k = y_cord - np.matmul(H, X_k_km1)
        # innovation covariance
        S_k = np.matmul(np.matmul(H, P_k_km1), H.T) + self.ckf.R_k
        # optimal Kalman Gain
        inv_S_k = S_k.copy()
        inv_S_k[0,0] = 1/inv_S_k[0,0]
        K = np.matmul(np.matmul(P_k_km1, H.T), inv_S_k)
        # x_k_k
        X = X_k_km1 + np.matmul(K, y_k)
        # P_K_k
        P = np.matmul((np.identity(self.ckf.fit_order+1) - np.matmul(K, H)),
                           P_k_km1)
        y_k_k = abs(y_cord - np.matmul(H, X))
        return X, P, y_k_k

    def propogate(self, max_range, points_map, max_res, try_all=False):
        if self.iteration > self.ckf.max_iteration:
            return
            #print(self.total_points)
        # if self.total_points == 9:
        #     a = 1
        # if np.all(self.points[-1] == np.array([18.48508348, -1.78393135])):
        #     a = 1
                # if np.all(self.full_points()[-1] == np.array([19.25, -4.98085009])):
        possible_next_points = []
        if try_all:
            directions = [True, False]
        else:
            directions = [True]
        for direction_following in directions:
            if possible_next_points:
                return
            for i in range(1, max_range):
                possible_next_points = []
                if direction_following:
                    next_cells = self.ckf.next_cells(self.curr_cell, i, prev=self.prev_cell)
                else:
                    next_cells = self.ckf.next_cells(self.curr_cell, i)
                for cell in next_cells:
                    try:
                        points_map[cell]
                    except:
                        continue
                    for next_point in points_map[cell]:
                        # set of checks including residual (do more)
                        X, P, res = self._update_kf(next_point[0], next_point[1])
                        if np.any([np.all(next_point == p) for p in self.full_points()]):
                            continue
                        #print(res[0])
                        #print(self.resid_func(max_res, next_point[0]))
                        if abs(res[0]) < self.resid_func(max_res, next_point[0]):
                            possible_next_points.append((X, P, next_point, cell))
                if len(possible_next_points) == 1:
                    X, P, next_point, cell = possible_next_points[0]
                    self.X = X
                    self.P = P
                    self.prev_cell = self.curr_cell
                    self.curr_cell = cell
                    #points_map_n = copy.deepcopy(points_map)
                    #del points_map_n[cell]
                    self.points.append(next_point)
                    self.total_points += 1
                    self.iteration += 1
                    self.propogate(max_range, points_map, max_res)
                    return
                    # branch
                elif len(possible_next_points) > 1:
                    self.children = []
                    for p in possible_next_points:
                        X, P, next_point, cell = p
                        self.children.append(CKF_branch(X, P, cell, self.curr_cell, [next_point],
                                                        self.total_points + 1, self.ckf, master=self, iteration=self.iteration+1))
                    for c in self.children:
                        #points_map_n = copy.deepcopy(points_map)
                        #del points_map_n[c.curr_cell]
                        c.propogate(max_range, points_map, max_res)
                    return
                    # test all these












