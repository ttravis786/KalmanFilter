import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from AdvancedKalmanFilter.kalman_filter import H_func
from Seeding import linear_regression
from combinatorial_kalman_filter_old_parameterisation.ckf_branch import CKF_branch
import  matplotlib.pyplot as plt


class CKF():
    def __init__(self, max_range=1, bin_width=0.5, min_points=8,
                 fit_order=2, start_range=((25, 28),(-7,7)), R_k=None,
                 max_res=2.0, check_branch_n=3, max_iteration=10):
        self.max_range = max_range
        self.bin_width = bin_width
        self.min_points = min_points
        self.fit_order = fit_order
        self.len_lin_reg = fit_order + 1
        self.event_data = []
        self.event_params = []
        self.start_range = start_range
        self.F = np.identity(fit_order+1)
        self.H = H_func
        self.max_res = max_res
        self.check_branch_n = check_branch_n
        self.max_iteration = max_iteration
        if R_k is None:
            var = 0.5**2
            self.R_k = np.zeros((self.fit_order+1, self.fit_order+1))
            self.R_k[0,0] = var
        else:
            self.R_k = R_k

    def bin_finder(self, point):
        return tuple([math.floor(i / self.bin_width) * self.bin_width for i in point])

    # split it into bins to be processed
    def add_event(self, data):
        # do it in a way easily implemented in C++ (no pandas functions)
        points_map = {}
        for point in data:
            bins = self.bin_finder(point)
            if not bins in points_map.keys():
                points_map[bins] = []
            points_map[bins].append(point)
        self.event_data.append(points_map)

    def _next__cells1D(self, pos, delta_sign, i, inclusive=True, double_length=False):
        if inclusive:
            extra = self.bin_width
            start = 0
        else:
            start = self.bin_width
            extra = 0
        if double_length:
            return np.arange(pos + (-i * self.bin_width + start), pos + (i * self.bin_width + extra), self.bin_width)
        return np.arange(pos, pos + delta_sign * (i * self.bin_width + extra), delta_sign * self.bin_width)


    def next_cells(self, pos, i, prev=False):
        if not prev:
            prev = pos
        delta_x = pos[0] - prev[0]
        delta_y = pos[1] - prev[1]
        import math

        if delta_x != 0:
            delta_x_sign = math.copysign(1, delta_x)
            if delta_y != 0:
                # x increasing y constant
                delta_y_sign = math.copysign(1, delta_y)
                x_inc =  [(x, pos[1] + delta_y_sign * i * self.bin_width) for x in
                          self._next__cells1D(pos[0], delta_x_sign, i, inclusive=True)]
                # take away corner
                y_inc = [(pos[0] + delta_x_sign * i * self.bin_width, y) for y in
                         self._next__cells1D(pos[1], delta_y_sign, i, inclusive=False)]
            else:
                # delta y double length not inclusive
                # double delta x inclusive
                x_points = self._next__cells1D(pos[0], delta_x_sign, i, inclusive=True)
                x_inc =  ([(x, pos[1] + i * self.bin_width) for x in x_points]
                          + [(x, pos[1] - i * self.bin_width) for x in x_points])
                # take away corner
                y_inc = [(pos[0] + delta_x_sign * i * self.bin_width, y) for y in
                         self._next__cells1D(pos[1], 1, i, inclusive=False, double_length=True)]
        else:
            if delta_y !=0:
                # delta x double length not inclusive
                # double delta y inclusive
                delta_y_sign = math.copysign(1, delta_y)
                y_points = self._next__cells1D(pos[1], delta_y_sign, i, inclusive=True)
                y_inc =  ([(pos[0] + i * self.bin_width, y) for y in y_points]
                          + [(pos[0] - i * self.bin_width, y) for y in y_points])
                # take away corner
                x_inc = [(x, pos[1] + delta_y_sign * i * self.bin_width) for x in
                         self._next__cells1D(pos[0], 1, i, inclusive=False, double_length=True)]
            else:
                # delta x double length not inclusive
                # double delta y double lenth inclusive
                y_points = self._next__cells1D(pos[1], 1, i, inclusive=True, double_length=True)
                x_points = self._next__cells1D(pos[0], 1, i, inclusive=False, double_length=True)
                y_inc =  ([(pos[0] + i * self.bin_width, y) for y in y_points]
                          + [(pos[0] - i * self.bin_width, y) for y in y_points])
                x_inc =  ([(x, pos[1] + i * self.bin_width) for x in x_points]
                          + [(x, pos[1] - i * self.bin_width) for x in x_points])
        return x_inc + y_inc

    def _add_points(self, points, curr_cell, prev_cell, current_l, max_l, points_map):
        if current_l == max_l:
            return [self._process_quintet(points, curr_cell, prev_cell, points_map)]
        possible_seeds = []
        for i in range(1, self.max_range):
            next_cells = self.next_cells(curr_cell, i, prev=prev_cell)
            for cell in next_cells:
                try:
                    points_map[cell]
                    print('has_next')
                except:
                    continue
                for next_point in points_map[cell]:
                    print('has_next')
                    new_seeds = self._add_points(points + [next_point], cell,
                                                 curr_cell, current_l+1, max_l,
                                                 points_map)
                    possible_seeds += new_seeds
            if possible_seeds:
                return possible_seeds
        return possible_seeds

    def _process_quintet(self, points, curr_cell, prev_cell, points_map):
        points_arr = np.array(points)
        # compute linear regression.
        B, B_cov_mat, var = linear_regression.linear_regression(points=points_arr, order=self.fit_order)
        # test params of quintet to see if it works? i.e charge_momentum etc
        # initiate kalman filter
        master_branch = CKF_branch(B, B_cov_mat, curr_cell, prev_cell, points, len(points), self)
        master_branch.propogate(self.max_range, points_map, self.max_res)
        best_child,_ ,_ = master_branch.get_best_child()
        return best_child, master_branch
        # now combinitorial add points, checking branch at each step
        # at each branch point create kalman filter



    def _process_seed(self, point, points_map,
                      pos):
        # do quartic for first five:
        fits = self._add_points([point], pos, pos, 1, self.fit_order+2, points_map)
        all_points = []
        true_fits = []
        for f, m in fits:
            if f is None:
                continue
            if f.total_points <= self.min_points:
                continue
            all_points += f.full_points()
            true_fits.append([f,m])
        return all_points, true_fits

    def _process_branch(self):
        pass

    # find tracks and corresponding paramaters
    def find_tracks(self, event=0):
        points_map = self.event_data[event]
        possible_tracks_arr = []
        track_seeds_used = []
        for x in np.arange(self.start_range[0][1], self.start_range[0][0] - self.bin_width, -self.bin_width):
            for y in np.arange(self.start_range[1][0], self.start_range[1][1] + self.bin_width, self.bin_width):
                try:
                    points_map[(x, y)]
                except:
                    continue
                for point in points_map[(x,y)]:
                    try:
                        if point in np.array(track_seeds_used):
                            continue
                    except:
                        pass
                    used_points, possible_tracks = self._process_seed(point, points_map,(x,y))
                    possible_tracks_arr.append(possible_tracks)
                    track_seeds_used += used_points
                    # for now
                    #return [possible_tracks]

        return possible_tracks_arr

