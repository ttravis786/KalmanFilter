import sys

import pandas as pd
sys.path.append("./CombinedSim/DetectorSim")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import time
from Seeding.linear_regression import linear_regression, linear_func
sys.path.append(r'\\wsl$\Ubuntu\home\tt1020\CombinedSim\DetectorSim')
from Detector import Detector
class PreProcesser():

    def __init__(self, df):
        self.original_df = df.sort_values(['time'])
    def geometric_cluster(self, t_step=2, cut_off=0.8, dist=0.6, merge = True):
        df_n = self.original_df.copy()
        bins = np.arange(df_n.time.min(), df_n.time.max(), t_step)
        group_by = df_n.groupby(pd.cut(df_n.time, bins))
        #group = group_by.get_group(list(group_by.groups.keys())[0])
        # look for 3*3
        data_dict = {'x':[], 'y':[], 'error':[], 'time':[]}
        for name, group in group_by:
            group = group.sort_values(['activation'], ascending=False)
            bool_arr = group.activation > cut_off
            activation_data = zip(group.activation[bool_arr],
                                group.panel_x[bool_arr],
                                group.panel_y[bool_arr])
            nn_groups = {}
            activations_checked = []
            for activation, x, y in activation_data:
                if activation in activations_checked:
                    continue
                nn_group = group[((group.panel_x >= x - dist) & (group.panel_x <= x + dist)
                            & (group.panel_y >= y - dist) & (group.panel_y <= y + dist))]
                nn_group_act_max = nn_group.activation.max()
                if nn_group_act_max == activation:
                    activations_checked += list(nn_group.activation)
                    nn_groups[activation] = nn_group
                else:
                    activations_checked += list(nn_group.activation)
                    if merge:
                        for group_name, group in nn_groups.items():
                            if (group.activation == nn_group_act_max).sum():
                                nn_groups[group_name] = pd.merge(group, nn_group, how='outer')
                    else:
                        nn_groups[activation] = nn_group

            for nn_group in nn_groups.values():
                central_point = ((nn_group.activation.dot(nn_group[['panel_x',  'panel_y']]))) / nn_group.activation.sum()
                error = 1 / np.sqrt(len(nn_group))
                data_dict['x'].append(central_point[0])
                data_dict['y'].append(central_point[1])
                data_dict['error'].append(error)
                data_dict['time'].append(nn_group['time'].mean())

        return pd.DataFrame(data_dict)


                # could merge to extend group probs won't


            # go through activations
            # select highest. remove all entries from table included in it's 3*3 for selecting next big points
            # proceed to next highest, if there is another point higher in 3*3 incorepate it. and repeat but use all values for total calc not just half empty table.
            # repeat until threshold reached.

            # go through these until one of the brighter activations touches another or the min threshold is reached.



        # sorts the group by activation.
        # looks for a 3*3 around it.
        # completes central point.
        # looks at next highest activation
        # firstly filter by time.
        # sum activation

import pickle
with open('test_data\detector.pickle', 'rb') as handle:
    detector = pickle.load(handle)

df = pd.read_csv(r'test_data/test.csv')
#del df['Unnamed: 0']
df.time = df.time * 1e9
df = df.sort_values(by='time')
# find cycle
df = df[df.time <=25]
preprocessor = PreProcesser(df)
a = time.time()
# for i in range(0,30):
#     preprocessed_data = preprocessor.geometric_cluster(t_step=1, cut_off=0.6, dist=0.6)
# print(time.time() - a)
fig, ax = plt.subplots()
#ax.plot(df.real_x,df.real_y, 'o')
# preprocessed_data = preprocessor.geometric_cluster(t_step=0.1, cut_off=1.0, dist=0.6, merge=True)
# ax.plot(preprocessed_data.x,preprocessed_data.y, '.')
ax.plot(df.real_x,df.real_y, '.')
preprocessed_data = preprocessor.geometric_cluster(t_step=0.3, cut_off=0.7, dist=0.6, merge=True)
ax.plot(preprocessed_data.x, preprocessed_data.y, 'x')
detector = Detector()
detector.plot_output_data(fig_ax=(fig,ax), df=df)

x, y = np.array(preprocessed_data.x), np.array(preprocessed_data.y)

points = np.array([x,y]).T

B, B_cov_mat, var = linear_regression(points=points)
x_fit = np.linspace(-27, 27, 1000)
y_fit = linear_func(x_fit, params=B)

plt.plot(x_fit, y_fit)
