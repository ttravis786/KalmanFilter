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

class PreProcesser():

    def __init__(self, df):
        self.original_df = df.sort_values(['time'])
    def geometric_cluster(self, t_step=2, cut_off=0.8, dist=0.6, merge = True, grouper='activation'):
        df_n = self.original_df.copy()
        bins = np.arange(df_n.time.min(), df_n.time.max(), t_step)
        group_by = df_n.groupby(pd.cut(df_n.time, bins))
        #group = group_by.get_group(list(group_by.groups.keys())[0])
        # look for 3*3
        data_dict = {'x':[], 'y':[], 'error':[], 'time':[]}
        for name, group in group_by:
            group = group.sort_values([grouper], ascending=False)
            bool_arr = group[grouper] > cut_off
            activation_data = zip(group.loc[bool_arr, grouper],
                                group.panel_x[bool_arr],
                                group.panel_y[bool_arr])
            nn_groups = {}
            activations_checked = []
            for activation, x, y in activation_data:
                if activation in activations_checked:
                    continue
                nn_group = group[((group.panel_x >= x - dist) & (group.panel_x <= x + dist)
                            & (group.panel_y >= y - dist) & (group.panel_y <= y + dist))]
                nn_group_act_max = nn_group[grouper].max()
                if nn_group_act_max == activation:
                    activations_checked += list(nn_group[grouper])
                    nn_groups[activation] = nn_group
                else:
                    activations_checked += list(nn_group[grouper])
                    if merge:
                        for group_name, group in nn_groups.items():
                            if (group[grouper] == nn_group_act_max).sum():
                                nn_groups[group_name] = pd.merge(group, nn_group, how='outer')
                    else:
                        nn_groups[activation] = nn_group

            for nn_group in nn_groups.values():
                central_point = ((nn_group[grouper].dot(nn_group[['panel_x',  'panel_y']]))) / nn_group[grouper].sum()
                error = 1 / np.sqrt(len(nn_group))
                data_dict['x'].append(central_point['panel_x'])
                data_dict['y'].append(central_point['panel_y'])
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
custom = True
if custom:
    sys.path.append(r'\\wsl$\Ubuntu\home\tt1020\CombinedSim')
    sys.path.append(r'\\wsl$\Ubuntu\home\tt1020\CombinedSim\DetectorSim')
    sys.path.append(r'\\wsl$\Ubuntu\home\tt1020\CombinedSim\TrackSim')
    import combinedsim
    output, detector = combinedsim.run_sim({'muon':1}, 0.5, 0.5)
    df = output['detector']
    df.time = df.time * 1e9
    df = df.sort_values(by='time')
    df = df[df.time <= 25]

if __name__ == "__main__":

    #df = pd.read_csv(r'test_data/one_muon_sigmoid1.csv')
    #del df['Unnamed: 0']

    # find cycle
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
    detector.plot_output_data(fig_ax=(fig,ax), df=df)

    x, y = np.array(preprocessed_data.x), np.array(preprocessed_data.y)

    points = np.array([x,y]).T

    B, B_cov_mat, var = linear_regression(points=points[0:18:4], order=2)
    x_fit = np.linspace(-27, 27, 1000)
    y_fit = linear_func(x_fit, B)

    plt.plot(x_fit, y_fit)
