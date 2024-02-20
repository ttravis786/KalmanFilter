location = r"\\wsl$\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-debug"
location_ten = r"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\simulation_data"
location = r"\\wsl$\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release"
from combinatorial_kalman_filter_old_parameterisation import ckf
from clustering import preprocessor
import sys
sys.path.append(r'\\wsl$\Ubuntu\home\tt1020\CombinedSim')
sys.path.append(r'\\wsl$\Ubuntu\home\tt1020\CombinedSim\DetectorSim')
sys.path.append(r'\\wsl$\Ubuntu\home\tt1020\CombinedSim\TrackSim')
import combinedsim
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from scipy.optimize import curve_fit

def for_CPP(filename, muonNum=4):
    # Get data
    output, detector = combinedsim.run_sim({'muon': muonNum}, 0.5, 0.5)
    df = output['detector']
    df.time = df.time * 1e9
    df = df.sort_values(by='time')
    df = df[df.time <= 25]

    # Process it
    preprocessor_ob = preprocessor.PreProcesser(df)
    # fig, ax = plt.subplots()
    # ax.plot(df.real_x, df.real_y, '.', label='truth data')
    preprocessed_data = preprocessor_ob.geometric_cluster(t_step=0.3, cut_off=0.7, dist=0.6, merge=True)
    # ax.plot(preprocessed_data.x, preprocessed_data.y, 'x', label='clustered data')
    # detector.plot_output_data(fig_ax=(fig, ax), df=df)|
    df_to_save = preprocessed_data[['x', 'y']]
    df_to_save.to_csv(rf"{location}\{filename}", index=False)

def tenThousandTracks():
    df = pd.read_json(rf"{location_ten}\tenthousand_events.json")
    # for each event start processing the data
    for i in range(df.panel_x.size):
        panel_x = np.array(list(df.panel_x[i].values()))
        panel_y = np.array(list(df.panel_y[i].values()))
        energy = np.array(list(df.panel_y[i].values()))
        time = np.array(list(df.panel_y[i].values()))

def for_CPP_ten_track(filename):
    df = pd.read_json(rf"{location_ten}\ten_track.json")
    df.time = df.time * 1e9
    df = df.sort_values(by='time')
    preprocessor_ob = preprocessor.PreProcesser(df)
    # fig, ax = plt.subplots()
    # ax.plot(df.real_x, df.real_y, '.', label='truth data')
    preprocessed_data = preprocessor_ob.geometric_cluster(t_step=0.3, cut_off=0.7, dist=0.6, merge=True,grouper='energy')
    # ax.plot(preprocessed_data.x, preprocessed_data.y, 'x', label='clustered data')
    # detector.plot_output_data(fig_ax=(fig, ax), df=df)|
    df_to_save = preprocessed_data[['x', 'y']]
    df_to_save.to_csv(rf"{location}\{filename}", index=False)

def for_cpp_custom(filename, x, y):
    df = pd.DataFrame({'x':x, 'y':y})
    df.to_csv(rf"{location}\{filename}", index=False)

def Gauss(x, A, B, C):
    y = A*np.exp(-1*(1/(2*B**2))*((x-C)**2))
    return y

def DoubleGauss(x, A1, B1, C1, A2, B2, C2):
    y = (A1*np.exp(-1*(1/(2*B1**2))*(x-C1)**2) + A2*np.exp(-1*(1/(2*B2**2))*(x-C2)**2))
    return y
def DoubleGaussC(x, A1, B1, C1, A2, B2, C2, c):
    y = (A1*np.exp(-1*(1/(2*B1**2))*(x-C1)**2) + A2*np.exp(-1*(1/(2*B2**2))*(x-C2)**2)) + c
    return y

def TripleGauss(x, A1, B1, C1, A2, B2, C2, A3, B3, C3):
    y = (A1*np.exp(-1*(1/(2*B1**2))*(x-C1)**2)
         + A2*np.exp(-1*(1/(2*B2**2))*(x-C2)**2)
         + A3*np.exp(-1*(1/(2*B3**2))*(x-C3)**2))
    return y

def fitTripleGauss(x, y, p0):
    parameters, covariance = curve_fit(TripleGauss, x, y, p0=p0)
    return parameters, covariance

def fitDoubleGauss(x, y, p0):
    parameters, covariance = curve_fit(DoubleGauss, x, y, p0=p0)
    return parameters, covariance

def fitDoubleGaussC(x, y, p0):
    parameters, covariance = curve_fit(DoubleGaussC, x, y, p0=p0)
    return parameters, covariance

def fitGauss(x, y, p0):
    parameters, covariance = curve_fit(Gauss, x, y, p0=p0)
    return parameters, covariance

def bins_to_points(data):
    x = (data[1][1:] + data[1][0:-1])/2
    y = data[0]
    return x,y

def a_to_momentum(a):
    r = - 2/a
    p = r * 0.5

def momentum_dist():
    df = pd.read_csv(r"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release\tenThousandTrackMomentumData.csv")
    fig,ax = plt.subplots()
    mom = -7.5/df.a
    mom = mom[((mom <3000) & (mom >-3000))]
    data_hist = ax.hist(mom, bins=150)
    hist_x, hist_y = bins_to_points(data_hist)
    parameters = [225,200,-1000, 60,200,0, 225,200,1000]
    x = np.linspace(-2000,2000,100000)
    parameters, covariance = fitTripleGauss(hist_x, hist_y, p0=parameters)
    ax.plot(x, TripleGauss(x, *parameters), label=f'Gaussian Fit')
    ax.set_xlabel("Momentum MeV/C")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig("MomentumDistribution.png")

    fig, ax = plt.subplots()
    mag_mom = 7.5/np.abs(df.a)
    mag_mom = mag_mom[mag_mom<3000]
    data_hist = ax.hist(mag_mom, bins=100)
    hist_x, hist_y = bins_to_points(data_hist)
    parameters = [150,200,0, 600,200,1000]
    x = np.linspace(0,2000,100000)
    parameters, covariance = fitDoubleGauss(hist_x, hist_y, p0=parameters)
    ax.plot(x, DoubleGauss(x, *parameters), label=f'Gaussian Fit\n'
                                                  f'Main Peak $\mu$ = {parameters[5]:.2f} $\pm$ {covariance[5,5]**1/2:.2f}\n'
                                                  f'$\sigma$ = {parameters[4]:.2f} $\pm$ {covariance[4,4]**1/2:.2f}')
    #fitDoubleGauss()
    ax.set_xlabel("Absolute Momentum MeV/c")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig("AbsoluteMomentumDistribution.png")
def assign_event_size(df):
    df_true_num = pd.read_csv(
        r'\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release\tenThousandTrackNum.csv').set_index('Event')
    df['event_size'] = df.event.apply(lambda x: df_true_num.loc[x].Size)
    return df

def momentum_dist_tracksize():
    df = pd.read_csv(
        r"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release\tenThousandTrackMomentumData.csv")
    df = df[df.event < 2000]
    mom = -7.5 / df.a
    df = df[((mom < 3000) & (mom > -3000))]
    df = assign_event_size(df)
    df_groups = df.groupby('event_size')
    gaussian_width = []
    gaussian_width_unc = []
    gaussian_mean = []
    track_size = []
    for key in df_groups.groups.keys():
        df_group = df_groups.get_group(key)
        fig, ax = plt.subplots()
        data_hist = ax.hist(7.5/np.abs(df_group.a), bins=20)
        hist_x, hist_y = bins_to_points(data_hist)
        parameters = [np.max(hist_y)/5,200,0, np.max(hist_y),250,1000]
        x = np.linspace(0,2000,100000)
        try:
            parameters, covariance = fitDoubleGauss(hist_x, hist_y, p0=parameters)
            ax.plot(x, DoubleGauss(x, *parameters), label=f'Gaussian Fits EventSize:{key}')
            gaussian_width.append(parameters[4])
            gaussian_mean.append(parameters[5])
            gaussian_width_unc.append(covariance[4,4]**0.5)
            track_size.append(key)
        except:
            try:
                parameters = [np.max(hist_y), 150, 1000]
                parameters, covariance = fitGauss(hist_x, hist_y, p0=parameters)
                ax.plot(x, Gauss(x, *parameters), label=f'Gaussian Fits EventSize:{key}')
                gaussian_width.append(parameters[1])
                gaussian_mean.append(parameters[2])
                gaussian_width_unc.append(covariance[1,1]**0.5)
                track_size.append(key)
            except:
                ax.plot(x, Gauss(x, *parameters), label=f'Gaussian Fits EventSize:{key}')
                print(f"no convergence track {key}")
        ax.set_xlabel("absolute momentum")
        ax.set_ylabel("freq")
        ax.legend()
        fig.tight_layout()
    fig, ax = plt.subplots()
    ax.errorbar(track_size, gaussian_width, yerr=gaussian_width_unc, fmt='o', color='b')
    ax.legend()
    ax.set_ylabel('Projected Momentum Standard Deviation MeV/c')
    ax.set_xlabel('Event Size (Number of Tracks)')
    fig.tight_layout()
    fig.savefig("STDMomVTrack.png")
    fig, ax = plt.subplots()
    ax.errorbar(track_size, gaussian_mean, yerr=gaussian_width, fmt='o', color='b',label=f"Value - Gaussian Fit: $\mu$ \n"
                                                                                         f"ErrorBar - Gaussian Fit: $\sigma$")
    ax.legend()
    ax.set_ylabel('Projected Momentum Mean MeV/c')
    ax.set_xlabel('Event Size (n.of tracks)')
    fig.tight_layout()
    fig.savefig("MomVTrack.png")



def truth_calc_mom_comp():
    df = pd.read_csv(r"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release\tenThousandTrackMomentumData.csv")
    df = df[df.TrueID != -1]
    df = df[df.event < 2000]
    mom = -7.5 / df.a
    df = df[((mom < 3000) & (mom > -3000))]
    fig, ax = plt.subplots()
    delta_momentum = np.abs(7.5/df.a) - df.TrueMom
    ax.hist(delta_momentum, bins=100)
    ax.set_xlabel("Reco - True Momentum MeV/c")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig("RecoTrueMomComp.png")

def track_num_comparison():
    df_true_num = pd.read_csv(r'\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release\tenThousandTrackNum.csv')
    df_metaData = pd.read_csv(r"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release\tenThousandTrackMetaData.csv")
    df_momentumData = pd.read_csv(r"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release\tenThousandTrackMomentumData.csv")
    diff = (df_true_num.set_index('Event').Size - df_metaData.predictedTrackNum).dropna()
    non_dup_num = df_momentumData.drop_duplicates(['TrueID', 'event']).groupby('event').size()
    non_dup_num.append(df2, ignore_index=True)
    eff = df_true_num.set_index('Event').Size - df_momentumData.drop_duplicates(['TrueID', 'event']).groupby('event').size()
    print(f"{diff.abs().mean()}")
    print(f"{df_true_num.set_index('Event').Size.mean()}")
    print(f"{df_metaData.predictedTrackNum.mean()}")
    fig, ax = plt.subplots()
    ax.hist(diff, bins=np.arange(-10,10))
    ax.set_xlabel("Reco - True Track Number")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig("RecoTrueTrackNumComp.png")

def trackNumData():
    df_dict = {'Event':list(range(0,2000)), 'Size':[]}
    for i in df_dict['Event']:
        df_dict['Size'].append(
            len(pd.read_csv(rf"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\simulation_data\tenThousandTracks\event{i}TrueStats.csv")))
    df = pd.DataFrame(df_dict)
    df.to_csv(r'\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release\tenThousandTrackNum.csv', index=False)


def time_dist():
    df = pd.read_csv(r"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release\tenThousandTrackMetaData.csv")
    fig,ax = plt.subplots()
    bins = np.logspace(0, np.log10(max(df.time)), 20)
    ax.hist(df.time, bins=bins, density=True)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency Density")
    fig.tight_layout()
    fig.savefig("Time.png")

### momentum against hits
def mom_hits():
    df = pd.read_csv(r"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release\tenThousandTrackMomentumData.csv")
    df = df[df.TrueID != -1]
    fig, ax = plt.subplots()
    p_mom = np.abs(7.5/df.a)
    df['p_mom'] = p_mom
    ax.plot(p_mom, df.numPoints, 'x')
    p_mom_g = df.groupby(pd.cut(df.numPoints, np.linspace(5, 150, 15))).p_mom.mean()
    p_mom_g_std = df.groupby(pd.cut(df.numPoints, np.linspace(5, 150, 15))).p_mom.sem()
    fig, ax = plt.subplots()
    ax.errorbar([p_mom_g.index[i].mid for i in range (0, len(p_mom_g.index))], p_mom_g, yerr=p_mom_g_std, label="Average Momentum")
    ax.set_xlabel('Track Number of Points')
    ax.set_ylabel('Average Momentum MeV/c')
    ax.legend()
    fig.tight_layout()
    fig.savefig("MomentumNumPoints.png")

## momentum uncertainty against momentum and hits

def mom_corr():
    df = pd.read_csv(r"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release\tenThousandTrackMomentumData.csv")
    df = df[df.TrueID != -1]
    fig, ax = plt.subplots()
    p_mom = np.abs(7.5/df.a)
    p_mom = np.abs(7.5 / df.a)
    ax.errorbar(df.TrueMom, p_mom, '.')
    corr = stats.pearsonr(df.TrueMom, p_mom)
    print(corr)
    ax.set_ylabel("Projected Momentum")
    ax.set_xlabel("True momentum")
    fig.tight_layout()
    fig.savefig("MomentumCorr.png")

# start posiition against momentum plot.
def mom_start_pos():
    df_pos = pd.read_csv(r"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release\tenThousandTrack.csv")
    df_pos = df_pos.drop_duplicates(['event','track']).set_index(['event', 'track'])
    df = pd.read_csv(
        r"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release\tenThousandTrackMomentumData.csv")
    mom = -7.5 / df.a
    df = df[((mom < 3000) & (mom > -3000))]
    df = df[df.TrueID != -1]
    df['x_start'] = [df_pos.loc[(int(df.iloc[i]['event']), int(df.iloc[i]['track']))].x for i in range(len(df.TrueID))]
    df['y_start'] = [df_pos.loc[(int(df.iloc[i]['event']), int(df.iloc[i]['track']))].y for i in range(len(df.TrueID))]

    p_mom = np.abs(7.5 / df.a)

    fig, ax = plt.subplots()
    ax.plot(df.x_start, p_mom, '.')
    ax.set_xlabel("X Position")
    ax.set_ylabel("Projected Momentum MeV/c")
    fig.tight_layout()
    fig.savefig("StartPosMom.png")
    fig, ax = plt.subplots()
    ax.plot(df.y_start, p_mom, '.')
    ax.set_xlabel("Y Position")
    ax.set_ylabel("Projected Momentum Mev/c")

    heatmap, xedges, yedges = np.histogram2d(df.x_start, p_mom, bins=30)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, origin='lower')
    plt.axis('square')
    plt.show()
    fig.tight_layout()
    fig.savefig("StartPosMomHeatMap.png")



def after_many_tracks_pos_plot(events = [1,2,3,4,5]):
    df = pd.read_csv(r"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release\tenThousandTrack.csv")
    df_event_grouped = df.groupby('event')
    for e in df_event_grouped.groups.keys():
        #df_cluster = pd.read_csv(rf"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\simulation_data\tenThousandTracks\event{e}.csv")
        df_truth = pd.read_csv(rf"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\simulation_data\tenThousandTracks\event{e}TrueHits.csv")
        if not e in events:
            continue
        df_g = df_event_grouped.get_group(e)
        df_grouped = df_g.groupby('track')
        fig, ax = plt.subplots()
        #ax.plot(df_cluster.x, df_cluster.y,'x', label=f'cluster')
        ax.plot(df_truth.X*10, df_truth.Y*10, 'x', label=f'truth')
        for g in df_grouped.groups.keys():
            df_g = df_grouped.get_group(g)
            ax.plot(df_g.x, df_g.y, 'x', label=f'track {g}')
        plt.legend()

def after_CPP(filename, trackwise=False, track=None):
    df = pd.read_csv(rf"{location}\{filename}")
    fig, ax = plt.subplots()
    if trackwise:
        df_grouped = df.groupby('track')
        for g in df_grouped.groups.keys():
            if track is None or track == g:
                df_g = df_grouped.get_group(g)
                ax.plot(df_g.x, df_g.y, 'x', label=f'track {g}')
        plt.legend()
    else:
        plt.plot(df.x, df.y, 'x')

def sullyCompare_mom_diff():
    pass

def sullyCompare_mom_dist():
    df = pd.read_csv(
        r"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release\tenThousandTrackMomentumData.csv")

    fig, ax = plt.subplots()
    mag_mom = 7.5 / np.abs(df.a)
    mag_mom = mag_mom[mag_mom < 3000]
    data_hist = ax.hist(mag_mom, bins=100, label='CKF')
    hist_x, hist_y = bins_to_points(data_hist)
    parameters = [150, 200, 0, 600, 200, 1000]
    x = np.linspace(0, 2000, 100000)
    parameters, covariance = fitDoubleGauss(hist_x, hist_y, p0=parameters)
    ax.plot(x, DoubleGauss(x, *parameters), label=f'Gaussian Fit: CKF\n'
                                                  f'Main Peak $\mu$ = {parameters[5]:.2f} $\pm$ {covariance[5, 5] ** 1 / 2:.2f}\n'
                                                  f'$\sigma$ = {parameters[4]:.2f} $\pm$ {covariance[4, 4] ** 1 / 2:.2f}')

    df_h = pd.read_csv(
        r"\\wsl.localhost\Ubuntu\home\tt1020\CPPKalmanFilter\cplusdemo\cmake-build-release\hough_recon_mom.csv")
    data_hist = ax.hist(df_h.recon_mom, bins=hist_x, label='Hough')
    hist_x, hist_y = bins_to_points(data_hist)
    parameters = [150, 200, 0, 600, 200, 1000]
    x = np.linspace(0, 2000, 100000)
    parameters, covariance = fitDoubleGauss(hist_x, hist_y, p0=parameters)
    ax.plot(x, DoubleGauss(x, *parameters), label=f'Gaussian Fit: Hough\n'
                                                  f'Main Peak $\mu$ = {parameters[5]:.2f} $\pm$ {covariance[5, 5] ** 1 / 2:.2f}\n'
                                                  f'$\sigma$ = {parameters[4]:.2f} $\pm$ {covariance[4, 4] ** 1 / 2:.2f}')
    # fitDoubleGauss()
    ax.set_xlabel("Absolute Momentum MeV/c")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig("AbsoluteMomentumDistributionHoughCKF.png")

def sullyCompare_trackNumDiff():
    pass


momentum_dist()

#for_CPP_ten_track("tenTracks.csv")
#after_CPP("sullyEG.csv")
# i=0
# while True:
#     try:
#         after_CPP(f"firstSavedTrack_{i}.csv")
#     except:
#         break
#     i+=1
#after_CPP('sullyEGBestChildTrack.csv', trackwise=True)