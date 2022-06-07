import pandas as pd
import pandas_profiling as pp
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as shc
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn import metrics


from d10_code import f11_import_processed_data as dfi
from d10_code import f21_filter_processed_data as dff
from d10_code import f31_clean_processed_data as dfc
from d10_code import f41_analyse_processed_data as dfa
from d10_code import f51_transform_processed_data as dft
from d10_code import f63_state_density_plots as st_vis


def string_to_vector(s):
    try:
        s = s.split('[')[1].split(']')[0]
        x = float(s.split()[0])
        y = float(s.split()[1])
        return np.array([x, y])
    except AttributeError:
        return None


def prepare_play_data(play_df):
    df = play_df[['gameId', 'playId', 'offense_h_presnap',
                  'offense_h_to_throw', 'offense_h_to_arrived',
                  'offense_p_presnap', 'offense_p_to_throw', 'offense_p_to_arrived',
                  'offense_m_presnap',
                  'offense_m_to_throw', 'offense_m_to_arrived', 'offense_v_presnap', 'offense_v_to_throw',
                  'offense_v_to_arrived',
                  'offense_a_presnap', 'offense_a_to_throw', 'offense_a_to_arrived',
                  'HITCH', 'OUT', 'FLAT', 'CROSS', 'GO', 'SLANT', 'SCREEN', 'CORNER',
                  'IN', 'ANGLE', 'POST', 'WHEEL', 'num_routes']].dropna().copy()

    df_scaled = normalize(df)
    return pd.DataFrame(df_scaled, columns=df.columns).drop(['gameId', 'playId'], axis=1)


def find_optimal_k(df_scaled, max_clusters):
    range_clusters = []
    for i in range(2, max_clusters + 1):
        range_clusters.append(i)

    silhouette_avg = []
    distortions = []

    for i in range_clusters:
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=1
        )
        km.fit(df_scaled)
        cluster_labels = km.labels_

        silhouette_avg.append(silhouette_score(df_scaled, cluster_labels))
        distortions.append(km.inertia_)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
    axs[0].plot(range_clusters, silhouette_avg, marker='o')
    # axs[0].set_xlabel('Distance Threshold (x10e-6)')
    axs[0].set_ylabel('Silhouette Score')
    axs[1].plot(range_clusters, distortions, marker='o')
    axs[1].set_xlabel('Values of K')
    axs[1].set_ylabel('Distortion')
    fig.suptitle('Silhouette and Elbow analysis for Optimal Clusters')
    plt.show()


def find_k_means(df_scaled, k):
    km = KMeans(
        n_clusters=k, init='random',
        n_init=3, max_iter=300,
        tol=1e-04, random_state=1
    )

    return km.fit_predict(df_scaled)


def find_optimal_agg_clusters(df_scaled, max_clusters):
    # Silhouette testing on n_clusters
    range_clusters = []
    n_samples = df_scaled.shape[0]
    [range_clusters.append(x) for x in range(2, max_clusters + 1)]
    silhouette_avg = []

    for i in range_clusters:
        model = AgglomerativeClustering(linkage='ward',
                                        affinity='euclidean',
                                        distance_threshold=None,
                                        n_clusters=i)
        model.fit(df_scaled)
        cluster_labels = model.labels_
        silhouette_avg.append(silhouette_score(df_scaled, cluster_labels))

    fig, ax = plt.subplots()
    ax.plot(range_clusters, silhouette_avg, marker='o')
    ax.set_xlabel('Distance Threshold')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette analysis for Optimal Clusters')
    plt.show()


def find_optimal_agg_threshold(df_scaled):
    # Silhouette testing on limit
    range_limit = [0.0000001, 0.0000003, 0.0000005, 0.0000007, 0.0000009, 0.0000011, 0.0000013]
    n_samples = df_scaled.shape[0]

    silhouette_avg = []
    clusters_found = []

    for i in range_limit:
        model = AgglomerativeClustering(linkage='ward',
                                        affinity='euclidean',
                                        distance_threshold=i,
                                        n_clusters=None)
        model.fit(df_scaled)
        cluster_labels = model.labels_
        clusters_found.append(np.unique(model.labels_).size)
        silhouette_avg.append(silhouette_score(df_scaled, cluster_labels))

    fig, axs = plt.subplots(2, 1, sharex='true', figsize=(15, 10))

    axs[0].plot(np.array(range_limit) * 10 ** 6, silhouette_avg, marker='o')
    # axs[0].set_xlabel('Distance Threshold (x10e-6)')
    axs[0].set_ylabel('Silhouette Score')

    axs[1].plot(np.array(range_limit) * 10 ** 6, clusters_found, marker='o')
    axs[1].set_xlabel('Distance Threshold (x10e-6)')
    axs[1].set_ylabel('Number of clusters found')
    fig.suptitle('Silhouette analysis for Optimal Clusters')

    plt.show()


def find_agg_clusters(df_scaled, threshold):
    model = AgglomerativeClustering(linkage='ward',
                                    affinity='euclidean',
                                    distance_threshold=threshold,
                                    n_clusters=None)
    model.fit(df_scaled)

    return model.labels_


def plot_relationship(df, xcol, ycol, labels, filename):
        fig = plt.figure(figsize=(6, 6))
        grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
        main_ax = fig.add_subplot(grid[:-1, 1:])
        y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
        x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

        main_ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            left=True,
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off
        x = df[xcol]
        y = df[ycol]
        main_ax.scatter(x, y, c=labels)

        x_hist.hist(x, 40, histtype='stepfilled', orientation='vertical', color='gray')
        x_hist.invert_yaxis()
        x_hist.set_xlabel(xcol)

        y_hist.hist(y, 40, histtype='stepfilled', orientation='horizontal', color='gray')
        y_hist.invert_xaxis()
        y_hist.set_ylabel(ycol)
        plt.savefig(f'{filename}{xcol}-{ycol}.png')


def plot_feature_pairs(df, cols, cluster_col):
    for subset in combinations(cols, 2):
        plot_relationship(df, subset[0], subset[1], df[cluster_col], './d30_results/fig_rel_plot_')


# Import data
def perform_clustering():
    data_directory = './d20_intermediate_files'
    print('import plays data')
    play_df = dfi.import_processed_play_data(data_directory + '/play_results.csv')
    print('import frames data')
    frame_df = dfi.import_processed_frame_data(data_directory + '/frame_results.csv')
    print('import tracking data')
    tracking_df = dfi.import_processed_tracking_data(data_directory + '/tracking_results.csv')

    tracking_df['pos'] = tracking_df['pos'].apply(string_to_vector)
    tracking_df['o_vec'] = tracking_df['o_vec'].apply(string_to_vector)
    tracking_df['dir_vec'] = tracking_df['dir_vec'].apply(string_to_vector)
    tracking_df['r_vec'] = tracking_df['r_vec'].apply(string_to_vector)

    # Prepare data
    play_df_scaled = prepare_play_data(play_df)

    # K-Means
    max_clusters = 20
    # TODO: Remove hardcoding of k_opt, either automate extraction from find_optimal_k or add as CLI parameter?
    find_optimal_k(play_df_scaled, max_clusters)
    k_opt = 6
    play_df['clusters_k'] = find_k_means(play_df_scaled, k=k_opt)

    # Agglomerative Clustering
    find_optimal_agg_clusters(play_df_scaled, max_clusters)
    find_optimal_agg_threshold(play_df_scaled)
    threshold = 1.1 * 10**-6
    play_df['clusters_agg'] = find_agg_clusters(play_df_scaled, threshold)

    # Plot important cluster relationships
    # TODO: Implement Random Forest training to quantitate the most important features, as per clustering.ipynb
    features = play_df.drop(['gameId', 'playId', 'clusters_agg', 'clusters_kmeans'], axis=1).columns.tolist()
    plot_feature_pairs(play_df, features, 'clusters_k')

    return play_df

