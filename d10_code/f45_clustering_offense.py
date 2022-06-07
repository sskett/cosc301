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
from scipy.stats import ttest_ind

def string_to_vector(s):
    try:
        s = s.split('[')[1].split(']')[0]
        x = float(s.split()[0])
        y = float(s.split()[1])
        return np.array([x, y])
    except AttributeError:
        return None


def get_position_delta(row):
    return row.s / 10 * row.dir_vec


def get_relative_position(row):
    if row.frameId == 1:
        return np.array([0, 0])
    else:
        last_pos = row.shift(1).rel_pos
        return last_pos + row.pos_delta


def find_kmeans(df, max_clusters, output_location):
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
        km.fit(df)
        cluster_labels = km.labels_

        silhouette_avg.append(silhouette_score(df, cluster_labels))
        distortions.append(km.inertia_)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
    axs[0].plot(range_clusters, silhouette_avg, marker='o')
    # axs[0].set_xlabel('Distance Threshold (x10e-6)')
    axs[0].set_ylabel('Silhouette Score')
    axs[1].plot(range_clusters, distortions, marker='o')
    axs[1].set_xlabel('Values of K')
    axs[1].set_ylabel('Distortion')
    # fig.suptitle('Silhouette and Elbow analysis for Optimal Clusters')
    plt.savefig(f'{output_location}figure9.png')


def kmeans_clustering(df, k):
    km = KMeans(
        n_clusters=k, init='random',
        n_init=3, max_iter=300,
        tol=1e-04, random_state=1
    )

    return km.fit_predict(df)


def agg_clustering(df, threshold):
    model = AgglomerativeClustering(linkage='ward',
                                    affinity='euclidean',
                                    distance_threshold=threshold,
                                    n_clusters=None)
    model.fit(df)
    return model.labels_


def find_agg_clusters(df, max_clusters, output_location):
    # Silhouette testing on n_clusters
    range_clusters = []
    n_samples = df.shape[0]
    [range_clusters.append(x) for x in range(2, max_clusters + 1)]
    silhouette_avg = []

    for i in range_clusters:
        model = AgglomerativeClustering(linkage='ward',
                                        affinity='euclidean',
                                        distance_threshold=None,
                                        n_clusters=i)
        model.fit(df)
        cluster_labels = model.labels_
        silhouette_avg.append(silhouette_score(df, cluster_labels))

    fig, ax = plt.subplots()
    ax.plot(range_clusters, silhouette_avg, marker='o')
    ax.set_xlabel('Distance Threshold')
    ax.set_ylabel('Silhouette Score')
    # ax.set_title('Silhouette analysis for Optimal Clusters')
    plt.savefig(f'{output_folder}figure10.png')

    # Silhouette testing on limit
    range_limit = [0.0000001, 0.0000003, 0.0000005, 0.0000007, 0.0000009, 0.0000011, 0.0000013]
    n_samples = df.shape[0]

    silhouette_avg = []
    clusters_found = []

    for i in range_limit:
        model = AgglomerativeClustering(linkage='ward',
                                        affinity='euclidean',
                                        distance_threshold=i,
                                        n_clusters=None)
        model.fit(df)
        cluster_labels = model.labels_
        clusters_found.append(np.unique(model.labels_).size)
        silhouette_avg.append(silhouette_score(df, cluster_labels))

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

    axs[0].plot(np.array(range_limit) * 10 ** 6, silhouette_avg, marker='o')
    # axs[0].set_xlabel('Distance Threshold (x10e-6)')
    axs[0].set_ylabel('Silhouette Score')

    axs[1].plot(np.array(range_limit) * 10 ** 6, clusters_found, marker='o')
    axs[1].set_xlabel(r'Distance Threshold (x$10^{-6}$)')
    axs[1].set_ylabel('Number of clusters found')
    # fig.suptitle('Silhouette analysis for Optimal Clusters')

    plt.savefig(f'{output_folder}figure11.png')


def plot_cluster_distributions(df, k_clusters, agg_clusters, output_location):
    width = 0.4
    ax = (df[k_clusters].value_counts() / df[k_clusters].value_counts().sum()).plot.bar(color='blue',
                                                                                        width=width,
                                                                                        position=0,
                                                                                        label='K-Means')
    (df[agg_clusters].value_counts() / df[agg_clusters].value_counts().sum()).plot.bar(ax=ax, rot=0,
                                                                                       title='Distribution of clusters',
                                                                                       xlabel='Cluster ID',
                                                                                       ylabel='Density',
                                                                                       color='orange', width=width,
                                                                                       position=1,
                                                                                       label='Agglomerative Clustering')
    plt.legend()
    plt.savefig(f'{output_location}figure12.png')


def plot_relationship(df, xcol, ycol, labels):
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
    main_ax = fig.add_subplot(grid[:-1, 1:])
    y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

    main_ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        left=True,
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labelleft=False) # labels along the bottom edge are off
    x = df[xcol]
    y = df[ycol]
    main_ax.scatter(x, y, c=labels)

    x_hist.hist(x, 40, histtype='stepfilled', orientation='vertical', color='gray')
    x_hist.invert_yaxis()
    x_hist.set_xlabel(xcol)

    y_hist.hist(y, 40, histtype='stepfilled', orientation='horizontal', color='gray')
    y_hist.invert_xaxis()
    y_hist.set_ylabel(ycol)


pl_cols = ['gameId', 'playId', 'offense_h_play', 'offense_h_presnap', 'offense_h_to_throw', 'offense_h_to_arrived', 'offense_h_to_end', 'defense_h_play', 'defense_h_presnap', 'defense_h_to_throw', 'defense_h_to_arrived', 'defense_h_to_end', 'offense_p_play', 'offense_p_presnap', 'offense_p_to_throw', 'offense_p_to_arrived', 'offense_p_to_end', 'offense_m_play', 'offense_m_presnap', 'offense_m_to_throw', 'offense_m_to_arrived', 'offense_m_to_end', 'offense_v_play', 'offense_v_presnap', 'offense_v_to_throw', 'offense_v_to_arrived', 'offense_v_to_end', 'offense_a_play', 'offense_a_presnap', 'offense_a_to_throw', 'offense_a_to_arrived','offense_a_to_end', 'defense_p_play', 'defense_p_presnap','defense_p_to_throw', 'defense_p_to_arrived', 'defense_p_to_end','defense_m_play', 'defense_m_presnap', 'defense_m_to_throw','defense_m_to_arrived', 'defense_m_to_end', 'defense_v_play', 'defense_v_presnap', 'defense_v_to_throw', 'defense_v_to_arrived','defense_v_to_end', 'defense_a_play', 'defense_a_presnap','defense_a_to_throw', 'defense_a_to_arrived', 'defense_a_to_end','HITCH', 'OUT', 'FLAT', 'CROSS', 'GO', 'SLANT', 'SCREEN', 'CORNER', 'IN', 'ANGLE', 'POST', 'WHEEL']

play_df = pd.read_csv('d20_intermediate_files/play_results.csv', usecols=pl_cols)
play_df['num_routes'] = play_df[['HITCH', 'OUT', 'FLAT', 'CROSS', 'GO', 'SLANT', 'SCREEN', 'CORNER', 'IN', 'ANGLE', 'POST', 'WHEEL']].T.sum()
play_df.drop(play_df[play_df['num_routes'] == 0].index, inplace=True)
play_df.dropna(inplace=True)

fr_cols = ['gameId', 'playId', 'frameId', 'offense_p_group', 'defense_p_group', 'offense_m_group', 'defense_m_group', 'o_state', 'd_state', 'offense_v_group', 'defense_v_group', 'offense_a_group', 'defense_a_group', 'a_group_ratio']
frame_df = pd.read_csv('d20_intermediate_files/frame_results.csv', usecols=fr_cols)

tr_cols = ['time', 's', 'a', 'dis', 'event', 'nflId', 'displayName', 'jerseyNumber', 'position', 'frameId', 'team', 'gameId', 'playId', 'playDirection', 'route', 'pos', 'teamType', 'o_vec', 'dir_vec', 'r_vec']
tracking_df = pd.read_csv('d20_intermediate_files/tracking_results.csv', usecols=tr_cols)

tracking_df['pos'] = tracking_df['pos'].apply(string_to_vector)
tracking_df['o_vec'] = tracking_df['o_vec'].apply(string_to_vector)
tracking_df['dir_vec'] = tracking_df['dir_vec'].apply(string_to_vector)
tracking_df['r_vec'] = tracking_df['r_vec'].apply(string_to_vector)

fig_height = 3
output_folder = './d30_results/'

# Clustering of play data

# Reduce data to that relating to the offense only
df = play_df[['gameId', 'playId', 'offense_h_presnap',
       'offense_h_to_throw', 'offense_h_to_arrived',
       'offense_p_presnap', 'offense_p_to_throw', 'offense_p_to_arrived',
       'offense_m_presnap',
       'offense_m_to_throw', 'offense_m_to_arrived', 'offense_v_presnap', 'offense_v_to_throw',
       'offense_v_to_arrived',
       'offense_a_presnap', 'offense_a_to_throw', 'offense_a_to_arrived',
       'HITCH', 'OUT', 'FLAT', 'CROSS', 'GO', 'SLANT', 'SCREEN', 'CORNER',
       'IN', 'ANGLE', 'POST', 'WHEEL', 'num_routes']].dropna().copy()

# Normalise the data to remove effects of differing scales in various dimensions
df_scaled = normalize(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns).drop(['gameId', 'playId'], axis=1)

max_clusters = 20
# TODO: automate selection of optimal K and threshold (or, more likely, allow to pass as a cmd line parameter)
# Perform K-Means clustering
find_kmeans(df_scaled, max_clusters, output_folder)
k_opt = 6
df['clusters_kmeans'] = kmeans_clustering(df, k_opt)

# Perform Agglomerative Clustering
threshold = 1.1 * 10**-6
df['clusters_agg'] = agg_clustering(df_scaled, threshold)

# Plot clustering results
plot_cluster_distributions(df, 'clusters_kmeans', 'clusters_agg', output_folder)

# Plot cluster relationships
df_columns = df[['offense_a_to_arrived', 'offense_a_to_throw', 'offense_h_to_throw', 'offense_h_to_arrived']].columns.tolist()
figure_label = 14
for subset in combinations(df_columns, 2):
    plot_relationship(df, subset[0], subset[1], df['clusters_kmeans'])
    plt.savefig(f'{output_folder}figure{figure_label}.png')
    figure_label += 1

# Analyse routes within clusters

