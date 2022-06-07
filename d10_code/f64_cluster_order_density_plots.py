import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def draw_prob_density(x_series, y_series, x_bins, y_bins, ax, xlab=None, ylab=None, cbar=True):
    dims = {'x': x_series.max() * 1.00001, 'y': y_series.max() * 1.00001}

    bins = np.zeros(x_bins * y_bins).reshape(y_bins, x_bins)
    rows = min(len(y_series), len(x_series))

    for row in range(rows):
        i = int(x_series.iloc[row] / dims['x'] * x_bins)
        j = int(y_series.iloc[row] / dims['y'] * y_bins)
        bins[j][i] += 1

    smoothing_grid = np.zeros(x_bins * y_bins).reshape(y_bins, x_bins)
    for i in range(x_bins):
        for j in range(y_bins):
            sum = 0
            count = 0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if 0 <= i + x < x_bins and 0 <= j + y < y_bins:
                        sum += bins[j + y][i + x]
                        count += 1
            smoothing_grid[j][i] = sum / count

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if cbar:
        c = ax.pcolor(smoothing_grid, cmap='jet')
        plt.colorbar(c, ax=ax)
    else:
        ax.pcolor(smoothing_grid, cmap='jet')


def plot_cluster_distribution(df, cluster_col):
    values = df[cluster_col].value_counts()
    totals = df[cluster_col].value_counts().sum()
    (values / totals).plot.bar(rot=0, title='Distribution of clusters', xlabel='Cluster ID', ylabel='Density')
    plt.savefig('./d30_results/cluster_distribution.png')


def find_cluster_route_differences(df_clustered, clusters):
    routes = ['HITCH', 'OUT', 'FLAT', 'CROSS', 'GO', 'SLANT', 'SCREEN', 'CORNER',
              'IN', 'ANGLE', 'POST', 'WHEEL']
    p_target = 0.02

    route_sig = {'HITCH': 0, 'OUT': 0, 'FLAT': 0, 'CROSS': 0, 'GO': 0, 'SLANT': 0, 'SCREEN': 0, 'CORNER': 0,
                 'IN': 0, 'ANGLE': 0, 'POST': 0, 'WHEEL': 0}
    grid_dim = clusters + 1
    differences = np.zeros(grid_dim ** 2).reshape(grid_dim, grid_dim)

    for i in range(clusters + 1):
        for j in range(clusters + 1):
            if not j <= i:
                for route in routes:
                    c1 = df_clustered.loc[df_clustered['cluster'] == i][route]
                    c2 = df_clustered.loc[df_clustered['cluster'] == j][route]
                    p_value = ttest_ind(c1, c2)[1]
                    if p_value <= p_target:
                        differences[i][j] += 1
                        route_sig[route] += 1
                        print(
                            f'P-Value for difference between clusters {i} and {j} for the {route} route: {p_value:.4f}')

    x = route_sig.keys()
    y = list(route_sig.values())
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.bar(x, (y / ((clusters ** 2 - clusters) / 2)))
    plt.savefig('./d30_results/route_difference_distribution.png')

    differences[:, :]

    df_clustered.groupby('cluster')[routes].mean().plot.box()
    plt.savefig('./d30_results/route_difference_boxplot.png')


def visualise_cluster_order_parameters(df_clustered, cluster_col):
    fig_height = 3
    clusters = df_clustered[cluster_col].max()

    # TODO: Convert plotting to functions
    # Plot P vs M and V
    # Draw aggregate
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(3 * fig_height, fig_height),
                            gridspec_kw={'width_ratios': [1, 2]})
    draw_prob_density(df_clustered['offense_m_play'], df_clustered['offense_p_play'], 50, 50, axs[0], xlab='m group',
                      ylab='p group', cbar=False)
    draw_prob_density(df_clustered['offense_v_play'], df_clustered['offense_p_play'], 60, 50, axs[1], xlab='v group')
    fig.suptitle(f'FIGURE 1: P Group vs M_Group and V_Group')
    # Draw by cluster
    for cluster in range(clusters + 1):
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(3 * fig_height, fig_height),
                                gridspec_kw={'width_ratios': [1, 2]})
        cluster_df = df_clustered[df_clustered[cluster_col] == cluster]
        draw_prob_density(cluster_df['offense_m_play'], cluster_df['offense_p_play'], 50, 50, axs[0], xlab='m group',
                          ylab='p group', cbar=False)
        draw_prob_density(cluster_df['offense_v_play'], cluster_df['offense_p_play'], 60, 50, axs[1], xlab='v group')
        fig.suptitle(f'FIGURE 1: P Group vs M_Group and V_Group for Cluster {cluster}')

    # Plot V vs H and A
    # Draw aggregate
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(3 * fig_height, fig_height),
                            gridspec_kw={'width_ratios': [1, 2]})
    draw_prob_density(df_clustered['offense_h_play'], df_clustered['offense_v_play'], 140, 60, axs[0], xlab='h group',
                      ylab='v group', cbar=False)
    draw_prob_density(df_clustered['offense_a_play'], df_clustered['offense_v_play'], 60, 60, axs[1], xlab='a group')
    fig.suptitle(f'FIGURE 1: P Group vs M_Group and V_Group')

    # Draw clusters
    for cluster in range(clusters + 1):
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(3 * fig_height, fig_height),
                                gridspec_kw={'width_ratios': [1, 2]})
        cluster_df = df_clustered[df_clustered[cluster_col] == cluster]
        draw_prob_density(cluster_df['offense_h_play'], cluster_df['offense_v_play'], 140, 60, axs[0], xlab='h group',
                          ylab='v group', cbar=False)
        draw_prob_density(cluster_df['offense_a_play'], cluster_df['offense_v_play'], 60, 60, axs[1], xlab='a group')
        fig.suptitle(f'FIGURE 1: P Group vs M_Group and V_Group for Cluster {cluster}')

