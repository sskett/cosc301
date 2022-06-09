import matplotlib.pyplot as plt
import numpy as np
import ray

from d10_code import f46_clustering as f46
from d10_code import f47_route_learning as f47
from d10_code import f64_cluster_order_density_plots as f64


def analyse_processed_data(play_df, frame_df, routes_df, n_procs):
    x, y = get_limits(routes_df, n_procs)
    # Plot an overlay of all routes in dataset (only use for constrained data!)
    # plot_route_data(routes_df, x, y)

    # Plot distribution of route types
    plot_route_probs(routes_df)

    # Clustering
    play_df = f46.perform_clustering(play_df)
    f64.visualise_cluster_order_parameters(play_df, 'clusters_k')

    # Route identification modelling
    dims = (int(abs(x[0]) + abs(x[1])), int(abs(y[0]) + abs(y[1])))
    routes_df['grids'] = get_position_grid(routes_df, dims)
    f47.train_route_finder_models(routes_df, dims)


def plot_route_data(df, x_lims, y_lims):
    for idx, row in df.iterrows():
        plt.scatter(row['pos'][:, 0], row['pos'][:, 1], c=row['pos'][:, 2])
    plt.xlim(x_lims[0] - 1, x_lims[1] + 1)
    plt.ylim(y_lims[0] - 1, y_lims[1] + 1)
    plt.show()


def plot_route_probs(df):
    # Plot a normalised bar chart of route distributions
    plt.figure(figsize=(20, 10))
    plt.bar(df['route'].value_counts().index,
            df['route'].value_counts().values / df['route'].value_counts().sum(), width=0.75)


def get_position_grid(df, dims):
    grids = []
    for idx, row in df.iterrows():
        grid = np.zeros(dims[0] * dims[1]).reshape(dims[0], dims[1])
        for position in row['pos']:
            grid[int(position[0]), int(position[1])] = position[2]
        grids.append(grid.reshape(dims[0] * dims[1]))
    return grids


@ray.remote
def find_min_max_from_dataframe(df):
    x = [0, 0]
    y = [0, 0]

    for idx, row in df.iterrows():
        if row['pos'][:, 0].min() < x[0]: x[0] = row['pos'][:, 0].min()
        if row['pos'][:, 0].max() > x[1]: x[1] = row['pos'][:, 0].max()
        if row['pos'][:, 1].min() < y[0]: y[0] = row['pos'][:, 1].min()
        if row['pos'][:, 1].max() > y[1]: y[1] = row['pos'][:, 1].max()

    x = [np.floor(x[0]), np.ceil(x[1])]
    y[0] = -1 * np.ceil(max(abs(y[0]), abs(y[1])))
    y[1] = abs(y[0])

    return x, y


def find_min_max_from_arrays(x_list, y_list):
    x = [0, 0]
    x[0] = x_list[:, 0].min()
    x[1] = x_list[:, 1].max()

    y = [0, 0]
    y[1] = np.abs(y_list.max())
    y[0] = -y[1]

    return x, y


def get_limits(df, n_procs):
    # Finds the min/max x,y coordinates over the set of player positions
    n_rows = len(df)
    max_rows = int(len(df[:n_rows]))
    row_sets = []
    set_pos = 0
    step = int(max_rows / (n_procs - 1))

    while set_pos < max_rows:
        start = 0 if set_pos == 0 else set_pos + 1
        set_pos = set_pos + step
        end = set_pos if set_pos < max_rows else max_rows
        row_sets.append((start, end))

    x = []
    y = []

    futures = [find_min_max_from_dataframe.remote(df[idx[0]:idx[1]]) for idx in row_sets]
    results = ray.get(futures)

    for i in range(0, len(results)):
        x.append(results[i][0])
        y.append(results[i][1])

    x, y = find_min_max_from_arrays(np.array(x), np.array(y))
    return x, y

