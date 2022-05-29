import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyse_routes_data(routes_df):
    x, y = get_limits(routes_df)
    plot_route_data(routes_df)


def get_limits(df):
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


def plot_route_data(df, x_lims, y_lims):

    for idx, row in df.iterrows():
        plt.scatter(row['pos'][:, 0], row['pos'][:, 1], c=row['pos'][:, 2])
    plt.xlim(x_lims[0] - 1, x_lims[1] + 1)
    plt.ylim(y_lims[0] - 1, y_lims[1] + 1)
    plt.show()
