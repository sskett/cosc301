import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


def analyse_routes_data(routes_df):
    x, y = get_limits(routes_df)
    plot_route_data(routes_df, x, y)
    plot_route_probs(routes_df)

    dims = (int(abs(x[0]) + abs(x[1])), int(abs(y[0]) + abs(y[1])))
    routes_df['grids'] = get_position_grid(routes_df, dims)

    x_train, x_test, y_train, y_test = train_test_split(routes_df['grids'], routes_df['route'], test_size=0.2, shuffle=False,
                                                        random_state=1)

    do_svm_analysis(x_train.tolist(), y_train, x_test.tolist(), y_test)
    do_MLP_analysis(routes_df)


def get_position_grid(df, dims):
    grids = []
    for idx, row in df.iterrows():
        grid = np.zeros(dims[0] * dims[1]).reshape(dims[0], dims[1])
        for position in row['pos']:
            grid[int(position[0]), int(position[1])] = position[2]
        grids.append(grid.reshape(dims[0] * dims[1]))
    return grids


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


def do_svm_analysis(x_train, y_train, x_test, y_test):

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.005, kernel='rbf')

    # Learn on the train subset
    clf.fit(x_train, y_train)

    # Predict the value of the test subset
    predicted = clf.predict(x_test)

    print(
        f'Classification report for classifier {clf}:\n'
        f'{metrics.classification_report(y_test, predicted)}\n'
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle('Confusion Matrix')
    print(f'Confusion Matrix:\n{disp.confusion_matrix}')
    plt.show()


def do_MLP_analysis(df):



def plot_route_probs(df):
    plt.figure(figsize=(20, 10))
    plt.bar(df['route'].value_counts().index,
            df['route'].value_counts().values / df['route'].value_counts().sum(), width=0.75)