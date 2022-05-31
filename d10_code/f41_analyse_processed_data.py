import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ray

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from d10_code import f42_pytorch_analysis as mcm


def analyse_routes_data(routes_df, n_procs):
    x, y = get_limits(routes_df, n_procs)
    #plot_route_data(routes_df, x, y)
    plot_route_probs(routes_df)

    dims = (int(abs(x[0]) + abs(x[1])), int(abs(y[0]) + abs(y[1])))
    routes_df['grids'] = get_position_grid(routes_df, dims)

    #x_train, x_test, y_train, y_test = train_test_split(routes_df['grids'], routes_df['route'], test_size=0.2, shuffle=False, random_state=1)

    print('performing SVM classification of route data')
    #do_svm_analysis(x_train.tolist(), y_train, x_test.tolist(), y_test)
    print('performing RF classification of route data')
    #do_rf_analysis(x_train.tolist(), y_train, x_test.tolist(), y_test)
    print('performing MLP classification of route data')
    #do_mlp_analysis(x_train.tolist(), y_train, x_test.tolist(), y_test)
    print('performing MCNN classification of route data')
    learning_rate = 0.5
    hidden_units = 256
    epochs = 2000
    mcm.do_mcnn_analysis(routes_df, dims, learning_rate, hidden_units, epochs, scaled=False)


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


def get_position_grid(df, dims):
    grids = []
    for idx, row in df.iterrows():
        grid = np.zeros(dims[0] * dims[1]).reshape(dims[0], dims[1])
        for position in row['pos']:
            grid[int(position[0]), int(position[1])] = position[2]
        grids.append(grid.reshape(dims[0] * dims[1]))
    return grids


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


def do_rf_analysis(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    predicted = model.predict(x_test)
    print(metrics.accuracy_score(predicted, y_test))

    print(
        f'Classification report for classifier {model}:\n'
        f'{metrics.classification_report(y_test, predicted)}\n'
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle('Confusion Matrix')
    print(f'Confusion Matrix:\n{disp.confusion_matrix}')
    plt.show()


def do_mlp_analysis(x_train, y_train, x_test, y_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    clf = MLPClassifier(solver='lbfgs', random_state=1, max_iter=1000)
    clf.fit(x_train_scaled, y_train)

    predicted = clf.predict(x_test_scaled)

    print(
        f'Classification report for classifier {clf}:\n'
        f'{metrics.classification_report(y_test, predicted)}\n'
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle('Confusion Matrix')
    print(f'Confusion Matrix:\n{disp.confusion_matrix}')
    plt.show()


def plot_route_probs(df):
    plt.figure(figsize=(20, 10))
    plt.bar(df['route'].value_counts().index,
            df['route'].value_counts().values / df['route'].value_counts().sum(), width=0.75)

