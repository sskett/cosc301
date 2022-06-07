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

from d10_code import f48_pytorch_analysis as mcm


def train_route_finder_models(df, dims, svm=True, rf=True, mlp=True, mcnn=True):
    if svm or rf or mlp:
        x_train, x_test, y_train, y_test = train_test_split(df['grids'], df['route'], test_size=0.2,
                                                        shuffle=False, random_state=1)
    if svm:
        print('performing SVM classification of route data')
        do_svm_analysis(x_train.tolist(), y_train, x_test.tolist(), y_test)
    if rf:
        print('performing RF classification of route data')
        do_rf_analysis(x_train.tolist(), y_train, x_test.tolist(), y_test)
    if mlp:
        print('performing MLP classification of route data')
        do_mlp_analysis(x_train.tolist(), y_train, x_test.tolist(), y_test)
    if mcnn:
        print('performing MCNN classification of route data')
        learning_rate = 0.5
        hidden_units = 256
        epochs = 2000
        mcm.do_mcnn_analysis(df, dims, learning_rate, hidden_units, epochs, scaled=False)


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
    model = RandomForestClassifier(n_jobs=-1, n_estimators=1000, max_depth=10)
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