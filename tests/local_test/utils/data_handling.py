import pandas as pd
from sklearn.model_selection import KFold
import numpy as np


def load_data(data_path):
    X_train = pd.read_csv(data_path + r'\X_train.csv')
    X_test = pd.read_csv(data_path + r'\X_test.csv')
    y_train = pd.read_csv(data_path + r'\y_train.csv')
    y_test = pd.read_csv(data_path + r'\y_test.csv')

    data_dict = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

    return data_dict


def create_cv_data(X_train, y_train, cv_config):
    data = kfolds_generator(X_train, y_train, **cv_config)

    return data


def kfolds_generator(X, y, n, seed):
    data_tasks = []
    kf = KFold(n_splits=n, random_state=seed, shuffle=True)

    for train_index, test_index in kf.split(X):
        data_dict = {'X_train': X.iloc[train_index, :], 'X_test': X.iloc[test_index, :],
                     'y_train': y.iloc[train_index, :], 'y_test': y.iloc[test_index, :]}

        data_tasks.append(data_dict)

    return data_tasks


def simple_cv_generator(X, y, ratio, seed):
    data_tasks = []

    np.random.seed(seed)
    msk = np.random.rand(len(X)) < ratio

    data_dict = {'X_train': X[msk], 'X_test': X[~msk], 'y_train': y[msk], 'y_test': y[~msk]}

    data_tasks.append(data_dict)

    return data_tasks
