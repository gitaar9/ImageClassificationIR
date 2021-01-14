import argparse
import os
import pprint

import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from skopt import BayesSearchCV


def load_dataset(dataset_path, normalize=False, name=""):
    X_train = np.load(os.path.join(dataset_path, f"{name}_train_features.npy"))
    y_train = np.load(os.path.join(dataset_path, f"{name}_train_labels.npy"))
    X_test = np.load(os.path.join(dataset_path, f"{name}_test_features.npy"))
    y_test = np.load(os.path.join(dataset_path, f"{name}_test_labels.npy"))

    if normalize:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    return X_train, y_train, X_test, y_test


def svc_grid_search(X, y, nfolds=3, n_jobs=5, cv=None, verbose=False):
    param_grid = {
        'C': list(np.logspace(-4, 3, 8)),
        'gamma': list(np.logspace(-5, 2, 8)),
        'degree': [0, 1, 2, 3],
        'kernel': ["rbf", "poly"],
        'max_iter': [10000]
    }
    if verbose:
        print("Performing grid search with the following parameters:\n")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(param_grid)

    if cv is None:
        cv = StratifiedShuffleSplit(n_splits=nfolds, test_size=0.2, random_state=42)
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=cv, n_jobs=n_jobs, verbose=1)
    grid_search.fit(X, y)

    if verbose:
        for params, result in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
            print(f"{params}: {result}")

    print("\nThe best accuracy was {:.2f}% using: {}".format(grid_search.best_score_ * 100.0, grid_search.best_params_))

    return grid_search.best_params_, grid_search.best_score_


def svm_bayes_search(X, y, nfolds):
    opt = BayesSearchCV(
        svm.SVC(),
        {
            'C': (1e-6, 1e+6, 'log-uniform'),
            'gamma': (1e-6, 1e+1, 'log-uniform'),
            'degree': (1, 8),  # integer valued parameter
            'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
        },
        n_iter=32,
        cv=nfolds,
        n_jobs=5
    )

    opt.fit(X, y)
    print("val. score: %s" % opt.best_score_)  # 0.458
    print(dir(opt))


def main(dataset_path=None, dataset_name=None, n_jobs=10):
    if dataset_path is None:
        dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets/'
    if dataset_name is None:
        dataset_name = "keras_ResNet152_224px_pp1"

    # Load the data
    X_train, y_train, X_test, y_test = load_dataset(os.path.join(dataset_path, dataset_name), normalize=False)
    print(X_train.shape, X_test.shape)

    # Hacky way of using the train/test sets for optimization
    all_data = np.concatenate((X_train, X_test))
    all_labels = np.concatenate((y_train, y_test))
    cv = [(list(range(len(X_train))), list(map(lambda idx: idx + len(X_train), range(len(X_test)))))]

    # Find the optimal parameters
    best_params, best_score = svc_grid_search(all_data, all_labels, 3, n_jobs=n_jobs, cv=cv)

    # Save the results to the csv file
    with open(os.path.join(dataset_path, 'results.csv'), 'a') as result_file:
        result_file.write(f'"{dataset_name}",{round(best_score * 100.0, 2)},"{best_params}"\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help="The path to the fodler right above the dataset folder", default=None)
    parser.add_argument('--dataset_name', help="The name of the folder containing the dataset", default=None)
    parser.add_argument('--nworkers', type=int, default=10,
                        help='Number of workers for data loading  (0 to do it using main process) [Default : 10]')
    opt = parser.parse_args()

    main(dataset_path=opt.dataset_path, dataset_name=opt.dataset_name, n_jobs=opt.nworkers)
