import argparse
import os
import pprint

import numpy as np
from sklearn import svm
# from skopt import BayesSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from ir_image_classification.data_visualization.util import get_random_permutation


def load_dataset(dataset_path, normalize=False, name="", subset_size=None, nr_selected_feature_with_pca=None):
    X_train = np.load(os.path.join(dataset_path, f"{name}train_features.npy"))
    y_train = np.load(os.path.join(dataset_path, f"{name}train_labels.npy"))
    X_test = np.load(os.path.join(dataset_path, f"{name}test_features.npy"))
    y_test = np.load(os.path.join(dataset_path, f"{name}test_labels.npy"))

    if nr_selected_feature_with_pca:
        pca = PCA(n_components=nr_selected_feature_with_pca)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print(f'Cumulative explained variation for {nr_selected_feature_with_pca} '
              f'principal components: {np.sum(pca.explained_variance_ratio_)}')

    if normalize:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if subset_size:
        # Shuffle the dataset
        rndperm = get_random_permutation(X_train.shape[0])

        # Take only subset_size samples
        print("Sizes before subsampling:", X_train.shape, y_train.shape)
        X_train = X_train[rndperm][:subset_size].copy()
        y_train = y_train[rndperm][:subset_size].copy()
        print("Sizes after subsampling:", X_train.shape, y_train.shape)

    return X_train, y_train, X_test, y_test


def svc_grid_search(X, y, nfolds=3, n_jobs=5, cv=None, verbose=False):
    param_grid = {
        'C': list(np.logspace(-5, 3, 5)),
        'gamma': list(np.logspace(-5, 3, 5)),
        'degree': [0, 1, 2],
        'kernel': ["rbf", "poly", "linear"],
        'max_iter': [2000]
    }
    if verbose:
        print("Performing grid search with the following parameters:\n")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(param_grid)

    if cv is None:
        cv = StratifiedShuffleSplit(n_splits=nfolds, test_size=0.2, random_state=42)
    print(f"Using {n_jobs} workers.")
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=cv, n_jobs=n_jobs, verbose=1)
    grid_search.fit(X, y)

    if verbose:
        print("#" * 30, '\n')
        for params, result in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
            print(f"{params}: {result}")

    print("\nThe best accuracy was {:.2f}% using: {}".format(grid_search.best_score_ * 100.0, grid_search.best_params_))

    return grid_search.best_params_, grid_search.best_score_


# def svm_bayes_search(X, y, nfolds):
#     opt = BayesSearchCV(
#         svm.SVC(),
#         {
#             'C': (1e-6, 1e+6, 'log-uniform'),
#             'gamma': (1e-6, 1e+1, 'log-uniform'),
#             'degree': (1, 8),  # integer valued parameter
#             'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
#         },
#         n_iter=32,
#         cv=nfolds,
#         n_jobs=5
#     )
#
#     opt.fit(X, y)
#     print("val. score: %s" % opt.best_score_)  # 0.458
#     print(dir(opt))


def main(dataset_path=None, dataset_name=None, n_jobs=10):
    # Load the data
    print(f'Loading {dataset_name}')
    dataset_path = os.path.join(dataset_path, dataset_name)
    name = ""  # dataset_name

    X_train, y_train, X_test, y_test = load_dataset(dataset_path, normalize=False, name=name)

    print(X_train.shape, X_test.shape)

    # Hacky way of using the train/test sets for optimization
    all_data = np.concatenate((X_train, X_test))
    all_labels = np.concatenate((y_train, y_test))
    cv = [(list(range(len(X_train))), list(map(lambda idx: idx + len(X_train), range(len(X_test)))))]

    # Find the optimal parameters
    best_params, best_score = svc_grid_search(all_data, all_labels, 3, n_jobs=n_jobs, cv=cv, verbose=True)

    # Save the results to the csv file
    with open(os.path.join(dataset_path, 'results.csv'), 'a') as result_file:
        result_file.write(f'"{dataset_name}",{round(best_score * 100.0, 2)},"{best_params}"\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help="The path to the folder right above the dataset folder", default=None)
    parser.add_argument('--dataset_name', help="The name of the folder containing the dataset", default=None)
    parser.add_argument('--nworkers', type=int, default=4,
                        help='Number of workers for data loading  (0 to do it using main process) [Default : 10]')
    opt = parser.parse_args()

    main(dataset_path=opt.dataset_path, dataset_name=opt.dataset_name, n_jobs=opt.nworkers)
