import argparse
import pprint

import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler


def svc_grid_search(X, y, nfolds=3, n_jobs=5, cv=None, verbose=False):
    param_grid = {
        'C': list(np.logspace(-4, 3, 4)),
        'gamma': list(np.logspace(-4, 3, 4)),
        'degree': [0, 1, 2],
        'kernel': ["rbf", "poly", "linear"],
        'max_iter': [100000]
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


def main(n_jobs=10, nr_selected_feature_with_pca=200):
    # Load the data
    normalize = False
    # X, y, = np.load("/home/gitaar9/AI/ewout/weak_feature_extractor/features.npy"), np.load("/home/gitaar9/AI/ewout/weak_feature_extractor/targets.npy")
    feature_path = '/home/gitaar9/AI/ewout/weak_feature_extractor/f_tract_features.npy'
    target_path = '/home/gitaar9/AI/ewout/weak_feature_extractor/data/category_targets.npy'

    # X, y, = np.load(feature_path), np.load(target_path)
    x1 = np.load('/home/gitaar9/AI/ewout/weak_feature_extractor/f_tracts_features.npy')
    x2 = np.load('/home/gitaar9/AI/ewout/weak_feature_extractor/s_tracts_features.npy')
    X = np.concatenate((x1, x2), axis=1)
    print(X.shape)
    y = np.load(target_path)
    print(np.max(X), np.min(X))
    # X = np.log10(np.abs(X))
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # Hacky way of using the train/test sets for optimization
    all_data = np.concatenate((X_train, X_test))
    all_labels = np.concatenate((y_train, y_test))
    cv = [(list(range(len(X_train))), list(map(lambda idx: idx + len(X_train), range(len(X_test)))))]

    # Find the optimal parameters
    best_params, best_score = svc_grid_search(all_data, all_labels, 2, n_jobs=n_jobs, cv=cv, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nworkers', type=int, default=4,
                        help='Number of workers for data loading  (0 to do it using main process) [Default : 10]')
    parser.add_argument('--num_pca_features', type=int, default=None,
                        help='feature reduction dimension')
    opt = parser.parse_args()

    main(n_jobs=opt.nworkers, nr_selected_feature_with_pca=opt.num_pca_features)
