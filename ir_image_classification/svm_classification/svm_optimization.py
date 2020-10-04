import os

from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit


def load_dataset(dataset_path, dataset_name, normalize=True):
    X_train = np.load(os.path.join(dataset_path, f"{dataset_name}_train_features.npy"))
    y_train = np.load(os.path.join(dataset_path, f"{dataset_name}_train_labels.npy"))
    X_test = np.load(os.path.join(dataset_path, f"{dataset_name}_test_features.npy"))
    y_test = np.load(os.path.join(dataset_path, f"{dataset_name}_test_labels.npy"))

    if normalize:
        scaler = StandardScaler()
        all_data = np.concatenate((X_train, X_test))
        all_data = scaler.fit_transform(all_data)  # Scale train and test data at the same time
        X_train = all_data[:len(X_train)]
        X_test = all_data[len(X_train):]

    return X_train, y_train, X_test, y_test


def svc_grid_search(X, y, nfolds, n_jobs=5):
    param_grid = {
        'C': [0.5, 1, 10, 20],
        'gamma': [0.005, 0.01, 0.1, 1],
        'degree': [1, 2, 3],
        'kernel': ["rbf", "poly"],  # "linear", "sigmoid"
    }
    # param_grid = {
    #     'C': np.logspace(-2, 5, 6),
    #     'gamma': np.logspace(-5, 3, 6),
    #     'degree': [1, 2, 3],
    #     'kernel': ["rbf", "poly"],
    # }
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=cv, n_jobs=n_jobs)
    grid_search.fit(X, y)
    for params, result in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
        print(f"{params}: {result}")

    print()
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    return grid_search.best_params_


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


def main():
    dataset_name = "resnet50_224px"
    dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets'
    X_train, y_train, X_test, y_test = load_dataset(dataset_path, dataset_name)
    svc_grid_search(X_train, y_train, 5, n_jobs=6)


if __name__ == "__main__":
    main()
