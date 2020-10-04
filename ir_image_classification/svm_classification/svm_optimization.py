import os

from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import pprint


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


def svc_grid_search(X, y, nfolds=3, n_jobs=5, cv=None):
    # param_grid = {
    #     'C': [0.5, 1, 10, 20],
    #     'gamma': [0.005, 0.01, 0.1, 1],
    #     'degree': [1, 2, 3],
    #     'kernel': ["rbf", "poly"],  # "linear", "sigmoid"
    # }
    param_grid = {
        'C': list(np.logspace(-3, 3, 7)),
        # 'nu': [0.2, 0.3, 0.5, 0.7, 0.8],
        'gamma': list(np.logspace(-5, 1, 7)),
        'degree': [0, 1, 2, 3],
        'kernel': ["rbf", "poly"],
        'max_iter': [10000]
    }

    print("Performing grid search with the following parameters:\n")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(param_grid)

    if cv is None:
        cv = StratifiedShuffleSplit(n_splits=nfolds, test_size=0.2, random_state=42)
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=cv, n_jobs=n_jobs, verbose=1)
    grid_search.fit(X, y)
    for params, result in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
        print(f"{params}: {result}")

    print("\nThe best accuracy was {:.2f} using:\n{}".format(grid_search.best_score_ * 100.0, grid_search.best_params_))

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


def main():
    # dataset_name = "resnet50_224px_normalized_between_0_1"
    # dataset_name = "resnet50_224px_relu"
    # dataset_name = "resnet50_224px"
    dataset_name = "resnet50_224px_RGB"
    # dataset_name = "resnet50_112px"
    # dataset_name = "resnet50_56px"
    # dataset_name = "resnet50_28px"
    # dataset_name = "resnet50_no_scaling"

    dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets'
    X_train, y_train, X_test, y_test = load_dataset(dataset_path, dataset_name)
    print(X_train.shape, X_test.shape)

    # Hacky way of using the train/test sets for optimization
    all_data = np.concatenate((X_train, X_test))
    all_labels = np.concatenate((y_train, y_test))
    cv = [(list(range(len(X_train))), list(map(lambda idx: idx + len(X_train), range(len(X_test)))))]

    svc_grid_search(all_data, all_labels, 3, n_jobs=10, cv=cv)


if __name__ == "__main__":
    main()
