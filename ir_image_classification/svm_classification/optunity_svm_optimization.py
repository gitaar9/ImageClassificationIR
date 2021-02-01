import argparse
import os
import pprint

import numpy as np
import optunity
import sklearn
from optunity.metrics import accuracy
from sklearn import svm
# from skopt import BayesSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from ir_image_classification.data_visualization.util import get_random_permutation
from ir_image_classification.svm_classification.svm_optimization import load_dataset

dataset_name = "Resnet_and_Pix2Vox_combined"
dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets'
dataset_path = os.path.join(dataset_path, dataset_name)
name = ""
pca_dimensions = None
normalize = True

# name = "side_other_view_early_newds_256_ft_300_"
# dataset_path = '/home/gitaar9/AI/TNO/Pix2VoxPP/extracted_datasets'
# pca_dimensions = 2048
# normalize = False
#
X_train, y_train, X_test, y_test = load_dataset(
    dataset_path,
    normalize=normalize,
    name=name,
    nr_selected_feature_with_pca=pca_dimensions
)
print(X_train.shape)
print(X_test.shape)


@optunity.cross_validated(x=X_train, y=y_train, num_folds=3)
def svm_default_auroc(x_train, y_train, x_test, y_test):
    model = sklearn.svm.SVC().fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    acc = accuracy(y_test, decision_values)
    return acc

svm_default_auroc()

# cv_decorator = optunity.cross_validated(x=X_train, y=y_train, num_folds=3)
#
# def svm_rbf_tuned_auroc(x_train, y_train, x_test, y_test, C, logGamma):
#     model = sklearn.svm.SVC(C=C, gamma=10 ** logGamma).fit(x_train, y_train)
#     decision_values = model.decision_function(x_test)
#     auc = optunity.metrics.roc_auc(y_test, decision_values)
#     return auc
#
# svm_rbf_tuned_auroc = cv_decorator(svm_rbf_tuned_auroc)
# svm_rbf_tuned_auroc(C=1.0, logGamma=0.0)