import os

from sklearn import svm, metrics
import matplotlib.pyplot as plt
from ir_image_classification.data_visualization.util import get_random_permutation, marvel_int_label_to_string
from ir_image_classification.feature_extraction.pytorch.resnet_feature_extractor import save_features_as_npy_files
from ir_image_classification.svm_classification.svm_optimization import load_dataset
from sklearn.metrics import plot_confusion_matrix
from joblib import dump
import numpy as np


# OLD LOAD DATAPART DONT THROW AWAY!!
# dataset_name = "resnet50_224px"
# dataset_name = "MARVEL_keras_ResNet152_224px"

# dataset_name = "MARVEL_side_other_view_keras_ResNet152_224px"
dataset_name = "Resnet_and_Pix2Vox_combined_normalized"
dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets'
dataset_path = os.path.join(dataset_path, dataset_name)
name = ""
pca_dimensions = None
normalize = False

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

# Int labels to string
# Add one for resnet since im stupid
# y_train += 1
# y_test += 1
y_train = np.array(list(map(marvel_int_label_to_string, y_train)))
y_test = np.array(list(map(marvel_int_label_to_string, y_test)))


# Create a svm Classifier
# clf = svm.SVC(
#     C=1000,
#     degree=0,
#     gamma=1e-05,
#     kernel='rbf',
#     max_iter=100000,
#     verbose=1
# )
clf = svm.SVC(
    C=0.001,
    degree=2,
    gamma=0.1,
    kernel='poly',
    max_iter=100000,
    verbose=1
)

# Train the model using the training set
clf.fit(X_train, y_train)

# Validation accuracy
pred_test = clf.predict(X_test)
validation_accuracy = metrics.accuracy_score(y_test, pred_test)
print("Validation Accuracy:", validation_accuracy)
plot_confusion_matrix(clf, X_test, y_test, normalize='true', xticks_rotation='vertical', values_format='.2f', cmap='hot')
plt.title("Validation accuracy: {:.2f}".format(validation_accuracy))
plt.show()

# Train accuracy
# pred_train = clf.predict(X_train)
# print("Train Accuracy:", metrics.accuracy_score(y_train, pred_train))
#
# name = "trained_on_newds_256_ft_not_normalized"
# dump(
#     clf,
#     f'/home/gitaar9/TNO_Thesis/ImageClassificationIR/ir_image_classification/svm_classification/trained_svms/{name}'
# )

#### Bullshit finetuned Pix2Vox++ results:
# For 5000 samples:
# Validation Accuracy: 0.15665739614114374
# Train Accuracy: 0.9718
# For 3000 samples with max_iter:
# [LibSVM]Validation Accuracy: 0.17056318442551713
# Train Accuracy: 0.9873333333333333

# For 10000 samples:
# [LibSVM]Validation Accuracy: 0.1760820441508778
# Train Accuracy: 0.9409
# For 10000 samples with max_iter:
# LibSVM]Validation Accuracy: 0.1759516773857118
# Train Accuracy: 0.9408

# For MARVEL_side_other_view_keras_ResNet152_224px
# The best accuracy was 74.53% using: {'C': 0.001, 'degree': 2, 'gamma': 0.1, 'kernel': 'poly', 'max_iter': 100000}