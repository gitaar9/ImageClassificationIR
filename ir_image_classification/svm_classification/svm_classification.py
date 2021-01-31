import os

from sklearn import svm, metrics

from ir_image_classification.data_visualization.util import get_random_permutation
from ir_image_classification.svm_classification.svm_optimization import load_dataset
from joblib import dump

# Load data
# dataset_name = "resnet50_224px"
# dataset_name = "MARVEL_keras_ResNet152_224px"

# dataset_name = "MARVEL_side_other_view_keras_ResNet152_224px"
# dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets'
# dataset_path = os.path.join(dataset_path, dataset_name)
# normalize = False
# name = ""

name = "side_other_view_early_newds_256_ft_300_"
dataset_path = '/home/gitaar9/AI/TNO/Pix2VoxPP/extracted_datasets'
normalize = False

X_train, y_train, X_test, y_test = load_dataset(dataset_path, normalize=normalize, name=name)
print(X_train.shape)
print(X_test.shape)
# Create a svm Classifier
clf = svm.SVC(
    C=1000,
    degree=3,
    gamma=1e-05,
    kernel='rbf',
    max_iter=100000,
    verbose=1
)

# Train the model using the training set
clf.fit(X_train, y_train)

# Validation accuracy
pred_test = clf.predict(X_test)
print("Validation Accuracy:", metrics.accuracy_score(y_test, pred_test))

# Train accuracy
pred_train = clf.predict(X_train)
print("Train Accuracy:", metrics.accuracy_score(y_train, pred_train))

name = "trained_on_newds_256_ft_not_normalized"
dump(
    clf,
    f'/home/gitaar9/TNO_Thesis/ImageClassificationIR/ir_image_classification/svm_classification/trained_svms/{name}'
)

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