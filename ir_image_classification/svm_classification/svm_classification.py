import os

from sklearn import svm, metrics

from ir_image_classification.data_visualization.util import get_random_permutation
from ir_image_classification.svm_classification.svm_optimization import load_dataset

# Load data
# dataset_name = "resnet50_224px"
# dataset_name = "MARVEL_keras_ResNet152_224px"

dataset_name = "MARVEL_side_other_view_keras_ResNet152_224px"
dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets'
dataset_path = os.path.join(dataset_path, dataset_name)
normalize = False
name = ""


# name = "with_random_bg_scratch_early_features_subset_"
# dataset_path = '/home/gitaar9/AI/TNO/Pix2VoxPP'
# normalize = True

X_train, y_train, X_test, y_test = load_dataset(dataset_path, normalize=normalize, name=name)

# Shuffle the dataset
rndperm = get_random_permutation(X_train.shape[0])

# Take only 30000 samples
N = 30000
print(X_train.shape, y_train.shape)
X_train_subset = X_train[rndperm][:N].copy()
y_train_subset = y_train[rndperm][:N].copy()
print(X_train_subset.shape, y_train_subset.shape)

# Create a svm Classifier
clf = svm.SVC(
    C=1,
    degree=3,
    gamma=0.1,
    kernel='poly',
    max_iter=100000,
    verbose=1
)

# Train the model using the training set
clf.fit(X_train_subset, y_train_subset)

# Validation accuracy
pred_test = clf.predict(X_test)
print("Validation Accuracy:", metrics.accuracy_score(y_test, pred_test))

# Train accuracy
pred_train = clf.predict(X_train_subset)
print("Train Accuracy:", metrics.accuracy_score(y_train_subset, pred_train))



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