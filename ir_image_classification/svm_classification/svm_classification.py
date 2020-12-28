import os

from sklearn import svm, metrics

from ir_image_classification.data_visualization.util import get_random_permutation
from ir_image_classification.svm_classification.svm_optimization import load_dataset

# Load data
# dataset_name = "resnet50_224px"

# dataset_name = "MARVEL_keras_ResNet152_224px"
# dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets'
# dataset_path = os.path.join(dataset_path, dataset_name)
# normalize = False

dataset_path = '/home/gitaar9/AI/TNO/Pix2VoxPP'
normalize = True

X_train, y_train, X_test, y_test = load_dataset(dataset_path, normalize=normalize)



# Shuffle the dataset
rndperm = get_random_permutation(X_train.shape[0])

# Take only 30000 samples
N = 20000
print(X_train.shape, y_train.shape)
X_train_subset = X_train[rndperm][:N].copy()
y_train_subset = y_train[rndperm][:N].copy()
print(X_train_subset.shape, y_train_subset.shape)

# Create a svm Classifier
clf = svm.SVC(
    C=1,
    degree=3,
    gamma=0.1,
    kernel='poly'
)

# Train the model using the training set
clf.fit(X_train_subset, y_train_subset)

# Validation accuracy
pred_test = clf.predict(X_test)
print("Validation Accuracy:", metrics.accuracy_score(y_test, pred_test))

# Train accuracy
pred_train = clf.predict(X_train_subset)
print("Train Accuracy:", metrics.accuracy_score(y_train_subset, pred_train))



