from sklearn import svm, metrics

from ir_image_classification.svm_classification.svm_optimization import load_dataset


# Load data
dataset_name = "resnet50_224px"
dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets'
X_train, y_train, X_test, y_test = load_dataset(dataset_path, dataset_name)

# Create a svm Classifier
clf = svm.SVC(
    C=20,
    degree=3,
    gamma=0.1,
    kernel='poly'
)

# Train the model using the training set
clf.fit(X_train, y_train)

# Validation accuracy
pred_test = clf.predict(X_test)
print("Validation Accuracy:", metrics.accuracy_score(y_test, pred_test))

# Train accuracy
pred_train = clf.predict(X_train)
print("Train Accuracy:", metrics.accuracy_score(y_train, pred_train))



