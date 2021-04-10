import math
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from ir_image_classification.data_visualization.util import marvel_int_label_to_string
from ir_image_classification.svm_classification.svm_optimization import load_dataset


def create_slices(limits=None, amount_of_slices=8, verbose=False):
    """
    This function return a list of pairs, for which each pair indicates the start and end value of the slice
    """
    limits = limits or (-math.pi, math.pi)
    slices = []
    slice_size = abs(limits[0] - limits[1]) / amount_of_slices
    for i in range(amount_of_slices):
        slice = (limits[0] + i * slice_size, limits[0] + (i + 1) * slice_size)
        slices.append(slice)
    if verbose:
        print([(math.degrees(s_start), math.degrees(s_end)) for s_start, s_end in slices])
    return slices


def sample_based_on_azimuth(X_train, y_train, a_train):
    slices = create_slices(amount_of_slices=40)
    slice_idxs = []
    for slice_start, slice_end in slices:
        mask_idxs = np.where((slice_start <= a_train) & (a_train < slice_end) & (a_train != 90))[0]
        slice_idxs.append(mask_idxs)
    desired_slice_size = max(list(map(len, slice_idxs)))

    sample_idxs = list(range(len(X_train)))
    for idxs in slice_idxs:
        amount_to_sample = desired_slice_size - len(idxs)
        sample_idxs.extend(np.random.choice(idxs, amount_to_sample))

    return X_train[sample_idxs], y_train[sample_idxs]


def load_azimuths(path):
    azimuths = []
    with open(path, 'r') as f:
        for l in f.readlines():
            name, a, e, t = l.strip().split(',')
            azimuths.append(float(a))
    azimuths = np.asarray(azimuths)
    mask = np.where(azimuths != 90)
    return azimuths, mask


def main():
    # dataset_name = "MARVEL_side_other_view_keras_ResNet152_224px"
    dataset_name = "MARVEL_keras_ResNet152_224px"
    dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets'
    dataset_path = os.path.join(dataset_path, dataset_name)

    X_train, y_train, X_test, y_test = load_dataset(
        dataset_path,
        normalize=False,
        name="",
        nr_selected_feature_with_pca=None
    )
    print(X_train.shape)
    print(X_test.shape)

    # Int labels to string
    # Add one for resnet since im stupid, removed these lines...
    y_train = np.array(list(map(marvel_int_label_to_string, y_train)))
    y_test = np.array(list(map(marvel_int_label_to_string, y_test)))


    # Azimuth stuff
    test_azimuth, test_azimuth_mask = load_azimuths('/home/gitaar9/AI/TNO/StarMap/tools/test_images_angles.txt')
    train_azimuth, train_azimuth_mask = load_azimuths('/home/gitaar9/AI/TNO/StarMap/tools/train_images_angles.txt')
    print(f"Train azimuths shape: {len(train_azimuth)}")


    # Train sampling based on azimuth
    X_train, y_train = sample_based_on_azimuth(X_train, y_train, train_azimuth)
    print(X_train.shape)

    # filtering out the tests with good azimuth estimation
    print(test_azimuth.shape)
    print(y_test.shape)
    print(X_test.shape)

    test_azimuth = test_azimuth[test_azimuth_mask]
    y_test = y_test[test_azimuth_mask]
    X_test = X_test[test_azimuth_mask]

    print(test_azimuth.shape)
    print(y_test.shape)
    print(X_test.shape)

    # Create a svm Classifier
    # clf = svm.SVC(
    #     C=1000,
    #     degree=0,
    #     gamma=1e-05,
    #     kernel='rbf',
    #     max_iter=100000,
    #     verbose=1
    # )
    # clf = svm.SVC(
    #     C=0.001,
    #     degree=2,
    #     gamma=0.1,
    #     kernel='poly',
    #     max_iter=1000,#100000,
    #     verbose=1
    # )
    # # Train the model using the training set
    # clf.fit(X_train, y_train)

    from sklearn.svm import LinearSVC
    clf = LinearSVC(C=0.001, random_state=0, tol=1e-5, max_iter=1000)
    clf.fit(X_train, y_train)

    # Validation accuracy
    pred_test = clf.predict(X_test)
    validation_accuracy = metrics.accuracy_score(y_test, pred_test)
    print("Validation Accuracy:", validation_accuracy)


    # Create a azimuth bin  accuracy plot:
    slices = create_slices(amount_of_slices=40)
    slice_accs = []
    xs = []
    for slice_start, slice_end in slices:
        x = math.degrees((slice_start + slice_end) / 2)
        xs.append(x)
        mask = np.where((slice_start <= test_azimuth) & (test_azimuth < slice_end))
        slice_pred = pred_test[mask]
        slice_gt = y_test[mask]
        slice_acc = metrics.accuracy_score(slice_gt, slice_pred) * 100
        slice_accs.append(slice_acc)
        print(int(x), slice_acc)
    plt.bar(["%.0f" % x for x in xs], slice_accs)
    plt.title('Average accuracy: {:.2f}'.format(validation_accuracy))
    plt.xlabel('Average bin azimuth in degrees')
    plt.ylabel('Average bin accuracy (%)')
    plt.xticks(rotation="vertical")
    plt.show()


if __name__ == "__main__":
    main()
