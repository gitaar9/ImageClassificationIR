import os

from ir_image_classification.feature_extraction.pytorch.resnet_feature_extractor import save_features_as_npy_files
from ir_image_classification.svm_classification.svm_optimization import load_dataset
import numpy as np


def combine_datasets():
    dataset_name = "MARVEL_side_other_view_keras_ResNet152_224px"
    dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets'
    dataset_path = os.path.join(dataset_path, dataset_name)
    name = ""
    pca_dimensions = None
    normalize = True
    X_train_ResNet, y_train_ResNet, X_test_ResNet, y_test_ResNet = load_dataset(
        dataset_path,
        normalize=normalize,
        name=name,
        nr_selected_feature_with_pca=pca_dimensions
    )
    print(f"Resnet train/test shape: {X_train_ResNet.shape}, {X_test_ResNet.shape}")

    dataset_name = "Pix2Vox_side_other_view_256_ft_300"
    dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets'
    dataset_path = os.path.join(dataset_path, dataset_name)
    name = ""
    pca_dimensions = 2048
    normalize = True

    X_train_P2V, y_train_P2V, X_test_P2V, y_test_P2V = load_dataset(
        dataset_path,
        normalize=normalize,
        name=name,
        nr_selected_feature_with_pca=pca_dimensions
    )
    print(f"Pix2Vox++ train/test shape: {X_train_P2V.shape}, {X_test_P2V.shape}")

    X_train = np.concatenate((X_train_ResNet, X_train_P2V), axis=1)
    X_test = np.concatenate((X_test_ResNet, X_test_P2V), axis=1)
    print(f"Combined train/test shape: {X_train.shape}, {X_test.shape}")

    output_name = 'Resnet_and_Pix2Vox_combined_normalized'

    save_features_as_npy_files(X_train, y_train_P2V, 'datasets/extracted_datasets', output_name, 'train')
    save_features_as_npy_files(X_test, y_test_P2V, 'datasets/extracted_datasets', output_name, 'test')

combine_datasets()