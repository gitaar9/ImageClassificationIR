import os

import PIL
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.applications.resnet import ResNet152
from torchvision.transforms import transforms

from ir_image_classification.feature_extraction.pytorch.pytorch_vais_dataset import VAISDataset
from ir_image_classification.feature_extraction.pytorch.resnet_feature_extractor import save_features_as_npy_files


def pytorch_dataset_to_np_arrays(*args, **kwargs):
    ds = VAISDataset(*args, **kwargs)
    data, labels = zip(*[ds[idx] for idx in range(len(ds))])
    data = np.array([np.asarray(img) for img in data])
    labels = np.array(labels)

    return data, labels


def load_train_test_set_from_pytorch_dataset(*args, **kwargs):
    """
    Using the Pytorch VAIS dataset to load in the train and test images to numpy arrays
    """
    kwargs['root_dir'] = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/VAIS'
    image_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)
    ])
    X_train, y_train = pytorch_dataset_to_np_arrays(is_train=True, transform=image_transform, *args, **kwargs)
    X_test, y_test = pytorch_dataset_to_np_arrays(is_train=False, transform=image_transform, *args, **kwargs)
    return X_train, y_train, X_test, y_test


def build_complete_feature_extraction_pipeline():
    """
    Loads a pretrained ResNet model and removes the dense layer, adds the standard preprocessing and returns the two
    as a pipeline.
    """
    # load pretrained model
    model = ResNet152(
        include_top=True,
        weights="imagenet",
        pooling=None,
    )
    model = tf.keras.Model(model.inputs, model.layers[-2].output)  # Remove the last dense layer
    preprocess_input = tf.keras.applications.resnet.preprocess_input  # Resnet preprocessing function

    # Create and compile the pipeline
    inputs = tf.keras.Input(shape=(224, 224, 3))
    preprocessed_inputs = preprocess_input(inputs)
    outputs = model(preprocessed_inputs)

    pipeline = tf.keras.Model(inputs, outputs)
    return pipeline


def main():
    # load data
    X_train, y_train, X_test, y_test = load_train_test_set_from_pytorch_dataset(is_ir=True)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # Create the model
    model = build_complete_feature_extraction_pipeline()

    root_extracted_dataset_dir = "/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets"

    for (X, y), set_name in zip([(X_train, y_train), (X_test, y_test)], ['train', 'test']):
        features = model.predict(X)
        output_dataset_name = f"keras_resnet152_224px_{set_name}"
        save_features_as_npy_files(features, y, os.path.join(root_extracted_dataset_dir, output_dataset_name))


if __name__ == "__main__":
    main()
