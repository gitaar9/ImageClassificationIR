import PIL
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.applications.resnet import ResNet152
from tensorflow.python.keras.applications.resnet_v2 import ResNet152V2
from torchvision.transforms import transforms

from ir_image_classification.feature_extraction.pytorch.pytorch_labeled_shapenet_dataset import LabeledShapeNetDataset
from ir_image_classification.feature_extraction.pytorch.resnet_feature_extractor import save_features_as_npy_files


def pytorch_dataset_to_np_arrays(ds, start_idx, end_idx):
    if end_idx > len(ds):
        end_idx = len(ds)
    data, labels = zip(*[ds[idx] for idx in range(start_idx, end_idx)])
    data = np.array([np.asarray(img) for img in data])
    labels = np.array(labels)

    return data, labels


def build_complete_feature_extraction_pipeline(cnn_class=ResNet152V2,
                                               preprocess_function=tf.keras.applications.resnet_v2.preprocess_input):
    """
    Loads a pretrained ResNet model and removes the dense layer, adds the standard preprocessing and returns the two
    as a pipeline.
    """
    # load pretrained model
    model = cnn_class(
        include_top=True,
        weights="imagenet",
        pooling=None,
    )
    model = tf.keras.Model(model.inputs, model.layers[-2].output)  # Remove the last dense layer

    # Create and compile the pipeline
    inputs = tf.keras.Input(shape=(224, 224, 3))
    preprocessed_inputs = preprocess_function(inputs)
    outputs = model(preprocessed_inputs)

    pipeline = tf.keras.Model(inputs, outputs)
    return pipeline


def main():
    # Settings
    cnn_class = ResNet152
    preprocess_function = tf.keras.applications.resnet.preprocess_input

    # Load the data
    # Create the model
    model = build_complete_feature_extraction_pipeline(
        cnn_class=cnn_class,
        preprocess_function=preprocess_function
    )

    # Directories and names of the extracted dataset
    root_extracted_dataset_dir = "/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets"
    output_dataset_name = f"labeled_shapenet_keras_{cnn_class.__name__}_224px"
    print(f"Saving the extracted dataset as: {output_dataset_name}")

    # Load data
    image_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)
    ])

    batch_size = 100

    train_ds = LabeledShapeNetDataset(image_transform)
    train_features = []
    train_labels = []
    for idx in range(0, len(train_ds), batch_size):
        print(idx)
        data, labels = pytorch_dataset_to_np_arrays(train_ds, idx, idx + batch_size)
        features = model.predict(data)
        train_features.append(features)
        train_labels.extend(labels)
    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.asarray(train_labels)
    print(train_features.shape)
    print(train_labels.shape)
    save_features_as_npy_files(train_features, train_labels, root_extracted_dataset_dir, output_dataset_name, 'train')


if __name__ == "__main__":
    main()
