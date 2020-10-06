import PIL
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.applications.resnet import ResNet152
from tensorflow.python.keras.applications.resnet_v2 import ResNet152V2
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
    load_ir_images = False
    preprocessing_method = VAISDataset.THREE_CHANNEL_NONE_INVERT_EQUALIZE

    # Load the data
    X_train, y_train, X_test, y_test = load_train_test_set_from_pytorch_dataset(
        is_ir=load_ir_images,
        preprocessing_method=preprocessing_method
    )
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # Create the model
    model = build_complete_feature_extraction_pipeline(
        cnn_class=cnn_class,
        preprocess_function=preprocess_function
    )

    # Directories and names of the extracted dataset
    root_extracted_dataset_dir = "/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets"
    output_dataset_name = f"keras_{cnn_class.__name__}_{X_train[0].shape[0]}px"
    if load_ir_images:
        output_dataset_name += f'_pp{preprocessing_method}'
    else:
        output_dataset_name += "_RGB"
    print(f"Saving the extracted dataset as: {output_dataset_name}")

    # Extract the features and save them as .npy files
    for (X, y), set_name in zip([(X_train, y_train), (X_test, y_test)], ['train', 'test']):
        features = model.predict(X)
        save_features_as_npy_files(features, y, root_extracted_dataset_dir, output_dataset_name, set_name)


if __name__ == "__main__":
    main()
