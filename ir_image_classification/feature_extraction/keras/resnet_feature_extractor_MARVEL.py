import PIL
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.resnet import ResNet152
from tensorflow.python.keras.applications.resnet_v2 import ResNet152V2
from torchvision.transforms import transforms

from ir_image_classification.feature_extraction.keras.keras_dataset import marvel_dataframe, \
    marvel_side_other_view_dataframe
from ir_image_classification.feature_extraction.pytorch.pytorch_marvel_dataset import MARVELDataset
from ir_image_classification.feature_extraction.pytorch.resnet_feature_extractor import save_features_as_npy_files


def pytorch_dataset_to_np_arrays(ds, start_idx, end_idx):
    if end_idx > len(ds):
        end_idx = len(ds)
    data, labels = zip(*[ds[idx] for idx in range(start_idx, end_idx)])
    data = np.array([np.asarray(img) for img in data])
    labels = np.array(labels)

    return data, labels


def build_complete_feature_extraction_pipeline(cnn_class=ResNet152V2):
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
    return model
    # # Create and compile the pipeline
    # inputs = tf.keras.Input(shape=(224, 224, 3))
    # preprocessed_inputs = preprocess_function(inputs)
    # outputs = model(preprocessed_inputs)
    #
    # pipeline = tf.keras.Model(inputs, outputs)
    # return pipeline


def main():
    # Settings
    cnn_class = ResNet152
    root_dir = '/home/gitaar9/AI/TNO/marveldataset2016/'
    batch_size = 100
    data_subset = 'test'

    # Create the model
    model = build_complete_feature_extraction_pipeline(cnn_class=cnn_class)

    # Load data
    df = marvel_dataframe(root_dir, is_train=(data_subset == 'train'))
    # df = marvel_side_other_view_dataframe(is_train=(data_subset == 'train'), cast_labels_to=int)
    datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input)
    data_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=None,
        x_col="paths",
        y_col="labels",
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        class_mode="raw",
        target_size=(224, 224),
    )

    # Directories and names of the extracted dataset
    root_extracted_dataset_dir = "/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets"
    output_dataset_name = f"MARVEL_side_other_view_keras_{cnn_class.__name__}_224px"

    # Main prediction loop
    all_features = []
    all_labels = []
    for batch_idx, (x_batch, y_batch) in enumerate(data_generator):
        if batch_idx >= len(data_generator):
            break
        print(f"{batch_idx}/{len(data_generator)}\r")
        features = model.predict(x_batch)
        all_features.append(features)
        all_labels.extend(y_batch)

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.asarray(all_labels)

    # Save the features to an npy file
    print(all_features.shape)
    print(all_labels.shape)
    print(f"Saving the extracted dataset as: {output_dataset_name}")
    save_features_as_npy_files(all_features, all_labels, root_extracted_dataset_dir, output_dataset_name, data_subset)


if __name__ == "__main__":
    main()
