import os
from typing import Tuple

import PIL
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from ir_image_classification.feature_extraction.pytorch.feature_extracting_pretrained_resnets import \
    feature_extracting_resnet152, FeatureExtractingResNet
from ir_image_classification.feature_extraction.pytorch.pytorch_vais_dataset import VAISDataset


def get_dataloader(batch_size: int = 20, *args, **kwargs) -> DataLoader:
    """
    Loads the VAISDataset into a dataloader
    :param batch_size: How big the batches that come out of the dataloader will be
    :param args: Arguments for the VAISDataset
    :param kwargs: Arguments for the VAISDataset
    """
    image_transform = transforms.Compose([
        # transforms.Resize((28, 28), interpolation=PIL.Image.BICUBIC),
        # transforms.Resize((112, 112), interpolation=PIL.Image.BICUBIC),
        transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = VAISDataset(root_dir='/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/VAIS',
                          transform=image_transform, *args, **kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return dataloader


def get_resnet_model(device: torch.device) -> FeatureExtractingResNet:
    """
    Instantiates the Resnet152 model
    :param device: Either cpu or gpu
    :return:
    """
    resnet = feature_extracting_resnet152(pretrained=True)
    resnet.to(device)
    resnet.eval()
    return resnet


def get_features(model: FeatureExtractingResNet,
                 dataloader: DataLoader,
                 device: torch.device) -> Tuple[np.array, np.array]:
    """
    Predict feature vectors for all data in the dataloader, returns the feature vectors and labels as np arrays.
    :param model: The pytorch model which outputs a feature vector
    :param dataloader: The dataloader from which we can load batches of data
    :param device: Either cpu or gpu
    """
    outputs = []
    labels = []
    for i_batch, (batch_inputs, batch_labels) in enumerate(dataloader):
        print(f"{i_batch}: {batch_inputs.shape}")
        batch_inputs = batch_inputs.to(device)
        batch_outputs = model(batch_inputs)
        outputs.append(batch_outputs.detach().cpu())
        labels.append(batch_labels.detach().cpu())

    outputs = torch.cat(outputs).numpy()
    labels = torch.cat(labels).numpy()

    return outputs, labels


def save_features_as_npy_files(features: np.array, labels: np.array, ds_dir: str, name: str, set_name: str):
    """
    Creates a directory with the given name in the given ds_dir and saves the features there as npy files.
    :param features: np array of features
    :param labels: np array of labels
    :param ds_dir: The parent directory of the new dataset
    :param name: The name of the new directory which contains the new dataset
    :param set_name: Whether the data we save is train or test data.
    """
    extracted_ds_dir = os.path.join(ds_dir, name)
    os.makedirs(extracted_ds_dir, exist_ok=True)  # Create the directory if needed
    np.save(os.path.join(extracted_ds_dir, f"{set_name}_features.npy"), features)
    np.save(os.path.join(extracted_ds_dir, f"{set_name}_labels.npy"), labels)


def main():
    """
    Creates a set of .npy files that can serve as a dataset for the SVM classification
    """
    # Settings
    load_ir_images = True
    preprocessing_method = VAISDataset.EQUALIZE

    # Decide to run on gpu or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    resnet_model = get_resnet_model(device)

    # Set the name and directory for the extracted dataset
    root_extracted_dataset_dir = "/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets"
    output_dataset_name = f"pytorch_resnet152_224px"
    if load_ir_images:
        output_dataset_name += f'_pp{preprocessing_method}'
    else:
        output_dataset_name += '_RGB'
    print(f"Saving the extracted dataset as: {output_dataset_name}")

    for set_name in ['train', 'test']:
        dataloader = get_dataloader(
            batch_size=10,
            is_train=(set_name == "train"),
            is_ir=load_ir_images,
            preprocessing_method=preprocessing_method
        )
        features, labels = get_features(resnet_model, dataloader, device)

        # Save the feature dataset as npy file
        save_features_as_npy_files(features, labels, root_extracted_dataset_dir, output_dataset_name, set_name)


if __name__ == "__main__":
    main()
