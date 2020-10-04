import os

import PIL
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from ir_image_classification.feature_extraction.feature_extracting_pretrained_resnets import feature_extracting_resnet50
from ir_image_classification.feature_extraction.pytorch_vais_dataset import VAISDataset


def get_dataloader(batch_size=20, *args, **kwargs):
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


def get_resnet_model(device):
    resnet = feature_extracting_resnet50(pretrained=True)
    resnet.to(device)
    resnet.eval()
    return resnet


def get_features(model, dataloader, device):
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


def save_features_as_npy_files(features, labels, name):
    np.save(f"{name}_features.npy", features)
    np.save(f"{name}_labels.npy", labels)


def main():
    """
    Creates a set of .npy files that can serve as a dataset
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    resnet_model = get_resnet_model(device)

    for set in ['train', 'test']:
        dataloader = get_dataloader(batch_size=10, is_train=(set == "train"), is_ir=False)
        features, labels = get_features(resnet_model, dataloader, device)

        # Save the feature dataset as npy file
        output_dataset_name = f"resnet50_224px_RGB_{set}"
        root_extracted_datset_dir = "/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets"
        save_features_as_npy_files(features, labels, os.path.join(root_extracted_datset_dir, output_dataset_name))


if __name__ == "__main__":
    main()
