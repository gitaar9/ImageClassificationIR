import glob
import os

import numpy as np
from PIL import Image
from cv2 import cv2
from torch.utils.data import Dataset

from ir_image_classification.feature_extraction.keras.keras_dataset import valid_image_paths_and_filtered_annotations


class MARVELDataset(Dataset):
    """MARVEL dataset."""

    def __init__(self, root_dir, is_train=True, transform=None):
        """
        :param root_dir: The root directory of the MARVEL dataset (where the annotations.txt file is)
        :param is_train: When this is true only training images are loaded (Can be True, False)
        :param transform: The transform that should be applied to every image
        """
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths, self.annotations = valid_image_paths_and_filtered_annotations(root_dir, is_train)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_id = image_path.split('/')[-1][:-4]

        image = cv2.imread(self.image_paths[idx])
        image = Image.fromarray(image, 'RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.annotations[image_id]['class_label']

    def show_image(self, idx):
        image_path = self.image_paths[idx]
        image_id = image_path.split('/')[-1][:-4]
        print(f'Label of shown image is {self.annotations[image_id]["class_label_name"]}'
              f'({self.annotations[image_id]["class_label"]})')
        self[idx][0].show()
