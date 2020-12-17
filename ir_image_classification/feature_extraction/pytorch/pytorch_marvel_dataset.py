import glob
import os

import numpy as np
from PIL import Image
from cv2 import cv2
from torch.utils.data import Dataset


def create_annotation_line_dict(image_id: str,
                                set_index: str,
                                class_label: str,
                                class_label_name: str) -> dict:
    return {
        "image_id": image_id,
        "is_train": True if set_index == '1' else False,
        "class_label": int(class_label),
        "class_label_name": class_label_name,
    }


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

        # Retrieve all images paths
        self.images_paths = [path for path in glob.glob(os.path.join(root_dir, 'W*_1/*.jpg'))]
        print(f'Found {len(self.images_paths)} MARVEL images.')

        # Read in annotations
        with open(os.path.join(root_dir, 'VesselClassification.dat')) as f:
            self.annotations = [line.strip().split(",") for line in f.readlines()]
        self.annotations = {
            image_id: {
                "is_train": True if set_index == '1' else False,
                "class_label": int(class_label),
                "class_label_name": class_label_name,
            } for image_id, set_index, class_label, class_label_name in self.annotations
        }

        # Remove annotations of not downloaded images + filter based on train/test set
        image_ids = {p.split('/')[-1][:-4] for p in self.images_paths}
        self.annotations = {
            key: value for key, value in self.annotations.items() if key in image_ids and value['is_train'] == is_train
        }

        # Filter image paths based on annotations
        self.images_paths = [p for p in self.images_paths if p.split('/')[-1][:-4] in self.annotations.keys()]
        print(f'{len(self.images_paths)} MARVEL images were loaded.')

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image_id = image_path.split('/')[-1][:-4]

        image = cv2.imread(self.images_paths[idx])
        image = Image.fromarray(image, 'RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.annotations[image_id]['class_label']

    def show_image(self, idx):
        image_path = self.images_paths[idx]
        image_id = image_path.split('/')[-1][:-4]
        print(f'Label of shown image is {self.annotations[image_id]["class_label_name"]}'
              f'({self.annotations[image_id]["class_label"]})')
        self[idx][0].show()

