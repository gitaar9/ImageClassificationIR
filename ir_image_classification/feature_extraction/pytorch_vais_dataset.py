import os

import numpy as np
import torch
from PIL import Image
from cv2 import cv2
from torch.utils.data import Dataset


def basic_label_to_int(basic_level):
    return {
        "small": 0,
        "passenger": 1,
        "medium-other": 2,
        "tug": 3,
        "sailing": 4,
        "cargo": 5
    }[basic_level]


def create_annotation_line_dict(root_dir, visible_path, ir_path, fine_grained_label, basic_label, unique_id, is_train,
                                is_night):
    return {
        "visible_path": os.path.join(root_dir, visible_path) if visible_path != 'null' else None,
        "ir_path": os.path.join(root_dir, ir_path) if ir_path != 'null' else None,
        "fine_grained_label": fine_grained_label,
        "basic_label": basic_label,
        "basic_label_int": basic_label_to_int(basic_label),
        "unique_id": int(unique_id),
        "is_train": bool(int(is_train)),
        "is_night": bool(int(is_night)),
    }


class VAISDataset(Dataset):
    """VAIS dataset."""

    def __init__(self, root_dir, is_train=None, is_night=None, is_ir=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_ir = is_ir
        self.is_train = is_train
        self.is_night = is_night

        # Read in annotations
        with open(os.path.join(root_dir, 'annotations.txt')) as f:
            self.annotations = f.readlines()
        self.annotations = [
            create_annotation_line_dict(root_dir, *line.strip().split(" ")) for line in self.annotations
        ]
        # Filter the images based on different arguments
        if is_train is not None:
            self.annotations = [annotation for annotation in self.annotations if annotation['is_train'] == is_train]
        if is_night is not None:
            self.annotations = [annotation for annotation in self.annotations if annotation['is_night'] == is_night]
        if is_ir:
            self.annotations = [annotation for annotation in self.annotations if annotation['ir_path']]
        else:
            self.annotations = [annotation for annotation in self.annotations if annotation['visible_path']]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.is_ir:
            image = cv2.imread(self.annotations[idx]['ir_path'], cv2.IMREAD_GRAYSCALE)
            image = np.stack([image] * 3)  # duplicate the single channel three times
        else:
            image = cv2.imread(self.annotations[idx]['visible_path'])

        image = Image.fromarray(image, 'RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.annotations[idx]['basic_label_int']

    def show_image(self, idx):
        img = Image.fromarray(self[idx][0], 'L' if self.is_ir else 'RGB')
        img.show()
