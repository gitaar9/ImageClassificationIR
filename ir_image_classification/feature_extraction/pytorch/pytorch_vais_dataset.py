import os

import numpy as np
from PIL import Image
from cv2 import cv2
from torch.utils.data import Dataset


def basic_label_to_int(basic_level: str) -> int:
    return {
        "small": 0,
        "passenger": 1,
        "medium-other": 2,
        "tug": 3,
        "sailing": 4,
        "cargo": 5
    }[basic_level]


def create_annotation_line_dict(root_dir: str,
                                visible_path: str,
                                ir_path: str,
                                fine_grained_label: str,
                                basic_label: str,
                                unique_id: str,
                                is_train: str,
                                is_night: str) -> dict:
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

    # Preprocessing methods
    NONE = 0
    INVERT_EQUALIZE = 1
    THREE_CHANNEL_NONE_INVERT_EQUALIZE = 2
    INVERT = 3
    EQUALIZE = 4

    def __init__(self, root_dir, is_train=None, is_night=None, is_ir=True, transform=None, preprocessing_method=0):
        """
        :param root_dir: The root directory of the VAIS dataset (where the annotations.txt file is)
        :param is_train: When this is true only training images are loaded (Can be True, False and None)
        :param is_night: When this is true only night images are loaded (Can be True, False and None)
        :param is_ir: When this is true only infrared images are loaded otherwise we load the RGB images
        :param transform: The transform that should be applied to every image
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_ir = is_ir
        self.is_train = is_train
        self.is_night = is_night
        self.preprocessing_method = preprocessing_method

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

    def preprocess_ir_image(self, image: np.array) -> np.array:
        if self.preprocessing_method == self.NONE:
            channels = [image] * 3
        elif self.preprocessing_method == self.INVERT_EQUALIZE:
            channels = [cv2.equalizeHist(np.invert(image))] * 3
        elif self.preprocessing_method == self.THREE_CHANNEL_NONE_INVERT_EQUALIZE:
            channels = [image, np.invert(image), cv2.equalizeHist(image)]
        elif self.preprocessing_method == self.INVERT:
            channels = [np.invert(image)] * 3
        elif self.preprocessing_method == self.EQUALIZE:
            channels = [cv2.equalizeHist(image)] * 3
        else:
            raise ValueError("Unknown preprocessing type")
        return np.stack(channels, axis=2)

    def __getitem__(self, idx):
        if self.is_ir:
            image = cv2.imread(self.annotations[idx]['ir_path'], cv2.IMREAD_GRAYSCALE)
            image = self.preprocess_ir_image(image)
        else:
            image = cv2.imread(self.annotations[idx]['visible_path'])

        image = Image.fromarray(image, 'RGB')

        if self.transform:
            image = self.transform(image)
        return image, self.annotations[idx]['basic_label_int']

    def show_image(self, idx, individual_channels_as_imgs=False):
        if individual_channels_as_imgs:
            array_of_image = np.asarray(self[idx][0])
            Image.fromarray(array_of_image[:, :, 0], 'L').show()
            Image.fromarray(array_of_image[:, :, 1], 'L').show()
            Image.fromarray(array_of_image[:, :, 2], 'L').show()
        else:
            self[idx][0].show()
