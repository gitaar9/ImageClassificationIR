from PIL import Image
from cv2 import cv2
from torch.utils.data import Dataset


class KerasDFDataset(Dataset):
    """Uses a keras df with paths and labels to create a pytorch dataset."""

    def __init__(self, df, transform=None):
        """
        :param df: The pandas df used by keras for it's flow_from_df method
        :param transform: The transform that should be applied to every image
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path, label = self.df.loc[idx]
        image = cv2.imread(path)
        image = Image.fromarray(image, 'RGB')

        if self.transform:
            image = self.transform(image)

        return image, int(label)

    def show_image(self, idx):
        image, label = self[idx]
        print(f"label = {label}")
        image.show()
