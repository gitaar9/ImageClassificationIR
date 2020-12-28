import PIL
from PIL import Image
from cv2 import cv2
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class LabeledShapeNetDataset(Dataset):
    """MARVEL dataset."""

    def __init__(self, root_dir=None, is_train=True, transform=None):
        """
        :param root_dir: The root directory of the MARVEL dataset (where the annotations.txt file is)
        :param is_train: When this is true only training images are loaded (Can be True, False)
        :param transform: The transform that should be applied to every image
        """
        self.image_path = '/home/gitaar9/AI/TNO/Pix2VoxPP/datasets/ShapeNet/ShapeNetRenderingOnlyWatercraft/04530566/{}/rendering/00.png'

        self.root_dir = root_dir
        self.transform = transform

        # Read in annotations
        annotation_path = '/home/gitaar9/AI/TNO/check_classes/shapenet_class_info'
        with open(annotation_path) as f:
            self.annotations = [line.strip().split(", ") for line in f.readlines()]
            # 65f8bb163e351bfa9399f7cdb64577ad, speedboat, 04273569, 0

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_id, class_name, synset_id, label = self.annotations[idx]
        image_path = self.image_path.format(image_id)

        image = cv2.imread(image_path)
        image = Image.fromarray(image, 'RGB')

        if self.transform:
            image = self.transform(image)

        return image, int(label)

    def show_image(self, idx):
        image_id, class_name, synset_id, label = self.annotations[idx]
        print(f'Label of shown image is {class_name}({label})')
        self[idx][0].show()

# image_transform = transforms.Compose([
#     transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)
# ])
#
# ds = LabeledShapeNetDataset(transform=image_transform)
# ds.show_image(0)
# ds.show_image(150)
