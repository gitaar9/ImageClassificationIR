import glob
import os

import pandas as pd


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


class MARVELDataset:
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

    def get_labels(self):
        labels = []
        for p in self.images_paths:
            image_id = p.split('/')[-1][:-4]
            labels.append(str(self.annotations[image_id]['class_label']))
        return labels

    def get_dataframe(self):
        paths = self.images_paths
        labels = self.get_labels()
        return pd.DataFrame({'paths': paths, 'labels': labels})


# train_ds = MARVELDataset('/home/gitaar9/AI/TNO/marveldataset2016/')
# test_ds = MARVELDataset('/home/gitaar9/AI/TNO/marveldataset2016/', is_train=False)
#
# train_df = train_ds.get_dataframe()
# print(train_df)
# datagen = ImageDataGenerator(
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     preprocessing_function=tf.keras.applications.resnet.preprocess_input
# )
#
# train_generator = datagen.flow_from_dataframe(
#     dataframe=train_df,
#     directory=None,
#     x_col="paths",
#     y_col="labels",
#     subset="training",
#     batch_size=32,
#     seed=42,
#     shuffle=False,
#     class_mode="categorical",
#     target_size=(224, 224),
# )
#
# # train =
#
# # print(len(train))
# print(len(train_generator))
# for batch_images, batch_labels in train_generator:
#     print(batch_images.shape)
#     for i in range(3):
#         a = batch_images[i]
#         a = np.interp(a, (a.min(), a.max()), (0, 255))
#         image = Image.fromarray(a.astype(np.uint8), 'RGB')
#         image.show()
#     break