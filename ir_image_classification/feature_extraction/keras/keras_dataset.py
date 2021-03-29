import glob
import os

import pandas as pd
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np

classes = ['Container Ship', 'Bulk Carrier', 'Passengers Ship', 'Ro-ro/passenger Ship', 'Ro-ro Cargo', 'Tug',
           'Vehicles Carrier', 'Reefer', 'Yacht', 'Sailing Vessel', 'Heavy Load Carrier', 'Wood Chips Carrier',
           'Livestock Carrier', 'Fire Fighting Vessel', 'Patrol Vessel', 'Platform', 'Standby Safety Vessel',
           'Combat Vessel', 'Training Ship', 'Icebreaker', 'Replenishment Vessel', 'Tankers', 'Fishing Vessels',
           'Supply Vessels', 'Carrier/Floating', 'Dredgers']


def df_to_histogram(df):
    hist = {}
    for (_, image_path, label) in df.itertuples(name=None):
        if label in hist:
            hist[label] += 1
        else:
            hist[label] = 1
    return hist


def limit_to_max_per_class(df, max_per_class):
    class_hist = {}
    new_df = pd.DataFrame({'paths': [], 'labels': []})
    for (_, image_path, label) in df.itertuples(name=None):
        if label in class_hist:
            class_hist[label] += 1
        else:
            class_hist[label] = 1
        if class_hist[label] < (max_per_class + 1):
            new_df = new_df.append({'paths': image_path, 'labels': label}, ignore_index=True)
    return new_df


def valid_image_paths_and_filtered_annotations(root_dir, is_train):
    images_paths = [path for path in glob.glob(os.path.join(root_dir, 'W*_1/*.jpg'))]
    print(f'Found {len(images_paths)} MARVEL images.')

    # Read in annotations
    with open(os.path.join(root_dir, 'VesselClassification.dat')) as f:
        annotations = [line.strip().split(",") for line in f.readlines()]
    image_ids = [image_id for image_id, _, _, _ in annotations]
    print(f'{len(image_ids)}/{len(set(image_ids))}')

    annotations = {
        image_id: {
            "is_train": True if set_index == '1' else False,
            "class_label": int(class_label),
            "class_label_name": class_label_name,
        } for image_id, set_index, class_label, class_label_name in annotations
    }

    # Remove annotations of not downloaded images + filter based on train/test set
    image_ids = {p.split('/')[-1][:-4] for p in images_paths}
    annotations = {
        key: value for key, value in annotations.items() if key in image_ids and value['is_train'] == is_train
    }

    # Filter image paths based on annotations
    images_paths = [p for p in images_paths if p.split('/')[-1][:-4] in annotations.keys()]
    print(f'{len(images_paths)} MARVEL images were loaded.')

    return images_paths, annotations


def marvel_dataframe(root_dir, is_train=True, cast_labels_to=str, max_images_per_class=None):
    image_paths, annotations = valid_image_paths_and_filtered_annotations(root_dir, is_train)

    # Retrieve labels from the annotation in the order of image_paths
    labels = []
    for p in image_paths:
        image_id = p.split('/')[-1][:-4]
        labels.append(cast_labels_to(annotations[image_id]['class_label']))

    if cast_labels_to == int:
        labels = [l - 1 for l in labels]

    df = pd.DataFrame({'paths': image_paths, 'labels': labels})
    if max_images_per_class:
        print(df_to_histogram(df))
        df = limit_to_max_per_class(df, max_images_per_class)
        print(df_to_histogram(df))
    return df


def marvel_side_other_view_dataframe(is_train=True, cast_labels_to=str):
    df = pd.read_hdf('side_view_images.hdf', 'df') if is_train else pd.read_hdf('other_view_images.hdf', 'df')
    df['labels'] = df['labels'].astype(cast_labels_to)

    if cast_labels_to == int:  # Repair some earlier mistake where I subtracted one from the label
        df['labels'] += 1
    return df


# marvel_root_dir = '/home/gitaar9/AI/TNO/marveldataset2016/'
#
# train_df = marvel_side_other_view_dataframe(is_train=False)
# print(train_df)
#
# datagen = ImageDataGenerator(
#     # shear_range=0.2,
#     # zoom_range=0.2,
#     # horizontal_flip=True,
#     # rotation_range=3,
#     # preprocessing_function=tf.keras.applications.resnet.preprocess_input
# )
#
# train_generator = datagen.flow_from_dataframe(
#     dataframe=train_df,
#     directory=None,
#     x_col="paths",
#     y_col="labels",
#     subset="training",
#     batch_size=32,
#     # seed=42,
#     shuffle=True,
#     class_mode="categorical",
#     target_size=(224, 224),
# )
# class_dict = {
#     1: 'Container Ship',
#     2: 'Bulk Carrier',
#     3: 'Passengers Ship',
#     4: 'Ro-ro/passenger Ship',
#     5: 'Ro-ro Cargo',
#     6: 'Tug',
#     7: 'Vehicles Carrier',
#     8: 'Reefer',
#     9: 'Yacht',
#     10: 'Sailing Vessel',
#     11: 'Heavy Load Carrier',
#     12: 'Wood Chips Carrier',
#     13: 'Livestock Carrier',
#     14: 'Fire Fighting Vessel',
#     15: 'Patrol Vessel',
#     16: 'Platform',
#     17: 'Standby Safety Vessel',
#     18: 'Combat Vessel',
#     19: 'Training Ship',
#     20: 'Icebreaker',
#     21: 'Replenishment Vessel',
#     22: 'Tankers',
#     23: 'Fishing Vessels',
#     24: 'Supply Vessels',
#     25: 'Carrier/Floating',
#     26: 'Dredgers',
# }
#
# for batch_images, batch_labels in train_generator:
#     print(batch_images.shape)
#     for i in range(32):
#         a = batch_images[i]
#         print(class_dict[np.where(batch_labels[i] == 1)[0][0] + 1])
#         a = np.interp(a, (a.min(), a.max()), (0, 255))
#         image = Image.fromarray(a.astype(np.uint8), 'RGB')
#         # image.show()
#         image.save(f'images_for_presentation/other_{i}.png')
#         # input('type enter')
#     break
