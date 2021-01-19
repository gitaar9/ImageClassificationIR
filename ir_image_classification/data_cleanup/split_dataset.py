import os
import time

import pandas as pd
import cv2
from keras_preprocessing.image import ImageDataGenerator

from ir_image_classification.feature_extraction.keras.keras_dataset import marvel_dataframe, classes
import numpy as np


# Load the existing panda dfs for sideview and otherviews
try:
    side_view_images = pd.read_hdf('side_view_images.hdf', 'df')
    other_view_images = pd.read_hdf('other_view_images.hdf', 'df')
    skipped_images = pd.read_hdf('skipped_images.hdf', 'df')
    print(f"Loaded {len(side_view_images)} sideview images + {len(other_view_images)} otherview images "
          f"+ {len(skipped_images)} skipped images")
except FileNotFoundError:
    print("Comment out the exit() if you want to reload your dataframes.")
    exit()
    side_view_images = pd.DataFrame({'paths': [], 'labels': []})
    other_view_images = pd.DataFrame({'paths': [], 'labels': []})
    skipped_images = pd.DataFrame({'paths': [], 'labels': []})

already_checked_paths = set(side_view_images['paths']).union(set(other_view_images['paths'])).union(
    set(skipped_images['paths'])
)

# Load the data from the standard marvel dataset
allowed_classes = {0, 5, 8, 9, 17, 19, 21, 22}
marvel_root_dir = '/home/gitaar9/AI/TNO/marveldataset2016/'
train_df = marvel_dataframe(marvel_root_dir, is_train=True, cast_labels_to=int)
train_df = train_df[train_df['labels'].isin(allowed_classes)]  # Remove not allowed classes
train_df = train_df[~train_df['paths'].isin(already_checked_paths)]  # Remove already done images
train_df = train_df.sample(frac=1)  # Shuffle the df


# Loop for splitting the images in sideview or otherview datasets
start_time = time.time()
total_images_added = 0.1
cv2.namedWindow("Split images")
for (_, image_path, label) in train_df.itertuples(name=None):
    print("Images per minute: {:.2f}".format(60 / ((time.time() - start_time) / total_images_added)))
    print(f"\nClass is {classes[label]}. 0(Sideview), 1(Otherview), 2(Skip), 3(Quit program)")
    print(image_path, label)

    image = cv2.imread(image_path)  # Load the image
    cv2.imshow("Split images", image.astype(np.uint8))  # Show the image
    k = chr(cv2.waitKey(100000))  # Wait for keyboard input

    if k == '0':  # toggle current image
        total_images_added += 1
        print('Add to sideview dataset')
        side_view_images = side_view_images.append({'paths': image_path, 'labels': label}, ignore_index=True)
        continue
    elif k == '1':
        total_images_added += 1
        print('Add to otherview dataset')
        other_view_images = other_view_images.append({'paths': image_path, 'labels': label}, ignore_index=True)
        continue
    elif k == '2':
        print('Skipped image')
        skipped_images = skipped_images.append({'paths': image_path, 'labels': label}, ignore_index=True)
        continue
    elif k == '3':
        print('Exiting...')
        break

# Save the dataframes in the end as well
side_view_images.to_hdf('side_view_images.hdf', 'df', mode='w')
other_view_images.to_hdf('other_view_images.hdf', 'df', mode='w')
skipped_images.to_hdf('skipped_images.hdf', 'df', mode='w')

cv2.destroyAllWindows()
