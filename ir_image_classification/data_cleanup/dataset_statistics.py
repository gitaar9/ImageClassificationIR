import os
import time

import pandas as pd
import cv2
from keras_preprocessing.image import ImageDataGenerator

from ir_image_classification.feature_extraction.keras.keras_dataset import marvel_dataframe, classes, df_to_histogram
import numpy as np


side_view_images = pd.read_hdf('side_view_images.hdf', 'df')
other_view_images = pd.read_hdf('other_view_images.hdf', 'df')
skipped_images = pd.read_hdf('skipped_images.hdf', 'df')
print(f"Loaded {len(side_view_images)} sideview images + {len(other_view_images)} otherview images "
      f"+ {len(skipped_images)} skipped images")


print("Side view:", df_to_histogram(side_view_images))
print("Other view:", df_to_histogram(other_view_images))

print(len(side_view_images.drop_duplicates(subset='paths', keep="last")))
print(len(other_view_images.drop_duplicates(subset='paths', keep="last")))
