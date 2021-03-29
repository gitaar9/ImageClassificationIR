import pandas as pd
import numpy as np
dataset_name = 'side_view_images'
df = pd.read_hdf(f'{dataset_name}.hdf', 'df')
lines = []
for idx, (path, label) in df.iterrows():
    lines.append(f"{path},{int(label)}\n")

print(lines[-20:])
np_labels = np.load('datasets/extracted_datasets/MARVEL_side_other_view_keras_ResNet152_224px/train_labels.npy')
print(np_labels[-20:] - 1)

# with open(f'{dataset_name}.txt', 'w') as f:
#     f.writelines(lines)

