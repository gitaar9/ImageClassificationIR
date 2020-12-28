import time

import numpy as np
import os
import numpy as np
import pandas as pd
# from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from ir_image_classification.data_visualization.util import load_data, data_to_df, get_random_permutation, \
    load_data_3d_features

# Load np arrays of data/labels
# dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets/labeled_shapenet_keras_ResNet152_224px'
# # dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets/MARVEL_keras_ResNet152_224px'
# # dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets/keras_ResNet152_224px_RGB'
# X, y = load_data(dataset_path)

dataset_path = '/home/gitaar9/AI/TNO/Pix2VoxPP'
X, y = load_data_3d_features(dataset_path, "gt")

print(np.max(y))  # Amount of colors for the plots
print(X.shape)
print(y.shape)

# Load data to pandas dataframe
df, feat_cols = data_to_df(X, y)
print('Size of the dataframe: {}'.format(df.shape))

# Shuffle the dataset
rndperm = get_random_permutation(df.shape[0])

# Take only 10000 samples
N = 30000
df_subset = df.loc[rndperm[:N], :].copy()
data_subset = df_subset[feat_cols].values

# Get 50 dimensions using PCA
n_dimensions = 250
pca = PCA(n_components=n_dimensions)
pca_result = pca.fit_transform(data_subset)
print(f'Cumulative explained variation for {n_dimensions} '
      f'principal components: {np.sum(pca.explained_variance_ratio_)}')


# TSNE
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=45, n_iter=500, n_jobs=6)
tsne_results = tsne.fit_transform(pca_result)  # replace with data_subset to perform tsne without pca
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# Plot the tsne
df_subset['tsne-2d-one'] = tsne_results[:, 0]
df_subset['tsne-2d-two'] = tsne_results[:, 1]
plt.figure(figsize=(20, 12))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="label",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=.5,
    style="label"
)
plt.show()


