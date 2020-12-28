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


from ir_image_classification.data_visualization.util import load_data, data_to_df, get_random_permutation, \
    load_data_3d_features

# Load np arrays of data/labels
# dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets/MARVEL_keras_ResNet152_224px'
# # dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets/keras_ResNet152_224px_RGB'
#
# X, y = load_data(dataset_path)

dataset_path = '/home/gitaar9/AI/TNO/Pix2VoxPP'
X, y = load_data_3d_features(dataset_path)

print(np.max(y))  # Amount of colors for the plots
print(X.shape)
print(y.shape)

# Load data to pandas dataframe
df, feat_cols = data_to_df(X, y)
print('Size of the dataframe: {}'.format(df.shape))

# Shuffle the dataset
rndperm = get_random_permutation(df.shape[0])

# PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:, 0]
df['pca-two'] = pca_result[:, 1]
df['pca-three'] = pca_result[:, 2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# Plot pca
plt.figure(figsize=(16, 10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 26),
    data=df.loc[rndperm, :],
    legend="full",
    alpha=0.3
)
plt.show()

# 3D plot pca
ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm, :]["pca-one"],
    ys=df.loc[rndperm, :]["pca-two"],
    zs=df.loc[rndperm, :]["pca-three"],
    c=df.loc[rndperm, :]["y"],
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()
