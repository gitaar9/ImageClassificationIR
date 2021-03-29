import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from umap import UMAP

from ir_image_classification.data_visualization.util import load_data, get_random_permutation, data_to_df, \
    load_data_3d_features
from os import path


number_of_classes = 10 # 26

# Load np arrays of data/labels
# dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets/MARVEL_keras_ResNet152_224px'
# dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets/keras_ResNet152_224px_RGB'

dataset_path = '/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/extracted_datasets/Pix2Vox_side_other_view_256_ft_300'
X, y = load_data(dataset_path)
train_length = len(y)
test_X, test_Y = np.load(path.join(dataset_path, "test_features.npy")), np.load(path.join(dataset_path, "test_labels.npy"))
print(X.shape, test_X.shape)
X = np.concatenate([X, test_X.copy()], axis=0)
y = np.concatenate([y, test_Y.copy()], axis=0)
del test_X
del test_Y

print(X.shape, y.shape)

# dataset_path = '/home/gitaar9/AI/TNO/Pix2VoxPP'
# X, y = load_data_3d_features(dataset_path)
#
# print(resnet_X.shape)
# print(X.shape)
# print()
#
# X = np.concatenate([X, resnet_X.copy()], axis=1)
# del resnet_X
# del resnet_y

# Load data to pandas dataframe
df, feat_cols = data_to_df(X, y)
print('Size of the dataframe: {}'.format(df.shape))

# Shuffle the dataset
# rndperm = get_random_permutation(X.shape[0])

# Take only 10000 samples
# N = 30000
# df_subset = df.loc[rndperm[:N], :].copy()

data_df = df[feat_cols].values
label_df = df['label'].values


# Get 50 dimensions using PCA
n_dimensions = 1000
pca = PCA(n_components=n_dimensions, random_state=42)
pca_result = pca.fit_transform(data_df)
print(f'Cumulative explained variation for {n_dimensions} '
      f'principal components: {np.sum(pca.explained_variance_ratio_)}')

# Run the umap algorithm
#https://umap-learn.readthedocs.io/en/latest/basic_usage.html
reducer = UMAP(random_state=42)
reducer.fit(pca_result)
embedding = reducer.transform(pca_result)

# UMAP(a=None, angular_rp_forest=False, b=None,
#      force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
#      local_connectivity=1.0, low_memory=False, metric='euclidean',
#      metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
#      n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
#      output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
#      set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
#      target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
#      transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)

# Plot the embeddings
# plt.scatter(embedding[:, 0], embedding[:, 1], c=label_subset, cmap='Spectral', s=5)
# plt.gca().set_aspect('equal', 'datalim')
# plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
# plt.title('UMAP projection of the Digits dataset', fontsize=24)

# Plot the tsne
df['umap-one'] = embedding[:, 0]
df['umap-two'] = embedding[:, 1]
plt.figure(figsize=(20, 12))
sns.scatterplot(
    x="umap-one", y="umap-two",
    hue="label",
    palette=sns.color_palette("hls", number_of_classes),
    data=df[:train_length].sort_values(by=['label']),
    legend="full",
    alpha=.5,
    marker="."
)
plt.xlim(1, 13)
plt.ylim(0, 9)

plt.show()
plt.figure(figsize=(20, 12))

sns.scatterplot(
    x="umap-one", y="umap-two",
    hue="label",
    palette=sns.color_palette("hls", number_of_classes),
    data=df[:train_length].sort_values(by=['label']),
    legend="full",
    alpha=.5,
    marker="."
)

sns.scatterplot(
    x="umap-one", y="umap-two",
    hue="label",
    palette=sns.color_palette("hls", number_of_classes),
    data=df[train_length:].sort_values(by=['label']),
    alpha=1,
    marker="."
)
plt.xlim(1, 13)
plt.ylim(0, 9)
plt.show()
