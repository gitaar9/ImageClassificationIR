from os import path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def marvel_int_label_to_string(int_label):
    return {
        1: 'Container Ship',
        2: 'Bulk Carrier',
        3: 'Passengers Ship',
        4: 'Ro-ro/passenger Ship',
        5: 'Ro-ro Cargo',
        6: 'Tug',
        7: 'Vehicles Carrier',
        8: 'Reefer',
        9: 'Yacht',
        10: 'Sailing Vessel',
        11: 'Heavy Load Carrier',
        12: 'Wood Chips Carrier',
        13: 'Livestock Carrier',
        14: 'Fire Fighting Vessel',
        15: 'Patrol Vessel',
        16: 'Platform',
        17: 'Standby Safety Vessel',
        18: 'Combat Vessel',
        19: 'Training Ship',
        20: 'Icebreaker',
        21: 'Replenishment Vessel',
        22: 'Tankers',
        23: 'Fishing Vessels',
        24: 'Supply Vessels',
        25: 'Carrier/Floating',
        26: 'Dredgers',
    }[int_label]


def shapenet_int_label_to_string(int_label):
    return {
        0: 'speedboat',
        1: 'canoe',
        2: 'fishingboat',
        3: 'sailing boat',
        4: 'cargo ship',
        5: 'destroyer',
        6: 'cruise ship',
        7: 'aircraft carrier',
        8: 'submarine',
        9: 'outboard moter boat'
    }[int_label]


def load_data(dataset_path):
    return np.load(path.join(dataset_path, "train_features.npy")), np.load(path.join(dataset_path, "train_labels.npy"))


def load_data_3d_features(dataset_path, file_name="3d"):
    X = StandardScaler().fit_transform(np.load(path.join(dataset_path, f"{file_name}_features.npy")))
    y = np.load(path.join(dataset_path, f"{file_name}_labels.npy"))
    return X, y


def data_to_df(X, y):
    feat_cols = ['feature ' + str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: shapenet_int_label_to_string(i))
    return df, feat_cols


def get_random_permutation(size):
    np.random.seed(42)
    return np.random.permutation(size)


def get_pca_features(data, output_dimensions):
    pca = PCA(n_components=output_dimensions)
    pca_result = pca.fit_transform(data)
    print(f'Cumulative explained variation for {output_dimensions} '
          f'principal components: {np.sum(pca.explained_variance_ratio_)}')
    return pca_result
