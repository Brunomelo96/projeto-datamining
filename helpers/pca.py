from typing import Tuple
from PIL import Image
import os
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
from helpers.plotters import plot_pca_3d


def get_features_from_img(img: Image.Image):
    img = np.array(img)

    return img.flatten()


def apply_pca(data, n_components=3) -> Tuple[np.ndarray, PCA]:
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(
        f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")

    return transformed, pca


def get_pca_from_dir(input_path, output_path, n_components=3):
    features_list = []
    labels = []
    for entry in os.scandir(input_path):
        img = Image.open(entry.path).convert('RGB')

        image_name, _ = entry.path.rsplit('.', 1)
        print(image_name)
        _, label = image_name.rsplit('_', 1)

        features = get_features_from_img(img)
        features_list.append(features)
        labels.append(int(label))

    transformed, pca = apply_pca(features_list, n_components)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(
        f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")

    plot_pca_3d(output_path, transformed, labels)
