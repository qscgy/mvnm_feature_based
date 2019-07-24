import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from sklearn.decomposition import PCA

def colors():
    reds = {}
    greens = {}
    train_path = 'dataset/train'
    train_labels = os.listdir(train_path)
    for training_name in train_labels:
        dir = os.path.join(train_path, training_name)
        print(dir)
        files = os.listdir(dir)
        tmp_reds = []
        tmp_greens = []
        for f in files:
            file = os.path.join(dir, f)
            image = cv2.imread(file)
            tmp_reds.append(np.mean(image[:,:,2]))
            tmp_greens.append(np.mean(image[:,:,1]))
        reds[training_name] = tmp_reds
        greens[training_name] = tmp_greens

    plt.figure()
    ax = plt.subplot(111)
    for key in reds:
        ax.scatter(reds[key], greens[key], label=key)
    ax.legend()
    plt.show()

def pca_plot():
    h5f_data = h5py.File('output/data.h5', 'r')
    h5f_label = h5py.File('output/labels.h5', 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels = np.array(global_labels_string)

    h5f_data.close()
    h5f_label.close()

    pca = PCA(n_components=50)
    pca.fit(global_features)
    print(pca.explained_variance_)
    print(np.sum(pca.explained_variance_))

pca_plot()