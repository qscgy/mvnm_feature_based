'''
Extracts features from training images and saves them in h5 files. Also saves a k-means classifier for use with
local features.
'''

from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import mahotas
import cv2
import h5py
import os, random
from process_local import fd_feature_hist, train_kmeans, fd_local_color_hist
from joblib import dump, load
from enum import Enum
import argparse

fixed_size = tuple((256, 256))
train_path = 'dataset/train'
save_folder = 'models/'
num_trees = 100
test_size = 0.10
seed = 9

class Feature(Enum):
    '''
    Enum to represent image features.
    '''
    COLOR_HIST = 1
    HARALICK = 2
    HU_MOMENTS = 3
    LOCAL_HIST = 4


def fd_hu_moments(image):
    '''
    Computes the Hu moments for an image. Specifically, the image is first converted to grayscale, then
    passed to cv2.HuMoments(). The output of this is flattened and then normalized.

    Arguments:
        image: the BGR image to process
    
    Returns:
        np.array: a normalized array of the Hu moments
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    '''
    Computes the first 13 Haralick features for an image.

    Arguments:
        image: the BGR image to process
    
    Returns:
        np.array: a normalized array of the first 13 Haralick features, averaged over the 4 directions
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)     # haralick() returns a 4x13 array, so we average the 4 values for each feature
    return haralick

def fd_histogram(image, bins=32):
    '''
    Computes a histogram for the hues in an image.

    Arguments:
        image: the BGR image to process
        bins: the number of bins to use for the histogram, default 32
    
    Returns:
        np.array: a histogram of the hue channel of image when converted to HSV
    '''
    # TODO change this to use mean and sd of the 3 color channels instead
    hist = cv2.calcHist([image], [1], None, [bins], [0, 256])   # calculate amount of green, since mangroves are green
    cv2.normalize(hist, hist)
    return hist.flatten()

def fd_statistics(image):
    '''
    Finds the mean and standard deviation of each BGR color channel in an image.
    '''
    blue = image[:,:,0]
    green = image[:,:,1]
    red = image[:,:,2]
    return np.array([np.mean(blue), np.mean(green), np.mean(red), np.std(blue), np.std(green), np.std(red)])

def calc_global_feature(file, color_model=None, orb_model=None,  local=False):
    '''
    Calculates an array of the global, and optionally local, features for an image, for use with ML models.

    Arguments:
        file: the relative path to the image to process
        model: the model to use for local feature classification, if used
        local: whether to compute local features, default True
    
    Returns:
        np.array: a 1D array of the features.
    '''
    image = cv2.imread(file)
    image = cv2.resize(image, fixed_size)   # force same size

    fv_haralick = fd_haralick(image)
    if local:
        fv_local = fd_feature_hist(image, orb_model)
        fv_local_color = fd_local_color_hist(image, model=color_model)
        return np.hstack([fv_haralick, fv_local, fv_local_color])
    fv_statistics = fd_statistics(image)
    return np.hstack([fv_statistics, fv_haralick])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--local', action='store_true', help='use local features')
    args = parser.parse_args()

    train_labels = os.listdir(train_path)   # use folder names as the class labels
    train_labels.sort()     # alphabetize for consistency
    print(train_labels)
    features = []
    labels = []
    file_list = []
    orb_kmeans = None
    color_kmeans = None
    if args.local:
        # Build the list of file paths to train the k-means classifier
        for r, d, f in os.walk(train_path):
            for file in f:
                file_list.append(os.path.join(r, file))
    
        # First, use k-means clustering to define our vocabulary
        orb_kmeans, color_kmeans = train_kmeans(file_list)

        print('[STATUS] Done training k-means, begin processing folders...')

    # Next, compile our training data, using global and optionally local features
    for training_name in train_labels:
        dir = os.path.join(train_path, training_name)   # get directory name
        current_label = training_name
        files = os.listdir(dir)
        for f in files:
            file = os.path.join(dir, f)
            global_feature = calc_global_feature(file, orb_model=orb_kmeans, color_model=color_kmeans, local=args.local)
            labels.append(current_label)
            features.append(global_feature)
        print ("[STATUS] processed folder: {}".format(current_label))

    print("[STATUS] feature vector size {}".format(np.array(features).shape))
    print("[STATUS] training Labels {}".format(np.array(labels).shape))
    target_names = np.unique(labels)
    print(target_names)
    print(train_labels)
    le = LabelEncoder()
    target = le.fit_transform(labels)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.array(features))
    print("[STATUS] training labels encoded...")

    print("[STATUS] target labels: {}".format(target))
    print("[STATUS] target labels shape: {}".format(target.shape))

    h5f_data = h5py.File('output/data.h5', 'w')
    h5f_data.create_dataset('dataset_1', data=scaled)
    h5f_label = h5py.File('output/labels.h5', 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))
    h5f_data.close()
    h5f_label.close()
    if args.local:
        print('[STATUS] Saved data, saving kmeans...')
        dump(orb_kmeans, os.path.join(save_folder, 'orb_kmeans.joblib'))
        dump(color_kmeans, os.path.join(save_folder, 'color_kmeans.joblib'))
    print('[STATUS] Saving scaler...')
    dump(scaler, os.path.join(save_folder, 'scaler.joblib'))
    dump(le, os.path.join(save_folder, 'le.joblib'))
    print("[STATUS] End of preprocessing...")

