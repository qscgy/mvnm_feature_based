import cv2
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import h5py
import os
from sklearn.preprocessing import LabelEncoder
from skimage.util.shape import view_as_windows
from functools import partial

def compute_local_color(image, im_size=256):
    '''
    Computes the array of color patch features.
    '''
    size = 32
    step = 32
    red = np.ascontiguousarray(image[:,:,2])
    green = np.ascontiguousarray(image[:,:,1])
    blue = np.ascontiguousarray(image[:,:,0])

    mean = partial(np.mean, axis=(-1, -2))     # so I don't have to type the same thing 6 times
    sd = partial(np.std, axis=(-1, -2))
    
    blue_w = view_as_windows(blue, (size, size), step=step)
    green_w = view_as_windows(green, (size, size), step=step)
    red_w = view_as_windows(red, (size, size), step=step)
    stats = np.asarray([mean(blue_w), mean(green_w), mean(red_w), sd(blue_w), sd(green_w), sd(red_w)])
    stats = np.concatenate(stats.T)
    return stats     # reshape the array to group each patch's features in the innermost array

def train_kmeans(im_files, num_cluster=100):
    '''
    Trains a k-means classifier on ORB local features for use with BoVW histograms.

    Arguments:
        im_files: a list of image files for training, with paths relative to the run directory
        num_cluster: the number of clusters in the classifier, default 100
    
    Returns:
        MinBatchKMeans: a trained k-means classifier for local ORB features
        MinBatchKMeans: a trained k-means classifier for local color features
    '''
    descriptors = []      # array of feature descriptors from test set
    orb = cv2.ORB_create()
    color_patches = []

    counter = 0
    for f in im_files:
        counter += 1
        image = cv2.imread(f)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, des = orb.detectAndCompute(gray, None)  # we only need the feature descriptors, not the actual keypoints
        if des is not None:
            descriptors.append(des)
        
        color_patches.append(compute_local_color(image))
        if counter % 100 == 0:
            print("Processed {}/{}".format(counter, len(im_files)))
        
    descriptors = np.asarray(descriptors)
    descriptors = np.concatenate(descriptors, axis=0)
    color_patches = np.asarray(color_patches)
    color_patches = np.concatenate(color_patches, axis=0)

    print("[STATUS] Training k-means...")
    kmeans_orb = MiniBatchKMeans(n_clusters=num_cluster, random_state=3).fit(descriptors)
    kmeans_color = MiniBatchKMeans(n_clusters=16, random_state=2).fit(color_patches)
    return kmeans_orb, kmeans_color

def process_patch(patch):
    '''
    Extracts the means and standard deviations for each color channel in an image patch.
    
    Arguments:
        patch: a section of an OpenCV image array
    
    Returns:
        stats: an array of the form [ch1mean, ch2mean, ch3mean, ch1std, ch2std, ch3std]
    '''
    stats = np.zeros(6)
    stats[0] = np.mean(patch[:,:,0])
    stats[1] = np.mean(patch[:,:,1])
    stats[2] = np.mean(patch[:,:,2])
    stats[3] = np.std(patch[:,:,0])
    stats[4] = np.std(patch[:,:,1])
    stats[5] = np.std(patch[:,:,2])
    return stats

def fd_feature_hist(image, model, bins=100):
    '''
    Computes the local feature histogram for an image. Features are detected with the ORB algorithm.

    Arguments:
        image: the BGR image to process
        model: the classifier to use for classifying local features
        bins: the number of bins for the histogram. Default is 100, but should be the number of clusters in `model`.
    
    Returns:
        np.array: a histogram of the local features
    '''
    orb = cv2.ORB_create()
    _, des = orb.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    if des is not None:
        fitted = model.predict(des)
        hist, _ = np.histogram(fitted, bins=bins)
        return hist
    return np.zeros(bins)

def fd_local_color_hist(image, model, bins=16):
    colors = compute_local_color(image)
    fitted = model.predict(colors)
    return np.histogram(fitted, bins=bins)[0]