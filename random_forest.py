import h5py
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
from process_global import fd_haralick, fd_hu_moments, fd_histogram
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

fixed_size = tuple((256, 256))
train_path = 'dataset/train'
test_size = 0.30
seed = 9
num_trees = 100
results = []
names = []
test_path = "dataset/test"

h5f_data = h5py.File('output/data.h5', 'r')
h5f_label = h5py.File('output/labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

(train_data_global, test_data_global, train_labels_global, test_labels_global) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size)

clf = RandomForestClassifier(n_estimators=num_trees)
