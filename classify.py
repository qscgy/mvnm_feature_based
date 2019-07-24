import h5py
import numpy as np
import os, shutil
import glob
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from preprocess_data import fd_haralick, fd_hu_moments, fd_histogram, calc_global_feature
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import argparse, random
from sklearn.svm import SVC
from joblib import load, dump
from pathlib import Path

# Site 1:
# 7199 m, 425 were nm
# 7177 nm, 265 were m

# Site 5:
# 2244 m, 121 were nm
# 3448 nm, 164 were m

# Site 7:
# 3484 m, 69 were nm
# 2688 nm, 757 were m

# Best result on site 8 was with global mean-std, Haralick, Hu moments

def grid_search_params(grid, svc, data, target):
    clf = GridSearchCV(svc, grid, cv=3)
    print('[STATUS] Beginning grid search...')
    clf.fit(data, target)
    print(clf.best_params_)

parser = argparse.ArgumentParser()
parser.add_argument('--rf', action='store_true', help='random forest')
parser.add_argument('--knn', action='store_true', help='k nearest neighbors')
parser.add_argument('--svm', action='store_true', help='support vector machine')
parser.add_argument('-l', '--local', action='store_true', help='use local features')
parser.add_argument('-s', '--sort', action='store_true', help='sort unlabeled tiles (MvNM)')
parser.add_argument('-v', '--validate', action='store_true', help='validate model on labeled tiles')
parser.add_argument('-i', '--input', default='dataset/test', help='the input directory')
parser.add_argument('-o', '--output', default='output/', help='the output directory for sorting')
parser.add_argument('--xv', action='store_true', help='10-fold cross-validate')
parser.add_argument('-r', '--retrain', action='store_true', help='retrain data set from h5 files')
parser.add_argument('--gs', action='store_true', help='grid search SVM params')
parser.add_argument('--show', action='store_true')
args = parser.parse_args()

fixed_size = tuple((256, 256))
train_path = 'dataset/train'
seed = 9
results = []
names = []
test_path = args.input
save_folder = 'models/'

h5f_data = h5py.File('output/data.h5', 'r')
h5f_label = h5py.File('output/labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

scaler = load(os.path.join(save_folder, 'scaler.joblib'))
le = load(os.path.join(save_folder, 'le.joblib'))

orb_kmeans = None
color_kmeans = None
if args.local:
    orb_kmeans = load(os.path.join(save_folder, 'orb_kmeans.joblib'))
    color_kmeans = load(os.path.join(save_folder, 'color_kmeans.joblib'))

# (train_data_global, test_data_global, train_labels_global, test_labels_global) = train_test_split(np.array(global_features),
#                                                                                           np.array(global_labels),
#                                                                                           test_size=0.2)

train_data_global = global_features
train_labels_global = global_labels

if args.retrain:
    if args.rf:
        clf = RandomForestClassifier(n_estimators=300, random_state=4)
    elif args.knn:
        clf = KNeighborsClassifier()
    elif args.svm:
        clf = SVC()
else:
    if args.rf:
        clf = load(os.path.join(save_folder, 'rf.joblib'))
    elif args.knn:
        clf = load(os.path.join(save_folder, 'knn.joblib'))
    elif args.svm:
        clf = load(os.path.join(save_folder, 'svm.joblib'))

clf.fit(train_data_global, train_labels_global)
# print("[STATUS] Fit KNN classifier to training data...")
# print("[STATUS] Beginning test...")
# print(test_data_global.shape)
# score = clf.score(test_data_global, test_labels_global)
# print("Score: {}".format(score))

train_labels = os.listdir(train_path)
out_root = args.output

if args.validate:
    test_dirs = os.listdir(test_path)
    test_dirs.sort()
    for d in test_dirs:
        test_images = glob.glob(os.path.join(test_path, d, '*.jpg',))
        im_count = 0
        correct = 0
        for file in test_images:
            image = cv2.imread(file)
            global_feature = calc_global_feature(file, orb_model=orb_kmeans, color_model=color_kmeans, local=args.local)
            global_feature = scaler.transform(global_feature.reshape(1, -1))
            prediction = clf.predict(global_feature)
            prediction_label = le.inverse_transform(prediction)[0]
            im_count += 1
            if d == prediction_label:
                correct += 1

            if args.show:
                cv2.putText(image, prediction_label, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
                cv2.imshow('image', image)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        print('Actual {}: {}/{} ({}%) labeled as {}'.format(d, correct, im_count, correct/im_count*100, d))
elif args.sort:
    print('[STATUS] Beginning sort...')
    test_images = glob.glob(os.path.join(test_path, '*.tif',))
    for file in test_images:
        print('[STATUS] Processing file {}'.format(file))
        image = cv2.imread(file)
        global_feature = calc_global_feature(file, orb_model=orb_kmeans, color_model=color_kmeans, local=args.local)
        prediction = clf.predict(global_feature.reshape(1, -1))
        prediction_label = le.inverse_transform(prediction)[0]
        imname = file.split('/')[-1]
        shutil.move(file, os.path.join(test_path, prediction_label, imname))
        print(prediction_label)
elif args.xv:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(clf, train_data_global, train_labels_global, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
    print(msg)
elif args.gs and args.svm:
    grid = {'gamma': [0.01, 0.02, 0.03]}
    grid_search_params(grid, svc=clf, data=global_features, target=global_labels)

if args.retrain:    # save model
    print('[STATUS] Saving model...')
    if args.rf:
        dump(clf, os.path.join(save_folder, 'rf.joblib'))
    elif args.knn:
        dump(clf, os.path.join(save_folder, 'knn.joblib'))
    elif args.svm:
        dump(clf, os.path.join(save_folder, 'svm.joblib'))
