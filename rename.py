import shutil
import sys
import glob

filedir = sys.argv[1]   # the directory where the files are (relative)
files = glob.glob(filedir + '/*.jpg')   # get all of the file paths
for f in files:
    if f[-5]=='w':
        dest = 'dataset/water/' + f.split('/')[-1]
        print(dest)
        shutil.move(f, dest)