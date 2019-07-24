import shutil
import sys
import glob

filedir = sys.argv[1]   # the directory where the files are (relative)
files = glob.glob(filedir + '/*.jpg')   # get all of the file paths
for f in files:
    dest = f[:-4] + 'n3.jpg'
    print(dest)
    shutil.move(f, dest)