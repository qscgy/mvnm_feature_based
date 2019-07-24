import cv2
import glob

for file in glob.glob('/home/sam/Downloads/Site4_120m_Tiles/*.tif'):
    image = cv2.imread(file)
    cv2.imwrite(file[:-3]+'jpg', image)