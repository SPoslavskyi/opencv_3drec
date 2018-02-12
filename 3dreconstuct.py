import os
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np


def undistPhoto(fname):
    # The following parameters come from running the OpenCV calibrate.py script
    K = np.array([[2365.85284, 0, 1970.24146],[0, 2364.17864, 1497.37745], [0, 0, 1]])
    d = np.array([.0181517127, .134277087, 0, 0, 0])

    # read one of your images
    img = cv.imread(fname)
    h, w = img.shape[:2]

    # undistort
    newcamera, roi = cv.getOptimalNewCameraMatrix(K, d, (w,h), 0)

    newimg = cv.undistort(img, K, d, None, newcamera)

    udfname = "./output/udst_" + os.path.basename(fname)
    cv.imwrite(udfname, newimg)
    return udfname



img1 = cv.imread(undistPhoto("../sample/DJI_0359.JPG"),0)  #queryimage # left image
img2 = cv.imread(undistPhoto("../sample/DJI_0360.JPG"),0) #trainimage # right image
orb = cv.ORB_create()
# find the keypoints and descriptors

kp1 = orb.detect(img1, None)
kp1, des1 = orb.compute(img1, kp1)
kp2 = orb.detect(img2, None)
kp2, des2 = orb.compute(img2, kp2)
img3 = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
#plt.imshow(img3), plt.show()
# FLANN parameters
FLANN_INDEX_LSH = 6
index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper

for i,(m,n) in enumerate(matches, start=0):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()