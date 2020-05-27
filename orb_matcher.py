import os
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np


img1 = cv.imread("./sample/1_1_s2.jpg", 0)  # queryimage # left image
img2 = cv.imread("./sample/1_3.JPG", 0)  # trainimage # right image
img2= cv.rotate(img2,cv.ROTATE_90_CLOCKWISE)

scale_percent = 25 # percent of original size
width = int(img2.shape[1] * scale_percent / 100)
height = int(img2.shape[0] * scale_percent / 100)
dim = (width, height)

img2 = cv.resize(img2, dim)

# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
print(orb.getMaxFeatures())
orb.setMaxFeatures(40)
kp2, des2 = orb.detectAndCompute(img2,None)
img3 = cv.drawKeypoints(img1,kp1,np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img4 = cv.drawKeypoints(img2,kp2,np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img5 = cv.drawMatches(img3,kp1,img4,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img5),plt.show()