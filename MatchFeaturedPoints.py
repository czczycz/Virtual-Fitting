import numpy as np
import cv2
import time
from matplotlib import pyplot as plt


# calculate runtime.
start = time.time()
# input queryImage
img1 = cv2.imread('90.jpg',0)#[348:786,335:552]
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
print(i.pt for i in kp1)
res1 = cv2.drawKeypoints(img1,kp1[:30],None,color=(255,0,0), flags=0)
# output featured points in queryImage
cv2.imwrite('test1.jpg',res1)

end1 = time.time()
# input trainImage and its keypoints
img2 = cv2.imread('105.jpg',0)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


# extract the locations of matched keypoints in both the images.
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])
print(src_pts)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
matchesMask = mask.ravel().tolist()
print(M)

h,w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
# draw matches in blue color.
draw_params = dict(matchColor = (255,0,0),
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

# output matched image.
res2 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

end2 = time.time()

cv2.imwrite("test2.jpg",res2)
t1 = int(1000*(end1-start))
t2 = int(1000*(end2-end1))
print("Runtime of finding featured points in '001.jpg' is {0}ms.\nRuntime of matching is {1}ms.\n".format(t1, t2))