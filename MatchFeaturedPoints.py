import numpy as np
import cv2
import time
from matplotlib import pyplot as plt


# calculate runtime.

# input queryImage
img1 = cv2.imread('90.jpg',0)#[347:429,227:312]
# Initiate SIFT detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with SIFT
start = time.time()
kp1, des1 = orb.detectAndCompute(img1,None)
end1 = time.time()
pts = np.float32([ [204,358],[204,511],[328,511],[328,358] ]).reshape(-1,1,2)
img1 = cv2.polylines(img1,[np.int32(pts)],True,255,3, cv2.LINE_AA)
res1 = cv2.drawKeypoints(img1,kp1,None,color=(255,0,0), flags=0)

# output featured points in queryImage
cv2.imwrite('featured.jpg',res1)


# input trainImage and its keypoints
img2 = cv2.imread('105.jpg',0)

end2 = time.time()
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
# store all the good matches as per Lowe's ratio test.
matches = sorted(matches, key = lambda x:x.distance)
good = []
#select area
for i in matches:
	if 204 <= kp1[i.queryIdx].pt[0] <= 328 and 358<= kp1[i.queryIdx].pt[1] <= 511 and len(good) < 50:
		good.append(i)

end3 = time.time()


# extract the locations of matched keypoints in both the images.
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
end4 = time.time()
matchesMask = mask.ravel().tolist()

#h,w = img1.shape
#pts = np.float32([ [204,358],[204,511],[328,511],[328,358] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

# draw matches in blue color.
draw_params = dict(matchColor = (255,0,0),
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

# output matched image.
res2 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

cv2.imwrite("matched.jpg",res2)
t1 = int(1000*(end1-start))
t2 = int(1000*(end3-end2))
t3 = int(1000*(end4-end3))


print("Runtime of finding feature points in p1 is {0}ms.\nRuntime of matching is {1}ms.\nRuntime of calculating homography is {2}ms.\n".format(t1, t2, t3))




