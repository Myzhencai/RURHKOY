# Standard imports
import cv2
import numpy as np


# # Setup SimpleBlobDetector parameters.
# params = cv2.SimpleBlobDetector_Params()
#
# # Change thresholds
# params.minThreshold = 0;
# params.maxThreshold = 200;
#
# # Filter by Area.
# params.filterByArea = True
# params.minArea = 150
#
# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1
#
# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.87
#
# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01
#
# # Create a detector with the parameters
# ver = (cv2.__version__).split('.')
# if int(ver[0]) < 3 :
# 	detector = cv2.SimpleBlobDetector(params)
# else :
# 	detector = cv2.SimpleBlobDetector_create(params)
#
#
# # Read image
# im = cv2.imread("/home/gaofei/GFT-master2/test.jpg", cv2.IMREAD_GRAYSCALE)
#
# # Set up the detector with default parameters.
# # detector = cv2.SimpleBlobDetector()
#
# # Detect blobs.
# keypoints = detector.detect(im)
#
# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# # Show keypoints
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)


import cv2

image = cv2.imread("/home/gaofei/GFT-master2/each1.jpg", cv2.IMREAD_GRAYSCALE)

blur = cv2.medianBlur(image, 3)

gray = blur
thresh = cv2.threshold(gray,200,255, cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

min_area = 16
white_dots = []
for c in cnts:
    area = cv2.contourArea(c)
    if area > min_area:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 1)
        white_dots.append(c)

print(len(white_dots))
cv2.imshow('thresh', thresh)
cv2.imshow('image', image)
cv2.imwrite('image.png', image)
cv2.waitKey()