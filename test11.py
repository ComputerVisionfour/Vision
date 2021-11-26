# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 21:43:28 2021

@author: Coding
"""
import cv2
import numpy as np

img1 = cv2.imread('pop_resize_009.jpg')
img = cv2.imread('pop_resize_009.jpg',0)
img_first = cv2.resize(img, dsize=(800, 800), interpolation=cv2.INTER_AREA)
img = img_first
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# Create mask
height,width = img.shape
mask = np.zeros((height,width), np.uint8)
edges = cv2.Canny(thresh, 100, 200)
#cv2.imshow('detected ',gray)
cimg=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1 = 100, param2 = 30, minRadius = 0, maxRadius = 0)
for i in circles[0]:
    # Draw on mask
    cv2.circle(mask,(int(i[0].item()),int(i[1].item())),50,(255,255,255),thickness=-1)

# Copy that image using that mask
masked_data = cv2.bitwise_and(img, img, mask=mask)

# Apply Threshold
_,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

# Find Contour
contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
x,y,w,h = cv2.boundingRect(contours[0])

# Crop masked_data
crop = masked_data[y:y+h,x:x+w]

#Code to close Window
cv2.imshow('detected Edge',mask)
#cv2.imshow('Cropped Eye',crop)
cv2.waitKey(0)
cv2.destroyAllWindows()