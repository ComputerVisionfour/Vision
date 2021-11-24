# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:03:34 2021

@author: Coding
"""
import cv2
import numpy as np
import math

img = cv2.imread("pop_ref_001.jpg", cv2.IMREAD_COLOR)
img2 = cv2.resize(img, dsize=(1280, 720), interpolation=cv2.INTER_AREA)
img_copy = img2.copy()
img_gray = cv2.cvtColor(img_copy,cv2.COLOR_BGR2GRAY)

threshold = 80

th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,1.9)

Reversed = cv2.bitwise_not(th)
cv2.imshow('Negative', Reversed)
cv2.waitKey(0)
cv2.destroyAllWindows()

try:
    circles = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,1,20,param1=150, param2 = 50, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
except:
    circles = 0
    
    
for i in circles[0,:]:
    cv2.circle(img2, (i[0],i[1]),i[2], (255,255,0), 2)
    a = i[0]
    b = i[1]
    print(a,b)
    
cv2.imshow('detected circles',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()