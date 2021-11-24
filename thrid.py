# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:13:44 2021

@author: Coding
"""
import numpy as np
import cv2

#이미지 한번 가공
src = cv2.imread("pop_ref_001.jpg")
img = cv2.resize(src, dsize=(1280, 720), interpolation=cv2.INTER_AREA)
img2 = img.copy() # 이미지 가공 전 카피
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#엣지 검출
dst = cv2.Canny(gray, 10 , 150)

#이진화
th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,1,9)

cv2.imshow("result", th)
cv2.waitKey(0)
cv2.destroyAllWindows()

#색반전
Reversed = cv2.bitwise_not(th)
cv2.imshow('Negative', Reversed)
cv2.waitKey(0)
cv2.destroyAllWindows()

#이미지 크롭 파트
#이미지를 크롭해서 하얀색 픽셀수를 계산

#img_crop = binary[120:180 , 200:300] # 이미지 크롭
#cropped_img = img[y: y + h, x: x + w] # 이미지 크롭 인자


try:
    circles = cv2.HoughCircles(Reversed,cv2.HOUGH_GRADIENT,1,20,param1=500, param2 = 39, minRadius=40, maxRadius=57)
    
    circles = np.uint16(np.around(circles))
except:
    circles = 0
    
    
for i in circles[0]:
    circle = cv2.circle(img,(i[0].item(),i[1].item()),i[2].item(),(255,255,0),2)

cv2.imshow("result", circle)
cv2.waitKey(0)
cv2.destroyAllWindows()
    

#pix_sum = np.sum(img_crop) # 픽셀값을 모두 더하기
#pix_sum_cnt = (int)(pix_sum/255) # 이진화 이미지에서 흰색 픽셀의 개수에 해당

#print(pix_sum_cnt)

# 원검출에 대한 코드/ 허프 변환을 이용하였음.
#circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,35,param1=70,param2=30,minRadius=30,maxRadius=50)
#circles = np.uint16(np.around(circles))
#for i in circles[0,:]:
    #cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)

#circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 100, param1 = 150, param2 = 10, minRadius = 80, maxRadius = 120)

#for i in circles[0]:
    #cv2.circle(img, (i[0], i[1]), i[2], (255, 255, 255), 5)

#cv2.imshow('image(crop)', img_crop) # 이진화 후 크롭된 이미지
#cv2.imshow("result", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()