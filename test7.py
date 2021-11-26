# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:17:11 2021

@author: Coding
"""
import cv2
import numpy as np
import math

data=[]

first = cv2.imread("pop_resize_009.jpg", cv2.IMREAD_COLOR) #이미지 열기
img = cv2.resize(first, dsize=(800, 800), interpolation=cv2.INTER_AREA) #이미지 resize
img2 = img.copy() #이미지 가공 전 copy
img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # 이미지 그레이스케일

edge = cv2.Canny(img_gray, 100, 100) # 캐니엣지

cv2.imshow("edge", edge)
cv2.waitKey(0) # 키를 누를 때까지 창을 띄워놓음
cv2.destroyAllWindows() # 키보드 자판의 아무 키나 누르면 창이 닫힘

#===== Mopolgy =============================================
# 모폴로지 연산(침식 or 팽창)
kernel_size_row = 1 # 이 값이 커질수록 침식 또는 팽창 효과가 강해집니다 (기본값 : 3)
kernel_size_col = 1 # 이 값이 커질수록 침식 또는 팽창 효과가 강해집니다 (기본값 : 3)

kernel = np.ones((kernel_size_row, kernel_size_col), np.uint8)

erosion_image = cv2.erode(edge, kernel, iterations=1)  # 침식연산이 적용된 이미지
                                                           # iterations를 2 이상으로 설정하면 반복 적용됨
dilation_image = cv2.dilate(edge, kernel, iterations=1)  # 팽창연산이 적용된 이미지
                                                             # iterations를 2 이상으로 설정하면 반복 적용됨
#==========================================================     

cv2.imshow("DetectionCircles", erosion_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

try:
    circles = cv2.HoughCircles(erosion_image, cv2.HOUGH_GRADIENT, 2, 20, \
             param1 = 500, param2 = 80, minRadius = 25, maxRadius = 40)
    circles = np.uint16(np.around(circles))
except:
    circles = 0                                          