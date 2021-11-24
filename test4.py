# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 04:00:17 2021

@author: Coding
"""

import cv2
import numpy as np
import math

data=[]

first = cv2.imread("pop_resize_009.jpg", cv2.IMREAD_COLOR) #이미지 열기
img = cv2.resize(first, dsize=(500, 500), interpolation=cv2.INTER_AREA) #이미지 resize
img2 = img.copy() #이미지 가공 전 copy
img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # 이미지 그레이스케일


adapt = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,19,3) # 이미지 이진화

denoise = cv2.medianBlur(adapt, 5) # 미디안 블러를 이용한 노이즈 제거

denoise2 = cv2.GaussianBlur(adapt, (5, 5), 0) # 가우시안 블러를 이용한 노이즈 제거


cv2.imshow("denoise", denoise)
cv2.waitKey(0) # 키를 누를 때까지 창을 띄워놓음
cv2.destroyAllWindows() # 키보드 자판의 아무 키나 누르면 창이 닫힘

#===== Mopolgy =============================================
# 모폴로지 연산(침식 or 팽창)
kernel_size_row = 3 # 이 값이 커질수록 침식 또는 팽창 효과가 강해집니다 (기본값 : 3)
kernel_size_col = 3 # 이 값이 커질수록 침식 또는 팽창 효과가 강해집니다 (기본값 : 3)

kernel = np.ones((kernel_size_row, kernel_size_col), np.uint8)

erosion_image = cv2.erode(denoise, kernel, iterations=1)  # 침식연산이 적용된 이미지
                                                           # iterations를 2 이상으로 설정하면 반복 적용됨
dilation_image = cv2.dilate(denoise, kernel, iterations=1)  # 팽창연산이 적용된 이미지
                                                             # iterations를 2 이상으로 설정하면 반복 적용됨
#==========================================================                                               

Reversed = cv2.bitwise_not(denoise) # 비트 반전을 통한 reverse

#cv2.imshow("Reversed", Reversed)
#cv2.waitKey(0) # 키를 누를 때까지 창을 띄워놓음
#cv2.destroyAllWindows() # 키보드 자판의 아무 키나 누르면 창이 닫힘

# 허프변환을 이용한 원 검출
try:
    circles = cv2.HoughCircles(Reversed, cv2.HOUGH_GRADIENT, 2, 20, \
             param1 = 500, param2 = 80, minRadius = 25, maxRadius = 40)
    circles = np.uint16(np.around(circles))
except:
    circles = 0
# for문을 반복하면서 원의 중심점을 기준으로 retangle 그림
for i in circles[0]:
    x = i[0].item()
    y = i[1].item()
    cv2.rectangle(img2,(x-30,y-30),(x+30,y+30),(0,255,0),2)
    img_crop = denoise[x-30:x+30 ,y-30:y+30 ] # 이미지 크롭
    pix_sum = np.sum(img_crop) # 픽셀값을 모두 더하기
    pix_sum_cnt = (int)(pix_sum/255) # 이미지에서 흰색 픽셀의 개수에 해당
    data.append(pix_sum_cnt)
    if 2399<=pix_sum_cnt:
        cv2.putText(img2, '1', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
sor = np.sort(data)
print(sor)

cv2.imshow("DetectionCircles", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
