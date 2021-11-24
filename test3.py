# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 18:05:49 2021

@author: Coding
"""
import cv2
import numpy as np
import math



#===== 이미지 열기 ========================================
first = cv2.imread("pop_ref_001.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(first, dsize=(500, 500), interpolation=cv2.INTER_AREA)
img2 = img.copy()
img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#===== 이미지 이진화 ======================================== 
threshold = 80 # 이 값보다 픽셀값이 크면 255, 작으면 0으로 변환

#===== Adeptive Treshold ===================================
adapt = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
cv2.THRESH_BINARY,15,1.9)

#===== 노이즈 재거 ===================================
denoise = cv2.bilateralFilter(adapt, -1, 10, 10)
denoise2 = cv2.bilateralFilter(denoise, -1, 50, 10)
denoise3 = cv2.medianBlur(denoise2, 5) # medianBlur를 이용해서 가공
# canny = cv2.Canny(adapt, 50, 150)

# 이미지 표시
#cv2.imshow("denoise", denoise)
#cv2.imshow("denoise2", denoise2)
cv2.imshow("denoise3", denoise3)
#cv2.imshow('GrayScale', img_gray) # 이진화 적용 전
#cv2.imshow('Binery(Adeptive)', adapt)
#cv2.imshow('canny', canny)
cv2.waitKey(0) # 키를 누를 때까지 창을 띄워놓음
cv2.destroyAllWindows() # 키보드 자판의 아무 키나 누르면 창이 닫힘

#===== Mopolgy =============================================
# 모폴로지 연산(침식 or 팽창)
#kernel_size_row = 5 # 이 값이 커질수록 침식 또는 팽창 효과가 강해집니다 (기본값 : 3)
#kernel_size_col = 5 # 이 값이 커질수록 침식 또는 팽창 효과가 강해집니다 (기본값 : 3)
# 3, 5, 7등으로 값을 바꿔서 실행해보세요. cv2.erode()나 cv.dilate를 반복하여 적용하는 것도 가능합니다.
#kernel = np.ones((kernel_size_row, kernel_size_col), np.uint8)

#erosion_image = cv2.erode(denoise3, kernel, iterations=1)  # 침식연산이 적용된 이미지
                                                           # iterations를 2 이상으로 설정하면 반복 적용됨
                                                           
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# 열림 연산 적용 ---②
gradient = cv2.morphologyEx(denoise3, cv2.MORPH_GRADIENT, k)

# 결과 출력
merged = np.hstack((denoise3, gradient))

#dilation_image = cv2.dilate(denoise3, kernel, iterations=1)  # 팽창연산이 적용된 이미지
                                                             # iterations를 2 이상으로 설정하면 반복 적용됨
#==========================================================


Reversed = cv2.bitwise_not(denoise3)
#cv2.imshow('Negative', Reversed)       # 색반전
#v2.waitKey(0)
#cv2.destroyAllWindows()
try:
    circles = cv2.HoughCircles(merged, cv2.HOUGH_GRADIENT, 2, 20, \
             param1 = 500, param2 = 80, minRadius = 25, maxRadius = 40)
    circles = np.uint16(np.around(circles))
except:
    circles = 0

for i in circles[0]:
    x = i[0].item()
    y = i[1].item()
    cv2.rectangle(img2,(x-25,y-25),(x+25,y+25),(0,255,0),2)
    img_crop = merged[x-25:x+25 ,y-25:y+25 ] # 이미지 크롭
    pix_sum = np.sum(img_crop) # 픽셀값을 모두 더하기
    pix_sum_cnt = (int)(pix_sum/255) # 이진화 이미지에서 흰색 픽셀의 개수에 해당
    print(pix_sum_cnt)
    if 1300<pix_sum_cnt<1480:
        #cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
        cv2.putText(img2, '1', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
    elif 1480<pix_sum_cnt<1530:
        cv2.putText(img2, '2', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
    elif 1530<pix_sum_cnt:
        cv2.putText(img2, '3', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)    


    
cv2.imshow('2.erosion_image', merged) # 침식
#cv2.imshow('3.dilation_image', dilation_image) # 팽창
cv2.imshow("DetectionCircles", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
