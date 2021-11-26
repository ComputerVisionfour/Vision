# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:39:53 2021

@author: Coding
"""

import cv2
import numpy as np

data=[]
#===== 이미지 열기 ========================================
first = cv2.imread("pop_resize_009.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(first, dsize=(800, 800), interpolation=cv2.INTER_AREA)
img_c = img.copy()
img_gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY) # 이미지 그레이스케일

k = cv2.getStructuringElement(cv2.MORPH_RECT, (50,250))



threhold = 80 # 이 값보다 픽셀값이 크면 255, 작으면 0으로 변환

#===== Adeptive Treshold ===================================
adapt = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
cv2.THRESH_BINARY,15,3)

denoise = cv2.medianBlur(adapt, 3)

blackhat = cv2.morphologyEx(denoise, cv2.MORPH_BLACKHAT, k)

#===== Mopolgy =============================================
# 모폴로지 연산(침식 or 팽창)
kernel_size_row = 2 # 이 값이 커질수록 침식 또는 팽창 효과가 강해집니다 (기본값 : 3)
kernel_size_col = 2 # 이 값이 커질수록 침식 또는 팽창 효과가 강해집니다 (기본값 : 3)
# 3, 5, 7등으로 값을 바꿔서 실행해보세요. cv2.erode()나 cv.dilate를 반복하여 적용하는 것도 가능합니다.
kernel = np.ones((kernel_size_row, kernel_size_col), np.uint8)

erosion_image = cv2.erode(adapt, kernel, iterations=1)  # 침식연산이 적용된 이미지
                                                           # iterations를 2 이상으로 설정하면 반복 적용됨
dilation_image = cv2.dilate(adapt, kernel, iterations=1)  # 팽창연산이 적용된 이미지
                                                             # iterations를 2 이상으로 설정하면 반복 적용됨
#==========================================================

denoise2 = cv2.medianBlur(erosion_image, 3)


try:
    # small_circles = cv2.HoughCircles(denoise2, cv2.HOUGH_GRADIENT, 2, 50, \
    #          param1 = 200, param2 = 78, minRadius = 22, maxRadius = 247)
    # small_circles = cv2.HoughCircles(denoise2, cv2.HOUGH_GRADIENT, 2, 50, \
    #          param1 = 200, param2 = 70, minRadius = 22, maxRadius = 27)
    big_circles = cv2.HoughCircles(denoise2, cv2.HOUGH_GRADIENT, 2, 50, \
             param1 = 90, param2 =85, minRadius = 44, maxRadius = 56)
    # small_circles = np.uint16(np.around(small_circles))
    big_circles = np.uint16(np.around(big_circles))
    # for i in small_circles[0]:
    #     img_c = cv2.circle(img_c, (i[0].item(), i[1].item()), i[2].item(), (255, 255, 0), 2)
    for i in big_circles[0]:
        x = i[0].item()
        y = i[1].item()
        cv2.circle(blackhat, (i[0].item(), i[1].item()), i[2].item(), (255, 0, 255), 2)
        img_crop = blackhat[i] # 이미지 크롭
        pix_sum = np.sum(img_crop) # 픽셀값을 모두 더하기
        pix_sum_cnt = (int)(pix_sum/255) # 이미지에서 흰색 픽셀의 개수에 해당
        data.append(pix_sum_cnt)
        if 200<pix_sum_cnt<357:
            cv2.putText(img_c, '1', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
except:
    circles = 0
    
sor = np.sort(data)
print(sor)

cv2.imshow('2.erosion_image', blackhat) # 침식
# cv2.imshow('3.dilation_image', dilation_image) # 팽창
cv2.imshow("DetectionCircles", img_c)
cv2.waitKey(0)
cv2.destroyAllWindows()