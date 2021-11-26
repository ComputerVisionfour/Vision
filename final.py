# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 03:18:30 2021

@author: Coding
"""

import cv2 # opencv import
import numpy as np # numpy import

data=[] # 흰색 픽셀 데이터 리스트 생성

img = cv2.imread("pop_resize_027.jpg", cv2.IMREAD_COLOR)
img_resized = cv2.resize(img , dsize=(800, 800), interpolation=cv2.INTER_AREA)
img_c = img_resized.copy()

#================== 이미지 선명도 증가 함수===========
def img_Contrast(img_c):
    lab = cv2.cvtColor(img_c, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(9, 9))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final
img_c=  img_Contrast(img_c)
#=====================================================


img_gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY) # 이미지 그레이스케일

img_stratched = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX) # 히스토그램 스트레칭은 NORM_MINMAX

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img2 = clahe.apply(img_stratched)

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (85,85))
blackhat = cv2.morphologyEx(img2, cv2.MORPH_TOPHAT, kernel1)

img_binary = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,3)


#===== Mopolgy2 =============================================
# 모폴로지 연산(침식 or 팽창)
kernel_size_row = 3 # 이 값이 커질수록 침식 또는 팽창 효과가 강해집니다 (기본값 : 3)
kernel_size_col = 3 # 이 값이 커질수록 침식 또는 팽창 효과가 강해집니다 (기본값 : 3)
# 3, 5, 7등으로 값을 바꿔서 실행해보세요. cv2.erode()나 cv.dilate를 반복하여 적용하는 것도 가능합니다.
kernel = np.ones((kernel_size_row, kernel_size_col), np.uint8)

erosion_image = cv2.erode(img_binary, kernel, iterations=1)  # 침식연산이 적용된 이미지
                                                           # iterations를 2 이상으로 설정하면 반복
dilation_image = cv2.dilate(img_binary, kernel, iterations=1)  # 팽창연산이 적용된 이미지
                                                             # iterations를 2 이상으로 설정하면 반복 적용됨
#==========================================================

denoise3 = cv2.medianBlur(erosion_image, 1)

try:
    circles = cv2.HoughCircles(denoise3, cv2.HOUGH_GRADIENT, 2, 90, param1 = 200, param2 =85, minRadius = 44, maxRadius = 63)
    circles = np.uint16(np.around(circles))
    for i in circles[0]:
         #x,y는 원의 중심값
        x = i[0].item()
        y = i[1].item()
        img_circle = cv2.circle(img_c, (x,y), 50, (255, 255, 255), 2)
        #cv2.rectangle(img_c,(x-25,y-25),(x+25,y+25),(0,0,0),2)
        img_crop = blackhat[y-25:y+25,x-25:x+25] # 이미지 크롭
        pix_sum = np.sum(img_crop) # 픽셀값을 모두 더하기
        pix_sum_cnt = int((pix_sum/255)) # 이미지에서 흰색 픽셀의 개수에 해당
        data.append(pix_sum_cnt) # 픽셀수를 다 더해서 data 리스트에 하나씩 담음
        if 850<=pix_sum_cnt<1500:
            cv2.putText(img_c, 'error', (x-20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 3)
        elif 850>pix_sum_cnt:
            cv2.putText(img_c, 'normal', (x-20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 3)
        elif 1500<pix_sum_cnt:
            cv2.putText(img_c, 'normal', (x-20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 3)
except:
    circles = 0
#=====================================================================

sor = np.sort(data)
print(sor) #sor 데이터 리스트 출력

#===================================최종 결과 확인====================
cv2.imshow("Result1", blackhat)
cv2.imshow("Result2", img_c)
cv2.waitKey(0)
cv2.destroyAllWindows()
#====================================================================