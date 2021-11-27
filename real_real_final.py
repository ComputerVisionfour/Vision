# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 03:18:30 2021

@author: Coding
"""
#========================================Library import========
import cv2 # opencv import
import numpy as np # numpy import
#========================================Library import========

# 이미지 열기
filename = "pop_resize_013.jpg" # 입력 이미지 파일명을 적으세요.

#========================================create List data======
prediction = [0]*28 # 각 푸시팝의 상태를 데이터 리스트로 저장
data=[] # 흰색 픽셀 데이터 리스트 생성
#========================================create List data======

#========================================load image============
first = cv2.imread(filename, cv2.IMREAD_COLOR)
img = cv2.resize(first, dsize=(800, 800), interpolation=cv2.INTER_AREA)
first_copy = img.copy()
hsv = cv2.cvtColor(first_copy, cv2.COLOR_BGR2HSV)
#===== 이미지 회전 =============================================
Degree=0
while True:
    if (hsv[0,0,0]>115 and hsv[0,0,0]<150):
        break
    height, width, channel = img.shape
    matrix = cv2.getRotationMatrix2D((width/2, height/2), 45*Degree, 1)
    Warp = cv2.warpAffine(first_copy, matrix, (width, height))         #돌린사진
    Crop = Warp[5:130 , 250:550]
    resize = cv2.resize(Crop, (1,1))
    hsv = cv2.cvtColor(resize, cv2.COLOR_BGR2HSV)
    Degree+=1
#================== 이미지 선명도 증가 함수===========
def img_Contrast(Warp):
    lab = cv2.cvtColor(Warp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(9, 9))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final
img_vis = img_Contrast(Warp)
#=====================================================

#=========================================이미지 가공파트==========
img_gray = cv2.cvtColor(img_vis, cv2.COLOR_BGR2GRAY) # 이미지 그레이스케일

img_stratched = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX) # 히스토그램 스트레칭은 NORM_MINMAX

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img2 = clahe.apply(img_stratched)
#========================TOPHAT filter를 통해서 밝은 부분을 강조======
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (85,85))
tophat = cv2.morphologyEx(img2, cv2.MORPH_TOPHAT, kernel1)
#====================================================================

img_binary = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,3)


#===== Mopolgy2 =============================================
# 모폴로지 연산(침식 or 팽창)
kernel_size_row = 3 # 이 값이 커질수록 침식 또는 팽창 효과가 강해집니다 (기본값 : 3)
kernel_size_col = 3 # 이 값이 커질수록 침식 또는 팽창 효과가 강해집니다 (기본값 : 3)

kernel = np.ones((kernel_size_row, kernel_size_col), np.uint8)

erosion_image = cv2.erode(img_binary, kernel, iterations=1)  # 침식연산이 적용된 이미지
                                                           # iterations를 2 이상으로 설정하면 반복
dilation_image = cv2.dilate(img_binary, kernel, iterations=1)  # 팽창연산이 적용된 이미지
                                                             # iterations를 2 이상으로 설정하면 반복 적용됨
#==========================================================

denoise3 = cv2.medianBlur(erosion_image, 1) # medianBlur를 통해서 이미지의 노이즈를 제거

#=========================================이미지 가공파트==========

circles = cv2.HoughCircles(denoise3, cv2.HOUGH_GRADIENT, 2, 90, param1 = 200, param2 =85, minRadius = 44, maxRadius = 63)
circles = np.uint16(np.around(circles))
for i in circles[0]:
    #x,y는 원의 중심값
    x = i[0].item() # 원 중심의 x좌표
    y = i[1].item() # 원 중심의 y좌표
    img_circle = cv2.circle(img_vis, (x,y), 50, (255, 255, 255), 2) # 원을 그림
    img_crop = tophat[y-25:y+25,x-25:x+25] # 이미지 크롭
    pix_sum = np.sum(img_crop) # 픽셀값을 모두 더하기
    pix_sum_cnt = int((pix_sum/255)) # 이미지에서 흰색 픽셀의 개수에 해당
    data.append(pix_sum_cnt) # 픽셀수를 다 더해서 data 리스트에 하나씩 담음
    if 850<=pix_sum_cnt<1600: # 불량의 경우
#=====================================조건문에 맞춰서 text를 중심에 넣고 state변수에 각 푸시팝의 state를 추가====
        cv2.putText(img_vis, 'error', (x-20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 3)
        if 200 > y:
            if 200 < x < 300:
                prediction[0] = 3
            if 300 < x < 450:
                prediction[1] = 3
            if 450 < x:
                prediction[2] = 3
        if 200 < y < 300:
            if 200 > x:
                prediction[3] = 3
            if 200 < x < 300:
                prediction[4] = 3
            if 300 < x < 500:
                prediction[5] = 3
            if 500 < x < 600:
                prediction[6] = 3
            if 600 < x:
                prediction[7] = 3
        if 300 < y < 400:
            if 200 > x:
                prediction[8] = 3
            if 200 < x < 300:
                prediction[9] = 3
            if 300 < x < 400:
                prediction[10] = 3
            if 400 < x < 500:
                prediction[11] = 3
            if 500 < x < 600:
                prediction[12] = 3
            if 600 < x :
                prediction[13] = 3
        if 400 < y < 500:
            if 200 > x:
                prediction[14] = 3
            if 200 < x < 300:
                prediction[15] = 3
            if 300 < x < 400:
                prediction[16] = 3
            if 400 < x < 500:
                prediction[17] = 3
            if 500 < x < 600:
                prediction[18] = 3
            if 600 < x :
                prediction[19] = 3
        if 500 < y < 650:
            if 200 > x:
                prediction[20] = 3
            if 200 < x < 300:
                prediction[21] = 3
            if 300 < x < 500:
                prediction[22] = 3
            if 500 < x < 600:
                prediction[23] = 3
            if 600 < x:
                prediction[24] = 3
        if 650 < y :
            if 200 < x < 300:
                prediction[25] = 3
            if 300 < x < 450:
                prediction[26] = 3
            if 450 < x:
                prediction[27] = 3
    elif 850>pix_sum_cnt:   # 눌림의 경우
        cv2.putText(img_vis, 'normal', (x-20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 3)
        if 200 > y:
            if 200 < x < 300:
                prediction[0] = 2
            if 300 < x < 450:
                prediction[1] = 2
            if 450 < x:
                prediction[2] = 2
        if 200 < y < 300:
            if 200 > x:
                prediction[3] = 2
            if 200 < x < 300:
                prediction[4] = 2
            if 300 < x < 500:
                prediction[5] = 2
            if 500 < x < 600:
                prediction[6] = 2
            if 600 < x:
                prediction[7] = 2
        if 300 < y < 400:
            if 200 > x:
                prediction[8] = 2
            if 200 < x < 300:
                prediction[9] = 2
            if 300 < x < 400:
                prediction[10] = 2
            if 400 < x < 500:
                prediction[11] = 2
            if 500 < x < 600:
                prediction[12] = 2
            if 600 < x :
                prediction[13] = 2
        if 400 < y < 500:
            if 200 > x:
                prediction[14] = 2
            if 200 < x < 300:
                prediction[15] = 2
            if 300 < x < 400:
                prediction[16] = 2
            if 400 < x < 500:
                prediction[17] = 2
            if 500 < x < 600:
                prediction[18] = 2
            if 600 < x :
                prediction[19] = 2
        if 500 < y < 650:
            if 200 > x:
                prediction[20] = 2
            if 200 < x < 300:
                prediction[21] = 2
            if 300 < x < 500:
                prediction[22] = 2
            if 500 < x < 600:
                prediction[23] = 2
            if 600 < x:
                prediction[24] = 2
        if 650 < y :
            if 200 < x < 300:
                prediction[25] = 2
            if 300 < x < 450:
                prediction[26] = 2
            if 450 < x:
                prediction[27] = 2
    elif 1600<pix_sum_cnt:  # 볼록의 경우
        cv2.putText(img_vis, 'normal', (x-20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 3)
        if 200 > y :
            if 200 < x < 300:
                prediction[0] = 1
            if 300 < x < 450:
                prediction[1] = 1
            if 450 < x:
                prediction[2] = 1
        if 200 < y < 300:
            if 200 > x:
                prediction[3] = 1
            if 200 < x < 300:
                prediction[4] = 1
            if 300 < x < 500:
                prediction[5] = 1
            if 500 < x < 600:
                prediction[6] = 1
            if 600 < x:
                prediction[7] = 1
        if 300 < y < 400:
            if 200 > x:
                prediction[8] = 1
            if 200 < x < 300:
                prediction[9] = 1
            if 300 < x < 400:
                prediction[10] = 1
            if 400 < x < 500:
                prediction[11] = 1
            if 500 < x < 600:
                prediction[12] = 1
            if 600 < x :
                prediction[13] = 1
        if 400 < y < 500:
            if 200 > x:
                prediction[14] = 1
            if 200 < x < 300:
                prediction[15] = 1
            if 300 < x < 400:
                prediction[16] = 1
            if 400 < x < 500:
                prediction[17] = 1
            if 500 < x < 600:
                prediction[18] = 1
            if 600 < x :
                prediction[19] = 1
        if 500 < y < 650:
            if 200 > x:
                prediction[20] = 1
            if 200 < x < 300:
                prediction[21] = 1
            if 300 < x < 500:
                prediction[22] = 1
            if 500 < x < 600:
                prediction[23] = 1
            if 600 < x:
                prediction[24] = 1
        if 650 < y :
            if 200 < x < 300:
                prediction[25] = 1
            if 300 < x < 450:
                prediction[26] = 1
            if 450 < x:
                prediction[27] = 1
#=====================================================================
#================================================알고리즘 출력값 확인==
#sor = np.sort(data)
#print(sor)
print(prediction) # prediction 정상적으로 출력되는지 확인
#================================================알고리즘 출력값 확인==

# 예측 결과 txt로 제출(성능평가에 사용, 이 부분은 수정하지 마시오)
txt_name = str( 'answer(' ) + filename + str( ').txt' )
f = open(txt_name, mode='wt', encoding='utf-8')
for i in range(28):
    f.write('%s\n' %str(prediction[i])) # 예측 결과 
f.close()

# 시각화 결과 표시(예측 결과 확인용, 이 부분은 수정하지 마시오)
cv2.imshow('visualization', img_vis) # 시각화
cv2.waitKey(0) 
cv2.destroyAllWindows() 