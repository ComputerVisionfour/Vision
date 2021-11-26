import cv2 # opencv import
import numpy as np # numpy import

data=[] # 흰색 픽셀 데이터 리스트 생성

img = cv2.imread("pop_resize_009.jpg", cv2.IMREAD_COLOR)
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

img_stratched = cv2.normalize(img_gray, None, 128, 255, cv2.NORM_MINMAX) # 히스토그램 스트레칭은 NORM_MINMAX

img_stratched2 = cv2.normalize(img_stratched, None, 200, 255, cv2.NORM_MINMAX) # 히스토그램 스트레칭은 NORM_MINMAX

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(9,9))

img2 = clahe.apply(img_stratched2) # 히스토그램 평탄화

clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(9,9))
img3 = clahe.apply(img2) # 히스토그램 평탄화

img4 = cv2.equalizeHist(img3)

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (80,80))
blackhat = cv2.morphologyEx(img3, cv2.MORPH_TOPHAT, kernel1)

cv2.imshow("Result2", img_stratched2)
cv2.imshow("Result1", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()