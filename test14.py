# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 01:08:19 2021

@author: Coding
"""
import cv2 # opencv import
import numpy as np # numpy import
import os


#===== 이미지 열기 ========================================
img = cv2.imread("pop_resize_009.jpg", cv2.IMREAD_COLOR)
img_first = cv2.resize(img, dsize=(800, 800), interpolation=cv2.INTER_AREA)
img_c = img_first.copy()

#================== 이미지 선명도 증가 함수===========
def img_Contrast(img_c):
    lab = cv2.cvtColor(img_c, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(9, 9))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final
img_c =  img_Contrast(img_c)
#=====================================================

img_gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY) # 이미지 그레이스케일

img_stratched = cv2.normalize(img_gray, None, 128, 255, cv2.NORM_MINMAX) # 히스토그램 스트레칭은 NORM_MINMAX

img_stratched2 = cv2.normalize(img_stratched, None, 128, 255, cv2.NORM_MINMAX) # 히스토그램 스트레칭은 NORM_MINMAX

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img2 = clahe.apply(img_stratched)


op_idx = {
    'gradient':cv2.MORPH_GRADIENT,
    'tophat':cv2.MORPH_TOPHAT,
    'blackhat':cv2.MORPH_BLACKHAT,    
}

def onChange(k, op_name):
    if k == 0:
        cv2.imshow(op_name, img_stratched2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k,k))
    dst = cv2.morphologyEx(img_stratched2, op_idx[op_name], kernel)
    cv2.imshow(op_name, dst)
    

cv2.imshow('src', img_stratched2)
cv2.imshow('blackhat', img_stratched2)
cv2.imshow('tophat', img_stratched2)
cv2.imshow('gradient', img_stratched2)

cv2.createTrackbar('k', 'blackhat', 0, 300, lambda x: onChange(k=x, op_name='blackhat'))
cv2.createTrackbar('k', 'tophat', 0, 300, lambda x: onChange(k=x, op_name='tophat')) 
cv2.createTrackbar('k', 'gradient', 0, 300, lambda x: onChange(k=x, op_name='gradient'))  
                   
cv2.waitKey()
cv2.destroyAllWindows()