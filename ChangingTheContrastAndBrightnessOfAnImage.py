# -*- coding:utf-8 -*-
'''
使用OpenCV添加（混合）两个图像
'''

import cv2 as cv
import plt
import numpy as np

fn = '..\\images\\fruits.jpg'
img = cv.imread(fn)
dst = img.copy()

'''构造一个表，以原来的值为x，新的值为lookUpTable[x]'''
gamma = 0.4
lookUpTable = np.empty(256, np.uint8)
for i in range(256):
    lookUpTable[i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

'''逐像素的查找值，改变为查表得到的新值'''
rows,cols,chs = dst.shape
for row in range(rows):
    for col in range(cols):
        for ch in range(chs):
            x = dst[row, col, ch]
            dst[row, col, ch] = lookUpTable[x]


plt.myshow(img, dst)
