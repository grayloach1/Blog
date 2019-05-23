# -*- coding:utf-8 -*-
'''
图像的基本操作
'''

import cv2 as cv
import plt
import numpy as np

fn = '..\\images\\dog.jpg'


img = cv.imread(fn)
gray = cv.imread(fn,cv.COLOR_BGR2GRAY)

img1 = np.copy(img)
img[:] = 0

#cv.imwrite('dog2.jpg', gray)

cv.namedWindow('image')
cv.imshow('image', img1)
cv.waitKey()