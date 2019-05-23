# -*- coding:utf-8 -*-
import cv2 as cv
import plt
import numpy as np

fn = '..\\images\\pixel.png'

'''
注意：以下的A，B将指向同一块内存区域，换言之，修改其中变量中的一个像素，其它变量的“值”也会改变。

'''
A = cv.imread(fn,0)
B = A
ROI = B[:,range(0,3)]

n = A[0,0]
A[0,0] = 255

print(n,A[0,0],ROI[0,0])

plt.myshow(A, B, ROI)