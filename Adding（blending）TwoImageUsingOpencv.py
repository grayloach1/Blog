# -*- coding:utf-8 -*-
'''
使用OpenCV添加（混合）两个图像
'''

import cv2 as cv
import plt
import numpy as np

fn = '..\\images\\LinuxLogo.jpg'
fn2 = '..\\images\\WindowsLogo.jpg'

src1 = cv.imread(fn)
src2 = cv.imread(fn2)

alpha = 0.7

dst = cv.addWeighted(src1, alpha, src2, 1-alpha, 0.0)

plt.myshow(src1, src2, dst)
