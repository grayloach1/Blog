# -*- coding:utf-8 -*-
'''
绘制基本图形
'''

import cv2 as cv
import plt
import numpy as np

# fn = '..\\images\\fruits.jpg'
# img = cv.imread(fn)

'''生成一个全黑的画布'''
W = 400
size = rows, cols, chs = W, W, 3
img = np.zeros(size, dtype=np.uint8)

cv.ellipse(img,(rows//2, cols//2),(100,50),30,0,360,255,1)
cv.circle(img,(200, 200), 63, (0,255,0), 3)

# pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# pts = pts.reshape((-1,1,2))
#cv.polylines(img, pts ,isClosed=True,color=(0,0,255),thickness=50)

cv.line(img,(0,0),(rows, cols), (0,0,255))
cv.rectangle(img, (100,100), (300,300), (255,255,0))

plt.myshow(img)
