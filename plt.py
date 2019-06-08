from matplotlib.pyplot import *
from time import time
import cv2 as cv

'''设置调用图像的本地默认工作路径'''
wd = 'D:\\Home\\2019-04-26_OpenCV-Python\\images\\'

class WinShow:
    count = 0
    def __init__(self,*args, trackbar1=None, trackbar2 =None,proccess = None):
        self._proccess = proccess
        self._args = args
        #self._kwargs = kwargs
        self._trackbar1 = trackbar1
        self._trackbar2 = trackbar2

        '''窗口名'''
        self._frame_name = str(WinShow.count)
        WinShow.count += 1
        cv.namedWindow(self._frame_name)
        cv.setMouseCallback(self._frame_name, self.on_mouse_callback)

        #trackbars = [i for i in self._kwargs if i.startswith('trackbar')]
        if trackbar1 != None:
            cv.createTrackbar('trackbar1', self._frame_name, trackbar1[1],trackbar1[2], self.on_change1)
        if trackbar2 != None:
            cv.createTrackbar('trackbar2', self._frame_name, trackbar2[1],trackbar2[2], self.on_change2)

        self.show()
        cv.waitKey()
    def on_mouse_callback(self,event,x, y, flag, *arg):
        if flag: print(x, y)
    def on_change1(self, val):
        self._trackbar1[1] = val
        self.show()
    def on_change2(self, val):
        self._trackbar2[1] = val
        self.show()

    def show(self):
        dst = self._proccess(*self._args,self._trackbar1, self._trackbar2)
        cv.imshow(self._frame_name, dst)

class myshow2():
    '''用于回调process(*args)，并显示该函数返回的图像'''
    def __init__(self,*args, hook = 0, trackbar_max=100,process=None):
        cv.namedWindow('image')

        self._proc = process
        self._hook = hook

        cv.createTrackbar('val','image', hook,trackbar_max,self.change)

        while True:
            k = cv.waitKey(400)
            self._proc.key = k                #把键盘值绑定到回调函数
            if k in (32, 27, 113):
                break

            if 'hook'in self._proc.__code__.co_varnames:
                dst = process(*args,hook = self._hook)
            else:
                dst = process(*args)

            cv.imshow('image', dst)
        cv.destroyAllWindows()
    def change(self,val):
        self._hook = val

def myshow(*imgs):
    '''显示多个图像源（<4），不再输入其它参数'''

    src = []
    for i in imgs:
        if i is None:
            print('WARNING: 传入的图像为空')
            return 0
        if len(i.shape) == 2:
            #i = np.uint8(i)
            temp = cv.cvtColor(np.uint8(i), cv.COLOR_GRAY2BGR)
            src.append(temp)
        else:
            src.append(i)
    images_number = len(src)

    if images_number > 4:
        raise TypeError(f'myshow() takes at less than 4 arguments ({images_number} given)')
    if images_number == 1:
        img0 = src[0]
        imshow(img0[:,:,::-1])
    if images_number == 2:
        img0,img1 = src
        subplot(121)
        imshow(img0[:,:,::-1])

        subplot(122)
        imshow(img1[:,:,::-1])
    if images_number == 3:
        img0,img1, img2 = src
        subplot(221)
        imshow(img0[:,:,::-1])

        subplot(222)
        imshow(img1[:,:,::-1])

        subplot(223)
        imshow(img2[:, :, ::-1])
    if images_number == 4:
        img0,img1, img2, img3 = src
        subplot(221)
        imshow(img0[:,:,::-1])

        subplot(222)
        imshow(img1[:,:,::-1])

        subplot(223)
        imshow(img2[:, :, ::-1])

        subplot(224)
        imshow(img3[:, :, ::-1])
    show()
