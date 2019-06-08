import plt
import cv2 as cv
import numpy as np

fn  = 'LinuxLogo.jpg'
fn2 = 'WindowsLogo.jpg'
fn2 = 'howdareyouare.jpg'
fn  = 'Lenna.jpg'
#fn2 = 'iris.png'
#fn  = 'horizontal.png'
#fn  = 'Lenna_full.jpg'
#fn  = 'color.png'
#fn2 = 'dog.jpg'
img1 = cv.imread(plt.wd + fn)
img2 = cv.imread(plt.wd + fn2)

class myshow():
    '''用于回调process(*args)，并显示该函数返回的图像'''
    count = 1
    def __init__(self, *args, process=None, **kw):
        self._args = args
        self._kw = kw
        self._proccess = process

        '''窗口名'''
        self._frame_name = str(myshow.count)
        self._windows = cv.namedWindow(self._frame_name)
        myshow.count += 1

        '''如果调用了hook，就生成一个'''
        for tb in [key for key in kw if key.startswith('trackbar')]:
            getattr(self, tb)
            #a, b, c = getattr(self, tb)
            tmp = getattr(self, tb)
            self._tb = tb
            print(tmp)
            #print(self.trackbar_2, kw[tb])
            cv.createTrackbar(tb, self._frame_name, tmp, 100, self.change_trackbar)

        self.show(0)
    def __getattr__(self, item):
        setattr(self, item, self._kw[item][1])

    def show(self, val):
        dst = self._proccess(*self._args)
        cv.imshow(self._frame_name, dst)

    def change_trackbar(self,val):
        self._temp = getattr(self,self._tb)
        self._temp = val

        print(self._temp)
        #self.tb = val
        self.show(self._temp)

    def key_press(self):
        while True:
            k = cv.waitKey()
            if k in (27, 32, 113):
                break

def func(src,trackbar = 0, trackbar_2 = 0):
    #print(f'trackbar:{trackbar},trackbar2:{trackbar_2}')
    size = trackbar * 2 + 1
    dst = cv.blur(src,(size, size))
    #print(trackbar)
    return dst

c1 = myshow(img2,trackbar=[0,10,255],trackbar_2=[0,50,100],process=func)
#c2 = myshow(img2,trackbar=[0,1,100],process=func)
c1.key_press()








