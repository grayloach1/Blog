"""
一个用于显示函数返回的图像的装饰器类

在opencv的调试和学习过程中，往往需要显示图像，直观的观察函数输出的图像效果。
而opencv的函数，根据不同的输入图像，往往需要不断的调整函数参数值。这给调试
过程带来很大不便，本模块可以用解决这个问题。

使用装饰器的方式调用该类，把 @show_trackbar 置于函数前面一行即可实现:
      1、显示函数返回的图像
      2、增加滚动条，函数可以调用滚动条的值。

详细使用方法，可以参见if __name__ == '__main__': 后面的使用示例。

v1.0 2019-06-16by老余
"""
import cv2 as cv
from inspect import signature


class show_trackbar:
    """
    使用装饰器的方式调用该类。把 @decorator_trackbar 置于函数前面一行即可。
    可以实现:
      1、显示函数返回的图像
      2、增加滚动条，函数可以调用滚动条的值。
    """
    def __init__(self, process):
        self._process = process
        self.__signature__ = signature(process)   # 保存函数定义时的函数签名
        cv.namedWindow('image')                   # 默认窗口名'image'，方便调用

    def __call__(self, *args, **kwargs):
        """
        把调用时传入的参数，绑定到函数签名上；
        创建滑动条，使它可以在显示中修改函数的参数值
        """
        sig_bind = self.__signature__.bind(*args, **kwargs)
        sig_bind.apply_defaults()                 # 绑定调用时传入的参数

        '''把参数存入self.tbs，用于和滑动条交换数据'''
        self.tbs = {k: v for k, v in sig_bind.arguments.items()}
        for k, v in self.tbs.items():
            if k.startswith('trackbar'):
                cv.createTrackbar(k, 'image', v[1], v[2], self.on_change(k))

        self.show(**sig_bind.arguments)
        cv.waitKey()

    def show(self, **kwargs):
        """根据传入的参数调用被装饰的函数并显示其返回的图像"""
        dst = self._process(**kwargs)
        cv.imshow('image', dst)

    def on_change(self, k):
        """
        该闭包函数，用于响应滑动条的动作，
        以滑动条的当前值作为参数，调用重新调用show（）"""
        def change(*args):
            tmp = self.tbs[k]
            tmp[1] = args[0]
            self.tbs[k] = tmp
            self.show(**self.tbs)
        return change


if __name__ == '__main__':
    '''
    !!!请把文件fn的值改为本地的图像文件
    
    示例图可以从这里下载：
    https://upload.wikimedia.org/wikipedia/zh/3/34/Lenna.jpg
    '''
    fn = '.\\images\\Lenna.jpg'
    img = cv.imread(fn)

    @show_trackbar
    def cvt_gray(src, trackbar_var=[0, 127, 256], trackbar_size=[0, 127, 255]):
        """
        这里是用户定义的用来处理图像的函数，
        当含有类似'trackbar???=[0,a,b]'形式的参数时，
        会创建一个以a为默认值，值域为[0, b]的滚动条，
        拖动滚动条，会改变默认值a，
        在程序中调用trackbar???[1]的值，
        即可实现改变参数值，并实时查看参数改变之后图像的处理效果。

        :return 必须返回一个图像，用于窗口显示。
        """
        size = trackbar_size[1] * 2 + 1
        dst = cv.blur(src, (size, size))
        return dst

    '''调用刚才自定义的函数'''
    cvt_gray(img, trackbar_size=[0, 3, 100])


