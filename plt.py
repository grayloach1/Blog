from matplotlib.pyplot import *
import cv2 as cv

def myshow(*imgs):
    '''显示多个图像源（<4），不在输入其它参数'''
    src = []
    for i in imgs:
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
