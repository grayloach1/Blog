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
fn2 = 'dog.jpg'

img1 = cv.imread(plt.wd + fn)
img2 = cv.imread(plt.wd + fn2)

def proc(img1,img2):
    dst = cv.addWeighted(img1, proc.val, img2, 1 - proc.val, 0)
    return dst

def cvCanny(src):
    dst = cv.Canny(src, cvCanny.val, 255)
    return dst

def filter(src):
    dst = np.zeros_like(src)
    size = int(filter.val)

    dst = cv.bilateralFilter(src,size, size*2, size/2)
    print(size)
    return dst

def changeShape(src):
    rows,cols=src.shape[:2]

    size = int(changeShape.val)
    print(size)
    kernel = np.ones((size, size), dtype=np.uint8)
    dst0 = cv.erode(src,kernel)
    dst1 = cv.dilate(dst0,kernel)
    return  src - dst1

def miss_or_hint():
    input_image = np.array((
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 255, 255, 255, 0, 0, 0, 255],
        [0, 255, 255, 255, 0, 0, 0, 0],
        [0, 255, 255, 255, 0, 255, 0, 0],
        [0, 0, 255, 0, 0, 0, 0, 0],
        [0, 0, 255, 0, 0, 255, 255, 0],
        [0,255, 0, 255, 0, 0, 255, 0],
        [0, 255, 255, 255, 0, 0, 0, 0]), dtype="uint8")
    kernel = np.array((
            [0, 0, 0],
            [0, -1, 1],
            [0, 1, 0]), dtype="int")

    dst = cv.morphologyEx(input_image,cv.MORPH_HITMISS,kernel)
    dst2 = cv.resize(dst,None,fx = 100, fy = 100, interpolation=cv.INTER_AREA)
    return dst2

def morphological(src):
    dst0 = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    dst1 = cv.bitwise_not(dst0)
    #val = int(morphological.val) * 2 +1

    dst2 = cv.adaptiveThreshold(dst1, 255,
                               cv.ADAPTIVE_THRESH_MEAN_C,
                               cv.THRESH_BINARY,15, - 2)

    dst3 = np.copy(dst2)
    h = np.copy(dst2)

    cols = dst3.shape[1]
    rows = dst3.shape[0]
    horizontal_size = cols // 20
    verticalsize = rows // 20

    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT,(horizontal_size,1))
    verticalStructure =  cv.getStructuringElement(cv.MORPH_RECT,(1,verticalsize))

    dst4 = cv.erode(dst3,verticalStructure)
    dst5 = cv.dilate(dst4,verticalStructure)

    dst6 = cv.erode(h, horizontalStructure)
    dst6 = cv.dilate(dst6, horizontalStructure)
    dst7 = cv.bitwise_not(dst5)

    edges = cv.adaptiveThreshold(dst7, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                 cv.THRESH_BINARY, 3, -2)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernel)

    smooth = np.copy(dst7)
    smooth = cv.blur(smooth, (2, 2))
    (rows, cols) = np.where(edges != 0)
    dst7[rows, cols] = smooth[rows, cols]

    return src,edges,dst7

def pyramids():
    global img1
    rows, cols, _channels = map(int, img1.shape)

    if pyramids.key == ord('i'):
        img1 = cv.pyrUp(img1,dstsize=(2*rows, 2*cols))
        print(pyramids.key)
    if pyramids.key == ord('o'):
        img1 = cv.pyrDown(img1, dstsize=(rows//2, cols//2))
        print(pyramids.key)
    return img1

def hist(src):
    scale = 256
    def make_pts(b_hist):   #转换成适合划线的点集
        temp = np.array([[i, j[0]] for i, j in enumerate(b_hist)])
        max_val = max(temp[:, 1])
        temp[:, 1] = scale - scale * temp[:, 1] / max_val
        # print(len(temp))
        tmp = temp.reshape(-1, 1, 2)
        return np.int32(tmp)

    if len(src.shape) != 2:
        src = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    #b,g,r = cv.split(src)
    histVal = cv.calcHist(src,[0],None,[scale],(0,scale))

    pts = make_pts(histVal)
    #pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    img = np.zeros((scale+20,scale),np.uint8)
    dst = cv.polylines(img,[pts],False,(255,0,0))

    return img

def threshold(src):
    code =  {'THRESH_BINARY': 0,
             'THRESH_BINARY_INV': 1,
             'THRESH_MASK': 7,
             'THRESH_OTSU': 8,
             'THRESH_TOZERO': 3,
             'THRESH_TOZERO_INV': 4,
             'THRESH_TRIANGLE': 16,
             'THRESH_TRUNC': 2}
    gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    _,dst = cv.threshold(gray,threshold.val,256,3)
    return dst

def thresholdRange(src):
    '''
    HSV颜色分量的取值范围
    H: 0 — 180,
    S: 0 — 255,
    V: 0 — 255
    '''
    dst0 = cv.cvtColor(src,cv.COLOR_BGR2HSV)
    val = thresholdRange.val
    #print(dst0)
    dst1 = cv.inRange(dst0,(20, 5,val),(36,255, 255))

    return dst1

def filterKernel(src):
    k3 = np.array([[1, -2, 1],
                   [2, -3, 2],
                   [1, -2, 1]])
    '''构造一个(kernel_size,kernel_size)的核心'''
    kernel_size = (filterKernel.val)*2 +1
    kernel = np.ones((kernel_size,kernel_size),dtype = np.float32)
    kernel /= (kernel_size * kernel_size)

    dst = cv.filter2D(src,-1, kernel)
    return dst

plt.myshow2(img2, val_min = 0 ,val_max= 100,process=filterKernel)
#plt.myshow(filterKernel(img2))













