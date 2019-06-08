import plt
import cv2 as cv
import numpy as np

fn  = 'LinuxLogo.jpg'
fn2 = 'WindowsLogo.jpg'
fn  = 'horizontal.png'
fn  = 'Lenna_full.jpg'
fn  = 'color.png'
fn  = 'Lenna.jpg'
fn  = 'sudoku.png'
fn2 = 'smarties.png'
fn2 = 'dog.jpg'
fn2 = 'howdareyouare.jpg'
fn2 = 'iris.png'


img1 = cv.imread(plt.wd + fn)
img2 = cv.imread(plt.wd + fn2)

def add_weighted(img1,img2,hook):
    hook = hook /100
    dst = cv.addWeighted(img1, hook, img2, 1 - hook, 0)
    return dst

def cvCanny(src,hook):
    dst = cv.Canny(src, hook, 255)
    return dst

def filter(src,hook):
    dst = np.zeros_like(src)
    size = int(hook)

    dst = cv.bilateralFilter(src,size, size*2, size/2)
    print(size)
    return dst

def changeShape(src,hook):
    rows,cols=src.shape[:2]

    size = int(hook)
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

def threshold(src,hook):
    code =  {'THRESH_BINARY': 0,
             'THRESH_BINARY_INV': 1,
             'THRESH_MASK': 7,
             'THRESH_OTSU': 8,
             'THRESH_TOZERO': 3,
             'THRESH_TOZERO_INV': 4,
             'THRESH_TRIANGLE': 16,
             'THRESH_TRUNC': 2}
    gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    _,dst = cv.threshold(gray,hook,256,3)
    return dst

def thresholdRange(src,hook):
    '''
    HSV颜色分量的取值范围
    H: 0 — 180,
    S: 0 — 255,
    V: 0 — 255
    '''
    dst0 = cv.cvtColor(src,cv.COLOR_BGR2HSV)
    #val = thresholdRange.val
    #print(dst0)
    dst1 = cv.inRange(dst0,(20, 5,hook),(36,255, 255))

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

def add_border(src):
    from random import randint
    code = { 'BORDER_CONSTANT': 0,
             'BORDER_DEFAULT': 4,
             'BORDER_ISOLATED': 16,
             'BORDER_REFLECT': 2,
             'BORDER_REFLECT101': 4,
             'BORDER_REFLECT_101': 4,
             'BORDER_REPLICATE': 1,
             'BORDER_TRANSPARENT': 5,
             'BORDER_WRAP': 3}
    value = [randint(0, 255), randint(0, 255), randint(0, 255)]
    size = add_border.val
    dst = cv.copyMakeBorder(src, size,size,size,size,1,value=value)
    return dst

def sobel_operator(src,hook):
    size = hook * 2 + 1
    gaussian = cv.GaussianBlur(src, (size, size),0)
    gray = cv.cvtColor(gaussian,cv.COLOR_BGR2GRAY)
    grad_x = cv.Sobel(gray, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    dst = grad
    return dst

def laplacian_operator(src, hook):
    size = hook * 2 + 1
    gaussian = cv.GaussianBlur(src, (size, size),0)
    gray = cv.cvtColor(gaussian,cv.COLOR_BGR2GRAY)
    lapla = cv.Laplacian(gray, cv.CV_16S,ksize=3)
    dst = cv.convertScaleAbs(lapla)

    return dst

def canny_edge_detector(src, hook):
    gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    img_blur = cv.blur(gray,(3,3))
    edges = cv.Canny(img_blur, hook, hook * 3, 3)

    mask = edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))

    return edges

def hough_line(src, hook):
    dst = src.copy()
    edges = cv.Canny(src, 50, 200, None, 3)

    min_line_length = hook
    max_line_gap = 10

    #linesP = cv.HoughLinesP(edges, 1, np.pi/180, 150, None,min_line_length, max_line_gap)

    # for pt in linesP:
    #     x1, y1, x2, y2 = pt[0]
    #     cv.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # return dst

    lines = cv.HoughLines(edges, 2, np.pi / 180, hook)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(dst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    return dst

def hough_circle(src, hook):

    hook = 17  #示例iris所用的专用经验值。
    ksize = hook * 2 + 1

    dst = src.copy()
    gray0 = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    kernel = np.ones((ksize, ksize), np.uint8)
    gray1 = cv.dilate(gray0, kernel)
    gray  = cv.erode(gray1, kernel)

    blur = cv.medianBlur(gray, 3)

    rows = gray.shape[0]
    circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1, rows/8,
                              param1 = 100, param2 = 30,
                              minRadius = rows//10, maxRadius= rows)


    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv.circle(dst, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv.circle(dst, (i[0], i[1]), 2, (0, 0, 255), 3)

    return dst

def mapping(src,hook):
    '''非官方版本，而是直接通过下标访问'''
    i = hook % 4
    maps = [src[::-1,:,:], src[:,::-1,:], src[::-1,::-1,:], src[:,:,:]]
    return maps[i]

def affine_trans(src, hook):
    srcTri = np.array([[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]]).astype(np.float32)
    dstTri = np.array([[0, src.shape[1] * 0.33], [src.shape[1] * 0.85, src.shape[0] * 0.25],
                       [src.shape[1] * 0.15, src.shape[0] * 0.7]]).astype(np.float32)
    warp_mat = cv.getAffineTransform(srcTri, dstTri)
    warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))

    center = (warp_dst.shape[1] // 2, warp_dst.shape[0] // 2)
    angle = -hook
    scale = 1

    rot_mat = cv.getRotationMatrix2D(center, angle, scale)
    warp_rotate_dst = cv.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))

    dst = warp_rotate_dst
    return dst

def fit_ellipse(src, hook):
    '''
    函数尚未完成...

    ellipse = cv2.fitEllipse(cnt)
    im = cv2.ellipse(im,ellipse,(0,255,0),2)

    Fitting an Ellipse
    '''
    hook = 17  # 示例iris所用的专用经验值。
    ksize = hook * 2 + 1

    dst = src.copy()
    gray0 = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    kernel = np.ones((ksize, ksize), np.uint8)
    gray1 = cv.dilate(gray0, kernel)
    gray = cv.erode(gray1, kernel)

    ret, thresh = cv.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, 1, 2)
    cnt = contours[0]
    M = cv.moments(cnt)

    ellipse = cv.fitEllipse(cnt)
    dst = cv.ellipse(dst, ellipse, (0, 255, 0), 2)

    #dst = gray
    return gray


plt.myshow2(img2, hook = 30,trackbar_max= 255,process=fit_ellipse)
#plt.myshow(add_border(img2))


