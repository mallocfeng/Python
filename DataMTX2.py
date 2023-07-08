from pylibdmtx import pylibdmtx
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import messagebox
import sys
import torch
import torchvision
#from sklearn.cluster import DBSCAN
import argparse
import os


rectangles = []
rect = (0, 0, 0, 0)
ix, iy, drawing = 0,0,False

#框选矩形框，获得在矩形框里的圆心，其他不选择
def filter_centers_GUI(ImgPath):
    rectangles.clear
    filter_image = cv.imread(ImgPath, cv.IMREAD_COLOR)
    scale_factor = 0.5
    filter_image = cv.resize(filter_image, (0, 0), fx=scale_factor, fy=scale_factor)
    # 创建窗口shou
    cv.namedWindow('image')
    # 绑定鼠标事件
    param = {'image': filter_image}
    cv.setMouseCallback('image', draw_rect,param)
    # 显示图片
    cv.imshow('image', filter_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 定义鼠标事件回调函数
def draw_rect(event, x, y, flags, param):
    global rect,ix, iy, drawing
    image = param['image']
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            img_copy = image.copy()
            cv.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv.imshow('image', img_copy)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
        width = abs(ix - x)
        height = abs(iy - y)
        x = min(ix, x)
        y = min(iy, y)
        print('矩形坐标：({},{})，矩形长宽：{}x{}'.format(x, y, width, height))
        cv.imshow('image', image)
        # 创建矩形
        rect = (x*2, y*2, width*2, height*2)
        rectangles.append(rect)
        #cv2.destroyAllWindows()
        #imgshowFlag = True


def find_circle(img):
    img = cv.imread(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 4, 100, param1=100, param2=400, minRadius=20, maxRadius=100)
    #circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT_ALT, 2, 100, param1=20, param2=100, minRadius=30, maxRadius=80)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    #把显示的img图片缩小5倍
    img = cv.resize(img, (int(img.shape[1] / 5), int(img.shape[0] / 5)))
    cv.imshow("detected circles", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


'''pop up message box'''
def pop_up_message(msg):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("title", msg)
    root.destroy()
    

def adjust_hue(img, hue_value):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv_img)
    h = np.uint8((h + hue_value) % 180)  # 对H通道进行色相调整
    hsv_img = cv.merge((h, s, v))
    return cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)

def adjust_contrast(img, contrast_value):
    # 对比度调整公式：new_img = alpha * img + beta
    alpha = contrast_value / 127.0
    beta = 127 - alpha * 127
    adjusted_img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted_img




def adjust_levels(img, low_in=0, high_in=255, low_out=0, high_out=255):
    """
    色阶调整函数，将输入图像img的像素值映射到指定的范围内。
    :param img: 输入图像
    :param low_in: 输入图像的最低像素值
    :param high_in: 输入图像的最高像素值
    :param low_out: 输出图像的最低像素值
    :param high_out: 输出图像的最高像素值
    :return: 调整后的图像
    """
    # 构建输入图像的LUT表，将像素值映射到指定范围内
    lut = np.interp(np.arange(0, 256), [low_in, high_in], [low_out, high_out]).astype('uint8')
    # 应用LUT表，对图像进行映射
    adjusted_img = cv.LUT(img, lut)
    return adjusted_img

def filter_close_points(locations, min_distance):
    filtered_locations = []
    for loc in locations:
        if not any(np.linalg.norm(np.array(loc) - np.array(f_loc)) < min_distance for f_loc in filtered_locations):
            filtered_locations.append(loc)
    return filtered_locations

#D:\AI\Report\2023-03\FLKH-4272217-V-QA\2023-03-31 09-55-58_ScanError-2023-03-31-09-56-26\code.bmp
#img = cv.imread(r"E:\Golden\barcode\1\code_E.bmp")
# 创建参数解析器
#parser = argparse.ArgumentParser()

# 添加命令行参数
#parser.add_argument('--arg1', default=r'D:\Image\25mm\Image__2023-06-13__16-23-21.bmp', help=r' ')
#parser.add_argument('--arg2', default=r"1", help=r" ")
# 添加更多参数...

# 解析命令行参数
#args = parser.parse_args()

#args = sys.argv
#img = cv.imread(r"D:\Image\Image__2023-06-13__14-09-51.bmp")

arg1 = sys.argv[1]
arg2 = sys.argv[2]

folder_path, file_name = os.path.split(arg1)
# 构建裁剪后图像的保存路径
output_path = os.path.join(folder_path, "barcode.bmp")

if arg2 == "0":
    filter_centers_GUI(arg1)
    image = cv.imread(arg1)
    # 提取矩形坐标
    #x, y, w, h = rect
    # 裁剪图像
    cropped_image = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    # 获取原始图像的文件夹路径和文件名
    
    # 保存裁剪后的图像
    cv.imwrite(output_path, cropped_image)
    sys.exit(0)

# 读取大图和小图
big_image = cv.imread(arg1)
# 获取图像的宽度和高度
big_height, big_width, big_channels = big_image.shape

#scale_factor = 0.5 
#big_image = cv.resize(big_image, (0, 0), fx=scale_factor, fy=scale_factor)
#big_image = cv.cvtColor(big_image, cv.COLOR_BGR2GRAY)
small_image = cv.imread(output_path)
#small_image = cv.resize(small_image, (0, 0), fx=scale_factor, fy=scale_factor)
#small_image = cv.cvtColor(small_image, cv.COLOR_BGR2GRAY)



# 将小图转换为灰度图像
#small_gray = cv.cvtColor(small_image, cv.COLOR_BGR2GRAY)


# 设置匹配阈值
threshold = 0.7

while True:
    threshold = threshold - 0.05
    # 使用模板匹配方法，在大图中查找小图特征
    result = cv.matchTemplate(big_image, small_image, cv.TM_CCOEFF_NORMED)
    # 获取匹配结果大于阈值的位置信息
    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))
    #########################################
    # 指定最小距离阈值
    min_distance = 30
    # 过滤相距较近的点
    filtered_locations = filter_close_points(locations, min_distance)
    num_locations = len(filtered_locations)
    if num_locations >= 1:
        break
    if threshold <= 0.2:
         sys.exit()
    ##########################################
# 显示标记后的大图
#cv.namedWindow('Matching Result', cv.WINDOW_NORMAL)
#cv.imshow("Matching Result", big_image)
#cv.resizeWindow('Matching Result', int(big_width / 4), int(big_height/4))

cropped_images = []

# 在大图中绘制矩形框标记匹配位置
for loc in filtered_locations:
    top_left = loc
    bottom_right = (top_left[0] + small_image.shape[1], top_left[1] + small_image.shape[0])
    cv.rectangle(big_image, top_left, bottom_right, (0, 255, 0), 2)
    # 裁剪方框区域
    cropped_image = big_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    # 显示裁剪后的图像
    #cv.imshow('Cropped Image', cropped_image)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    # 将裁剪后的图像添加到列表中
    cropped_images.append(cropped_image)

  

#cv.waitKey(0)
#cv.destroyAllWindows()

img = cropped_images[0]
#sys.argv[1] = 
#sys.argv[1]
#E:\Golden\barcode\1\code1.bmp
#img = cv.imread(r"C:\Users\suzmfeng\Desktop\code.bmp")
#find_circle(r"E:\Golden\barcode\1\LeftTop_0_2023-04-12 14_30_03_255.jpg")
#img = cv.fastNlMeansDenoisingColored(img,None,3,6,10,10)
adjusted_img = adjust_levels(img, low_in=50, high_in=180, low_out=0, high_out=255)
#cv.imshow('adjusted image', adjusted_img)
#mg = adjusted_img
#img = cv.fastNlMeansDenoisingColored(img,None,20,20,5,15)
#cv.imshow('Mask', adjusted_img)
all_barcode_info = pylibdmtx.decode(adjusted_img, timeout=500, max_count=1)
barcodeCount = len(all_barcode_info)
if barcodeCount > 0:
   print(all_barcode_info[0].data.decode("utf-8"))
   sys.exit(0)


img = adjust_contrast(img, 150) #default 150
imghue = adjust_hue(img, 130)
#cv.imshow('Mask', imghue)

all_barcode_info = pylibdmtx.decode(imghue, timeout=500, max_count=1)
barcodeCount = len(all_barcode_info)
if barcodeCount > 0:
   print(all_barcode_info[0].data.decode("utf-8"))
   sys.exit(0)


############################

lower_red = np.array([10, 0, 0])
upper_red = np.array([100, 255, 255])  # thers is two ranges of red

# range of red
lower_red2 = np.array([255, 255, 255])
upper_red2 = np.array([255, 255, 255])

hsv = cv.cvtColor(imghue, cv.COLOR_BGR2HSV)
mask_r = cv.inRange(hsv, lower_red, upper_red)
mask_r2 = cv.inRange(hsv, lower_red2, upper_red2)
mask = mask_r + mask_r2
#cv.namedWindow('Mask', 0)
#cv.imshow('Mask', mask)
#############################


cv.imshow('Result', mask)
all_barcode_info = pylibdmtx.decode(mask, timeout=500, max_count=1)
barcodeCount = len(all_barcode_info);
if barcodeCount > 0:
   print(all_barcode_info[0].data.decode("utf-8"))
   sys.exit(0)


for i in range(1, 3):
    lab= cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv.merge((cl,a,b))
    enhanced_img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    result = np.hstack((img, enhanced_img))
    #cv.imshow('Result', result)
    img = enhanced_img


#lab= cv.cvtColor(img, cv.COLOR_BGR2LAB)
#l_channel, a, b = cv.split(lab)
#clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#cl = clahe.apply(l_channel)
#limg = cv.merge((cl,a,b))
#enhanced_img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
#result = np.hstack((img, enhanced_img))
#img = enhanced_img


#cv.imshow('Result', result)


#lab= cv.cvtColor(img, cv.COLOR_BGR2LAB)
#l_channel, a, b = cv.split(lab)
#clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#cl = clahe.apply(l_channel)
#limg = cv.merge((cl,a,b))
#enhanced_img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
#result = np.hstack((img, enhanced_img))
#cv.imshow('Result', result)


#img = cv.fastNlMeansDenoisingColored(img,None,20,20,7,21)
img = cv.fastNlMeansDenoisingColored(img,None,20,20,5,15)
#cv.imshow('Result', img)

all_barcode_info = pylibdmtx.decode(img, timeout=500, max_count=1)
barcodeCount = len(all_barcode_info)
if barcodeCount > 0:
   print(all_barcode_info[0].data.decode("utf-8"))
   sys.exit(0)

# 彩色图像转换为灰度图像（3通道变为1通道）
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#print(gray.shape)
# 最大图像灰度值减去原图像，即可得到反转的图像
img = 255 - gray
#cv.imshow('dst', img)




#img.save(r"E:\Golden\barcode\1\1.bmp")
all_barcode_info = pylibdmtx.decode(img, timeout=500, max_count=1)
barcodeCount = len(all_barcode_info)
if barcodeCount > 0:
   print(all_barcode_info[0].data.decode("utf-8"))
else:
    print("No barcode found")
#try:
#    print(all_barcode_info[0].data.decode("utf-8"))
#except:
#    print("No barcode found")
#print(all_barcode_info[0].data.decode("utf-8"))
#print(sys.argv[1])
#pop_up_message(all_barcode_info[0].data.decode("utf-8"))

