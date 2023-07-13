import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.optimize import minimize
import sys
import os
import ast
from CircleDetect import find_circles

ix, iy, drawing = 0,0,False
rectangles = []

# 定义鼠标事件回调函数
def draw_rect(event, x, y, flags, param):
    global rect,ix, iy, drawing
    image = param['image']
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img_copy = image.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
        width = abs(ix - x)
        height = abs(iy - y)
        x = min(ix, x)
        y = min(iy, y)
        print('矩形坐标：({},{})，矩形长宽：{}x{}'.format(x, y, width, height))
        cv2.imshow('image', image)
        # 创建矩形
        rect = (x, y, width, height)
        rectangles.append(rect)
        #cv2.destroyAllWindows()
        #imgshowFlag = True

def filter_centers(center_list, rectangles_List):
    filtered_centers = []
    if len(rectangles_List) == 0:
        return center_list
    for center in center_list:
        x, y = center
        
        # 检查每个圆心坐标是否在任何矩形区域内
        inside_rectangle = False
        for rect in rectangles_List:
            rect_x, rect_y, rect_width, rect_height = rect
            if rect_x <= x <= rect_x + rect_width and rect_y <= y <= rect_y + rect_height:
                inside_rectangle = True
                break
        
        # 如果圆心在矩形区域内，则将其添加到过滤后的列表中
        if inside_rectangle:
            filtered_centers.append(center)
    
    return filtered_centers


def detect_circles(image,Radius = 20):
    circles, result_img = find_circles(image)
    center_list = []  # 存储圆心坐标的列表
    for circle in circles:
        if circle[2] > Radius:
            center_list.append((circle[0],circle[1]))
    center_list = filter_centers(center_list,rectangles)
    return center_list,result_img  # 返回圆心坐标列表和图片

def detect_circles1(image,Radius = 20):
    # 读取图像
    #image = cv2.imread(image_path)
    #scale_factor = 0.5
    #image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    output = image.copy()

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用高斯模糊降噪out
    blurred = cv2.GaussianBlur(gray, (17, 17), 0)
    #blurred = cv2.bilateralFilter(image, 5, 75, 5)

     
    #blurred = gray
    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    #cv2.imshow("edges Circles", edges)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_list = []  # 存储圆心坐标的列表
    count = 0
    for contour in contours:
        # 计算轮廓的面积和周长
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # 使用Hu矩来判断形状
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            #if 0.85 < circularity < 1.5:  # 这个范围可以根据实际情况调整
            if 0.5 < circularity < 1.7:
            #if 0.7 < circularity < 1:
                # 计算最小外接圆
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                
                
                
                # 在图像上绘制圆形和中心点
                if radius > Radius:
                    #if(count == 0):
                    #    cv2.circle(output, center, int(radius), (0, 0, 255), 2)
                    #if(count == 1):
                    #    cv2.circle(output, center, int(radius), (0, 255, 0), 2)
                    #if(count == 2):
                    #    cv2.circle(output, center, int(radius), (255, 0, 0), 2)
                    #if(count > 2):
                    #    cv2.circle(output, center, int(radius), (255, 255, 0), 2)
                    #cv2.circle(output, center, 1, (0, 0, 255), 5)
                    center_list.append(center)  # 将圆心坐标添加到列表中
                    #print(f"Circle center: {center}, radius: {radius}")
                    count = count + 1
                #print("############")

    # 显示结果
    #cv2.imshow("Detected Circles", output)
    #big_height, big_width, big_channels = output.shape
    #cv2.resizeWindow('Detected Circles', int(big_width / 4), int(big_height/4))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    center_list = filter_centers(center_list,rectangles)
    return center_list,output  # 返回圆心坐标列表和图片

def filter_centers_GUI(ImgPath):
    #rectangles.clear
    image = cv2.imread(ImgPath, cv2.IMREAD_COLOR)
    scale_factor = 1
    image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    # 创建窗口
    cv2.namedWindow('image')
    # 绑定鼠标事件
    param = {'image': image}
    cv2.setMouseCallback('image', draw_rect,param)

    # 显示图片
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def transformCoordinate(srcPoints, dstPoints, points):
    # 计算矩阵变换系数
    x = 0
    y = 0
    for i in range(8):
        a = (dstPoints[i+1][1] - dstPoints[i][1]) / (srcPoints[i+1][1] - srcPoints[i][1])
        b = dstPoints[i][1] - a * srcPoints[i][1]
        c = (dstPoints[i+1][0] - dstPoints[i][0]) / (srcPoints[i+1][0] - srcPoints[i][0])
        d = dstPoints[i][0] - c * srcPoints[i][0]

        # 对每个点进行坐标系转换
        x = points[0] * c + d + x
        y = points[1] * a + b + y
        #x = round(x)
        #y = round(y)
    return x/8, y/8
# 文件夹路径
folder_path = 'D:\\Image\\25mm'
output_file_path = 'D:\\Image\\25mm\\file.txt'

# 遍历文件夹
with open(output_file_path, 'w') as f:
    for i in range(1, 10):
        # 构造文件名
        #filename = f'Pos{i}.bmp'
        filename = r'D:\Image\25mm\Actual_Fixture.bmp'
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, filename)
        
        # 检查文件是否存在
        if os.path.exists(file_path):
            # 读取图片
            img = cv2.imread(file_path)
            #alpha = 3  # 对比度增益因子，大于1增加对比度，小于1减小对比度
            #beta = -200   # 亮度调整参数，可以为0
            #img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            rectangles = []
            filter_centers_GUI(file_path)
            centers_Pos1,Pos1_img = detect_circles(img)
            for center_original in centers_Pos1:
                x, y = center_original
                print(f"CenterPoint{i}: {x},{y}\n")
                f.write(f"CenterPoint{i}: {x},{y}\n")
                break
        else:
            print(f'File {file_path} does not exist.')


vision_coords = []
#output_file_path = '/path/to/your/output/file.txt'  # 你的文件路径

with open(output_file_path, 'r') as f:
    for line in f:
        # 去除行尾的换行符
        line = line.strip()
        # 分割字符串
        parts = line.split(': ')
        if len(parts) == 2:
            # 获取坐标部分
            coords_str = parts[1]
            # 分割坐标
            coords_parts = coords_str.split(',')
            if len(coords_parts) == 2:
                # 转换坐标为浮点数并添加到列表
                x = float(coords_parts[0])
                y = float(coords_parts[1])
                vision_coords.append((x, y))

# 将列表转换为NumPy数组
vision_coords = np.array(vision_coords)

arm_coords = []
file_path = 'D:\\Image\\25mm\\file2.txt'  # 你的文件路径

with open(file_path, 'r') as f:
    for line in f:
        # 去除行尾的换行符
        line = line.strip()
        # 将字符串转换为列表
        coords = ast.literal_eval(line)
        # 取前三个数字并添加到列表
        arm_coords.append(coords[:3])

# 将列表转换为NumPy数组
arm_coords = np.array(arm_coords)

# 假设你已经获取了机械臂坐标和视觉坐标
# 这里我们只是创建一些随机数据作为示例
#arm_coords = np.random.rand(9, 3)  # 机械臂坐标
#vision_coords = np.random.rand(9, 2)  # 视觉坐标

# 使用OpenCV的findHomography函数计算单应性矩阵
#h, status = cv2.findHomography(vision_coords, arm_coords)

# 现在你可以使用A来将视觉坐标转换为机械臂坐标
filename = f'Pos10.bmp'
# 构造完整的文件路径
file_path = os.path.join(folder_path, filename)
img = cv2.imread(file_path)
alpha = 3  # 对比度增益因子，大于1增加对比度，小于1减小对比度
beta = -300   # 亮度调整参数，可以为0
img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
rectangles = []
filter_centers_GUI(file_path)
centers_Pos1,Pos1_img = detect_circles(img)
for center_original in centers_Pos1:
    x, y = center_original
    print(f"CenterPointfinal: {x},{y}\n")
    break

#[-557.40555, 96.560426, 345.884097, -3.141593, -0.0, -0.409395]

# 使用OpenCV的getAffineTransform函数计算仿射变换矩阵
#M = cv2.getAffineTransform(vision_coords, arm_coords)

x1,y1 = transformCoordinate(vision_coords,arm_coords,[x,y])
print(f"{x1},{y1}")
sys.exit(0)


StartImgPath = 'D:\\Image\\25mm\\Pos1.bmp'
EndImgPath = 'D:\\Image\\25mm\\Pos2.bmp'
EndImgPath2 = 'D:\\Image\\25mm\\Pos3.bmp'

StartImgPath = sys.argv[1]
EndImgPath = sys.argv[2]
EndImgPath2 = sys.argv[3]

#机械臂x轴移动距离
ArmXaxis = 1000
ArmYaxis = 100

ArmXaxis = float(sys.argv[4])
ArmYaxis = float(sys.argv[5])

RootPath = os.path.dirname(StartImgPath)


img = cv2.imread(StartImgPath)
alpha = 2  # 对比度增益因子，大于1增加对比度，小于1减小对比度
beta = -200   # 亮度调整参数，可以为0
img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
filter_centers_GUI(StartImgPath)
centers_Pos1,Pos1_img = detect_circles(img)
for center_original in centers_Pos1:
    x, y = center_original
    print(f"CenterPoint1: {x},{y}\n")

img = cv2.imread(EndImgPath)
img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
filter_centers_GUI(EndImgPath)
centers_Pos2,Pos2_img = detect_circles(img)
for center_original in centers_Pos2:
    x, y = center_original
    print(f"CenterPoint2: {x},{y}\n")

img = cv2.imread(EndImgPath2)
img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
filter_centers_GUI(EndImgPath2)
centers_Pos3,Pos3_img = detect_circles(img)
for center_original in centers_Pos3:
    x, y = center_original
    print(f"CenterPoint3: {x},{y}\n")

center_Pos1 = centers_Pos1[0]
center_Pos2 = centers_Pos2[0]
center_Pos3 = centers_Pos3[0]
# 计算x轴上的距离（绝对值）
distance_x = abs(center_Pos2[0] - center_Pos1[0])
# 计算y轴上的距离（绝对值）
distance_y = abs(center_Pos3[1] - center_Pos1[1])

if ArmXaxis == 0:
    print("x-axis ratio:", 1)
else:
    print("x-axis ratio:", ArmXaxis/distance_x)
if ArmYaxis == 0:
    print("y-axis ratio:", 1)
else:
    print("y-axis ratio:", ArmYaxis/distance_y)

