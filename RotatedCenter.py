import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.optimize import minimize
import sys
from CircleDetect import find_circles
import configparser
import os
#r'D:\Project3\ICT Automation\picture\6\initial\1.png'
# 读取图片并进行预处理

rect = (0, 0, 0, 0)
ix, iy, drawing = 0,0,False
rectangles = []

def filter_centers(center_list, rectangles_List):
    filtered_centers = []
    if len(rectangles_List) == 0:
        return center_list
    
    for rect in rectangles_List:
        rect_x, rect_y, rect_width, rect_height = rect    
        # 检查每个圆心坐标是否在任何矩形区域内
        inside_rectangle = False
        for center in center_list:
            x, y = center
            if rect_x <= x <= rect_x + rect_width and rect_y <= y <= rect_y + rect_height:
                inside_rectangle = True
                print(center)
                break
        
        # 如果圆心在矩形区域内，则将其添加到过滤后的列表中
        if inside_rectangle:
            filtered_centers.append(center)
    
    return filtered_centers

def calculate_angle_horizontal(center1, center2):
    x1, y1 = center1
    x2, y2 = center2

    # 计算直线斜率
    slope = (y2 - y1) / (x2 - x1)

    # 计算直线与水平线之间的夹角（以度数为单位）
    angle = math.degrees(math.atan(slope))

    return angle

def line_intersects_rectangle(pt1, pt2, x, y, w, h):
    rect_pt1 = (x, y)
    rect_pt2 = (x + w, y)
    rect_pt3 = (x + w, y + h)
    rect_pt4 = (x, y + h)

    # 直线与矩形的四条边进行相交性检测
    intersects = (
        line_intersects_segment(pt1, pt2, rect_pt1, rect_pt2)
        or line_intersects_segment(pt1, pt2, rect_pt2, rect_pt3)
        or line_intersects_segment(pt1, pt2, rect_pt3, rect_pt4)
        or line_intersects_segment(pt1, pt2, rect_pt4, rect_pt1)
    )

    return intersects
    

def line_intersects_segment(pt1, pt2, seg_pt1, seg_pt2):
    # 判断两条线段是否相交
    d1 = direction(seg_pt1, seg_pt2, pt1)
    d2 = direction(seg_pt1, seg_pt2, pt2)
    d3 = direction(pt1, pt2, seg_pt1)
    d4 = direction(pt1, pt2, seg_pt2)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    elif d1 == 0 and on_segment(seg_pt1, seg_pt2, pt1):
        return True
    elif d2 == 0 and on_segment(seg_pt1, seg_pt2, pt2):
        return True
    elif d3 == 0 and on_segment(pt1, pt2, seg_pt1):
        return True
    elif d4 == 0 and on_segment(pt1, pt2, seg_pt2):
        return True

    return False

def direction(pt1, pt2, pt3):
    return (pt3[0] - pt1[0]) * (pt2[1] - pt1[1]) - (pt2[0] - pt1[0]) * (pt3[1] - pt1[1])

def on_segment(pt1, pt2, pt3):
    return min(pt1[0], pt2[0]) <= pt3[0] <= max(pt1[0], pt2[0]) and min(pt1[1], pt2[1]) <= pt3[1] <= max(pt1[1], pt2[1])


def calculate_angle_line(line1, line2):
    # 提取 line1 和 line2 的起点和终点坐标
    pt1_1, pt2_1 = line1
    pt1_2, pt2_2 = line2
    # 计算 line1 的向量
    vec1 = np.array([pt2_1[0] - pt1_1[0], pt2_1[1] - pt1_1[1]])
    # 计算 line2 的向量
    vec2 = np.array([pt2_2[0] - pt1_2[0], pt2_2[1] - pt1_2[1]])
    # 计算向量的内积
    dot_product = np.dot(vec1, vec2)
    # 计算向量的模
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    # 计算夹角的余弦值
    cosine = dot_product / (norm1 * norm2)
    # 计算夹角的弧度值
    angle_rad = np.arccos(cosine)
    # 将弧度转换为角度
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def calculate_angle(line):
    # 提取直线的起点和终点坐标
    pt1, pt2 = line

    # 计算直线的角度
    angle_rad = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def find_lines_with_angle(lines, threshold):
    # 用于存储符合条件的直线
    filtered_lines = []

    for line in lines:
        angle = calculate_angle(line)

        # 检查直线与水平线之间的夹角是否小于阈值
        if abs(angle) < threshold:
            filtered_lines.append(line)

    return filtered_lines

def rotate_image_to_horizontal(image, center, angle):
    # 获取图像的尺寸
    height, width = image.shape[:2]

    # 创建旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # 对图像进行旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image

def detect_circles(image,Radius = 20):
    circles, result_img = find_circles(image,191)
    center_list = []  # 存储圆心坐标的列表
    for circle in circles:
        if circle[2] > Radius:
            center_list.append((circle[0],circle[1]))
    center_list = [(x*2, y*2) for x, y in center_list]
    center_list = filter_centers(center_list,rectangles)
    
    return center_list,result_img  # 返回圆心坐标列表和图片


def detect_circles1(image):
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
            #if 0.8 < circularity < 1.2:  # 这个范围可以根据实际情况调整
            if 0.85 < circularity < 1.5:
                # 计算最小外接圆
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))

                # 在图像上绘制圆形和中心点
                if radius > 20:
                    if(count == 0):
                        cv2.circle(output, center, int(radius), (0, 0, 255), 2)
                    if(count == 1):
                        cv2.circle(output, center, int(radius), (0, 255, 0), 2)
                    if(count == 2):
                        cv2.circle(output, center, int(radius), (255, 0, 0), 2)
                    if(count > 2):
                        cv2.circle(output, center, int(radius), (255, 255, 0), 2)
                    cv2.circle(output, center, 1, (0, 0, 255), 3)
                    #print(f"Circle center: {center}, radius: {radius}")
                    center_list.append(center)  # 将圆心坐标添加到列表中
                    count = count + 1
                

    # 显示结果
    #cv2.imshow("Detected Circles", output)
    #big_height, big_width, big_channels = output.shape
    #cv2.resizeWindow('Detected Circles', int(big_width / 4), int(big_height/4))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    center_list = [(x*2, y*2) for x, y in center_list]
    center_list = filter_centers(center_list,rectangles)
    
    return center_list,output  # 返回圆心坐标列表和图片


def find_right_angles(lines, threshold=10):
    right_angles = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            angle = angle_between_lines(line1, line2)
            if 90 - threshold <= angle <= 90 + threshold:
                right_angles.append((line1, line2))
    return right_angles


def angle_between_lines(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    angle = abs(np.rad2deg(theta1 - theta2))
    if angle > 90:
        angle = 180 - angle
    return angle

def merge_lines(lines, angle_threshold=20, min_angle=85, max_angle=95):
    merged_lines = []
    for line in lines:
        if not merged_lines:
            merged_lines.append(line)
            continue

        for merged_line in merged_lines:
            angle = angle_between_lines(line, merged_line)
            if angle > min_angle and angle < max_angle:
                if abs(line[0][0] - merged_line[0][0]) < angle_threshold:
                    break
        else:
            merged_lines.append(line)

    return merged_lines


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
        rect = (x*2, y*2, width*2, height*2)
        rectangles.append(rect)
        #cv2.destroyAllWindows()
        #imgshowFlag = True

#框选矩形框，获得在矩形框里的圆心，其他不选择
def filter_centers_GUI(ImgPath):
    #rectangles.clear
    filter_image = cv2.imread(ImgPath, cv2.IMREAD_COLOR)
    scale_factor = 0.5
    filter_image = cv2.resize(filter_image, (0, 0), fx=scale_factor, fy=scale_factor)
    # 创建窗口shou
    cv2.namedWindow('image')
    # 绑定鼠标事件
    param = {'image': filter_image}
    cv2.setMouseCallback('image', draw_rect,param)

    # 显示图片
    cv2.imshow('image', filter_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rotate_image(image, center_x, center_y, angle):
    # 获取图像宽度和高度
    height, width = image.shape[:2]
    
    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    
    # 执行图像旋转
    rotated_image = cv2.warpAffine(image, M, (width, height))
    
    return rotated_image

def rotate_point(x, y, theta, x0, y0):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    return (
        np.cos(theta) * (x - x0) - np.sin(theta) * (y - y0) + x0,
        np.sin(theta) * (x - x0) + np.cos(theta) * (y - y0) + y0
    )

def objective_func(x0, points, rotated_points,RotationAangle):
    """
    Objective function to be minimized to find the center of rotation.
    """
    x0, y0 = x0
    error = 0
    for (x, y), (xp, yp) in zip(points, rotated_points):
        xr, yr = rotate_point(x, y, np.radians(float(RotationAangle)), x0, y0)  #如果是90度，这里改成 np.pi/2
        error += (xr - xp)**2 + (yr - yp)**2
    return error

# 替换为你的图像路径
#image_path = r'D:\Project3\ICT Automation\picture\6\initial\4.png'
#image_path = r'D:\Image\Image__2023-06-14__15-44-55.bmp'
#D:\Image\Image__2023-06-15__10-03-37.bmp

RootPath = "D:\\Image\\25mm\\"
image_path = RootPath + '1.bmp'
RootPath + '2.bmp'

arg1 = sys.argv[1]
arg2 = sys.argv[2]
arg3 = sys.argv[3]
arg4 = sys.argv[4]

#D:\\Image\\25mm\\Camera2StdRotationCenter.bmp D:\\Image\\25mm\\Camera2StdRotationCenter15.bmp D:\\Image\\25mm\\final.jpg 15
# arg1 = "D:\\Image\\25mm\\Camera1StdRotationCenter.bmp"
# arg2 = "D:\\Image\\25mm\\Camera1StdRotationCenter90.bmp"
# arg3 = "D:\\Image\\25mm\\final.jpg"
# arg4 = -90

RootPath = os.path.dirname(arg1) + "\\"


image_path = arg1
image_Rotate_path = arg2
image_output_path = arg3
RotationAangle = arg4
CameraID = "Camera1"
if(int(RotationAangle) == 90):
    CameraID = "Camera1"
if(int(RotationAangle) == 10):
    CameraID = "Camera2"

image = cv2.imread(image_path)

# 增加对比度
#alpha = 2  # 对比度增益因子，大于1增加对比度，小于1减小对比度
#beta = -200   # 亮度调整参数，可以为0
#image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
rectangles = []
filter_centers_GUI(image_path)

scale_factor = 0.5
image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
centers,tmpImg = detect_circles(image)

cv2.imshow("Detected Circles", tmpImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

for center in centers:
    x, y = center
    print(f"Center: ({x}, {y})")

angle = calculate_angle_horizontal(centers[0], centers[1])
print(f"夹角：{angle} 度")

image = cv2.imread(image_Rotate_path)
# 增加对比度
#alpha = 2  # 对比度增益因子，大于1增加对比度，小于1减小对比度
#beta = -200   # 亮度调整参数，可以为0
#image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

#scale_factor = 1
#image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
center_x,center_y = centers[0]
#rotated_image = rotate_image(image, center_x - 200, center_y - 100, -90)
#cv2.imwrite(image_Rotate_path, rotated_image,[cv2.IMWRITE_JPEG_QUALITY, 100])
rotated_image = image
filter_centers_GUI(image_Rotate_path)

# 显示结果
#cv2.imshow("Detected Circles", rotated_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
scale_factor = 0.5
rotated_image = cv2.resize(rotated_image, (0, 0), fx=scale_factor, fy=scale_factor)

centers_rotated,tmpImg = detect_circles(rotated_image)
for center in centers_rotated:
    x, y = center
    print(f"rotated Center: ({x}, {y})")


points = [centers[0], centers[1], centers[2]]
rotated_points = [centers_rotated[0], centers_rotated[1], centers_rotated[2]]

# Initial guess for the center of rotation
x0_initial = (0, 0)
# Minimize the objective function
result = minimize(objective_func, x0_initial, args=(points, rotated_points,RotationAangle))
# The optimal center of rotation
x0_optimal = result.x
x0_optimal = (math.ceil(x0_optimal[0]), math.ceil(x0_optimal[1]))
print("Rotation Center: ", x0_optimal)
img = cv2.imread(image_path)

scale_factor = 1
img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)

cv2.circle(img, (int(x0_optimal[0]), int(x0_optimal[1])), 6, (0, 0, 255), -1)
cv2.putText(img, f"Calc Rotation Center: ({int(x0_optimal[0])}, {int(x0_optimal[1])})",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#cv2.putText(img, f"Rotation Center: ({int(center_x - 200)}, {int(center_y - 50)})",
#            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imwrite(image_output_path, img,[cv2.IMWRITE_JPEG_QUALITY, 100])
scale_factor = 0.5
img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)

cv2.imshow('Image with Rotation Center', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

def save_rotation_center(filename, rotation_center):
    #print(f"Rotation Center: {rotation_center[0]}, {rotation_center[1]}")
    with open(filename, 'w') as f:
        f.write(f"Rotation Center: {rotation_center[0]}, {rotation_center[1]}")

# 调用函数保存旋转中心坐标到文本文件
#save_rotation_center(RootPath + 'rotation_center.txt', x0_optimal)

#保存旋转中心坐标
config = configparser.ConfigParser()
# 设置值
config['rotation_center'] = {CameraID : f"{x0_optimal[0]}, {x0_optimal[1]}"}
# 写入文件
with open(RootPath + 'CalcConfig.ini', 'w') as configfile:
    config.write(configfile)