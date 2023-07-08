import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.optimize import minimize
#r'D:\Project3\ICT Automation\picture\6\initial\1.png'
# 读取图片并进行预处理

rect = (0, 0, 0, 0)
ix, iy, drawing = 0,0,False

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

def detect_circles(image):
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
    cv2.imshow("edges Circles", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
                
                print(f"Circle center: {center}, radius: {radius}")
                
                # 在图像上绘制圆形和中心点
                if radius > 30:
                    if(count == 0):
                        cv2.circle(output, center, int(radius), (0, 0, 255), 2)
                    if(count == 1):
                        cv2.circle(output, center, int(radius), (0, 255, 0), 2)
                    if(count == 2):
                        cv2.circle(output, center, int(radius), (255, 0, 0), 2)
                    if(count > 2):
                        cv2.circle(output, center, int(radius), (255, 255, 0), 2)
                    cv2.circle(output, center, 1, (0, 0, 255), 3)
                    center_list.append(center)  # 将圆心坐标添加到列表中
                    count = count + 1
                

    # 显示结果
    cv2.imshow("Detected Circles", output)
    #big_height, big_width, big_channels = output.shape
    #cv2.resizeWindow('Detected Circles', int(big_width / 4), int(big_height/4))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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


# 定义鼠标事件回调函数
def draw_rect(event, x, y, flags, param):
    global rect,ix, iy, drawing

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
        #cv2.destroyAllWindows()
        #imgshowFlag = True

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

def objective_func(x0, points, rotated_points):
    """
    Objective function to be minimized to find the center of rotation.
    """
    x0, y0 = x0
    error = 0
    for (x, y), (xp, yp) in zip(points, rotated_points):
        xr, yr = rotate_point(x, y, np.pi/2, x0, y0)  #如果是90度，这里改成 np.pi/2
        error += (xr - xp)**2 + (yr - yp)**2
    return error




#####################################################################
# 读取图像
#image = cv2.imread(r'D:\Project3\ICT Automation\picture\flex\25mm\Image_20230601095800134.bmp')
image = cv2.imread(r'D:\Image\25mm\Image_20230601095829893.bmp', cv2.IMREAD_COLOR)
scale_factor = 0.3
image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
# 创建窗口
cv2.namedWindow('image')
# 绑定鼠标事件
cv2.setMouseCallback('image', draw_rect)


# 显示图片
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Read the image


#image = cv2.imread(r'D:\Project3\ICT Automation\picture\flex\25mm\Image_20230601095843150.bmp', cv2.IMREAD_COLOR)
#scale_factor = 0.3
#image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# Detect lines using HoughLines
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
Boardlines = []

x, y, w, h = rect[0], rect[1], rect[2], rect[3]
# 遍历所有直线
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    #x1 = int(x0 + 1000 * (-b))
    #y1 = int(y0 + 1000 * (a))
    #x2 = int(x0 - 1000 * (-b))
    #y2 = int(y0 - 1000 * (a))

    # 直线的长度
    line_length = max(image.shape[0], image.shape[1])

    # 直线的起点坐标
    pt1 = (int(x0 - line_length * b), int(y0 + line_length * a))

    # 直线的终点坐标
    pt2 = (int(x0 + line_length * b), int(y0 - line_length * a))
    print('顶点坐标1：({},{})，顶点坐标2：({},{})'.format(pt1[0], pt1[1], pt2[0], pt2[1]))
    # 在图像上绘制直线
    if line_intersects_rectangle(pt1,pt2,x, y, w, h):
        cv2.line(image, pt1, pt2, (0, 0, 255), 1)
        Boardlines.append([pt1, pt2])
    #cv2.line(image, pt1, pt2, (0, 0, 255), 1)
LineAngle = calculate_angle_line(Boardlines[0],Boardlines[1])
print(str(LineAngle))

filtered_lines = find_lines_with_angle(Boardlines, 45)
# 计算直线的角度
angle_rad = np.arctan2(filtered_lines[0][1][1] - filtered_lines[0][0][1], filtered_lines[0][1][0] - filtered_lines[0][0][0])
angle_deg = np.degrees(angle_rad)
image = rotate_image_to_horizontal(image, [20,20], angle_deg)

# 显示图片
text = f"Angle: {LineAngle:.2f}"
cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
text = f"Angle: {angle_deg:.2f}"
cv2.putText(image, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

