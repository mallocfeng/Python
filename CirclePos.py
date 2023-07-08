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

def calculate_distance(center1, center2):
    x1, y1 = center1
    x2, y2 = center2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

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
            if 0.85 < circularity < 1.5:
            #if 0.7 < circularity < 1:
                # 计算最小外接圆
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                
                
                
                # 在图像上绘制圆形和中心点
                if radius > 50:
                    #if(count == 0):
                    #    cv2.circle(output, center, int(radius), (0, 0, 255), 2)
                    #if(count == 1):
                    #    cv2.circle(output, center, int(radius), (0, 255, 0), 2)
                    #if(count == 2):
                    #    cv2.circle(output, center, int(radius), (255, 0, 0), 2)
                    #if(count > 2):
                    #    cv2.circle(output, center, int(radius), (255, 255, 0), 2)
                    #cv2.circle(output, center, 1, (0, 0, 255), 3)
                    center_list.append(center)  # 将圆心坐标添加到列表中
                    print(f"Circle center: {center}, radius: {radius}")
                    count = count + 1
                

    # 显示结果
    #cv2.imshow("Detected Circles", output)
    #big_height, big_width, big_channels = output.shape
    #cv2.resizeWindow('Detected Circles', int(big_width / 4), int(big_height/4))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
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


def calculate_rotation_angle(center1, center2, center3, center4, centerC):
    # 将坐标系原点移到 centerC 处
    x1_shifted = center1[0] - centerC[0]
    y1_shifted = center1[1] - centerC[1]
    x2_shifted = center2[0] - centerC[0]
    y2_shifted = center2[1] - centerC[1]

    x3_shifted = center3[0] - centerC[0]
    y3_shifted = center3[1] - centerC[1]
    x4_shifted = center4[0] - centerC[0]
    y4_shifted = center4[1] - centerC[1]

    # 计算标准直线的斜率
    slope_standard = (y4_shifted - y3_shifted) / (x4_shifted - x3_shifted)

    # 计算旋转前的直线斜率
    slope_original = (y2_shifted - y1_shifted) / (x2_shifted - x1_shifted)

    # 计算旋转角度（以弧度为单位）
    rotation_angle_rad = math.atan(slope_standard) - math.atan(slope_original)

    # 将旋转角度转换为度数
    rotation_angle_deg = math.degrees(rotation_angle_rad)

    return rotation_angle_deg


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def calculate_scaling_factor(distance1, distance2):
    scaling_factor = distance2 / distance1
    return scaling_factor

#平移图像
def translate_image(image, shift_x, shift_y):
    # 获取图像宽度和高度
    height, width = image.shape[:2]

    # 定义平移矩阵
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    # 进行平移操作
    translated_image = cv2.warpAffine(image, M, (width, height))

    return translated_image

#img = cv2.imread(r'D:\Image\25mm\Image_20230601095800134.bmp')
#Image_20230601095843150.bmp
img = cv2.imread(r'D:\Image\25mm\Image_20230601095856758.bmp')
# 增加对比度
alpha = 2  # 对比度增益因子，大于1增加对比度，小于1减小对比度
beta = -200   # 亮度调整参数，可以为0
img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
cv2.imwrite(r"D:\Image\25mm\output.bmp", img)
#scale_factor = 1.0
#img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
big_height, big_width, big_channels = img.shape
centers,img = detect_circles(img)
#removed_element = centers.pop(0) 
TestLoation = 30
PointAngle = 0
for center in centers:
    x, y = center
    #print(f"CircleCenter: ({x}, {y})")
    text = f"CircleCenter: {x}, {y}"
    #cv2.putText(img, text, (10, TestLoation), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    TestLoation = TestLoation + 30

if len(centers) >= 2:
    PointAngle = calculate_angle_horizontal(centers[0],centers[1])
    text = f"LineAngle: {PointAngle:.2f}"
    cv2.putText(img, text, (10, TestLoation), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    Pointdistance = calculate_distance(centers[0],centers[1])
    text = f"PointDistance: {Pointdistance:.2f}"
    cv2.putText(img, text, (10, TestLoation + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    img = rotate_image_to_horizontal(img, [centers[1][0] + 100,centers[1][1] + 100], PointAngle-75)

    ###############################################
    #图像扩大1.0倍
    #scale_factor = 1.0
    #img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
    #平移图像
    img = translate_image(img,200,200)
    # 保存图像
    cv2.imwrite(r"D:\Image\25mm\output.bmp", img)
    img = cv2.imread(r"D:\Image\25mm\output.bmp")
    centers_2,img = detect_circles(img)
    TestLoation = 30
    PointAngle = 0
    for center_2 in centers_2:
        x, y = center_2
        print(f"CircleCenter: ({x}, {y})")
        text = f"CircleCenter: {x}, {y}"
        cv2.putText(img, text, (10, TestLoation), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        TestLoation = TestLoation + 30
    if len(centers_2) >= 2:
        F_Angle = calculate_rotation_angle(centers_2[0], centers_2[1], centers[0], centers[1], [centers[1][0] + 100,centers[1][1] + 100])
        img = rotate_image_to_horizontal(img, [centers[1][0] + 100,centers[1][1] + 100], -F_Angle)
        cv2.imwrite(r"D:\Image\25mm\rotateBack.bmp", img)
        centers_3,img = detect_circles(img)
        if len(centers_3) >= 2:
            # 计算 center1 和 center2 之间的距离
            distance1 = calculate_distance(centers[0], centers[1])
            # 计算 center3 和 center4 之间的距离
            distance2 = calculate_distance(centers_2[0], centers_2[1])
            # 计算缩放因子
            scaling_factor = calculate_scaling_factor(distance2, distance1)
            scale_factor = scaling_factor
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
            img = translate_image(img,centers[0][0] - centers_3[0][0],centers[0][1] - centers_3[0][1])


scale_factor = 0.5
img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
#以某个指定旋转中心旋转

# 显示标注后的图像
cv2.imshow('Circles', img)
#cv2.resizeWindow('Circles', int(big_width / 4), int(big_height/4))
cv2.waitKey(0)
cv2.destroyAllWindows()