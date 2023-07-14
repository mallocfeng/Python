import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.optimize import minimize
import sys
from TransCoordsToArms import transformCoordinateOffset,transformCoordinatePoint,transformCoordinateToImage
from CircleDetect import find_circles
from ninepointCalibration import map_space_to_pixel,map_pixel_to_space,getlArmPositionOffset,getlImgPositionOffset
#r'D:\Project3\ICT Automation\picture\6\initial\1.png'
# 读取图片并进行预处理

#rect = (0, 0, 0, 0)
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
        rect = (x*2, y*2, width*2, height*2)
        rectangles.append(rect)
        #cv2.destroyAllWindows()
        #imgshowFlag = True

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

def detect_circles(image,Radius = 50):
    circles, result_img = find_circles(image)
    center_list = []  # 存储圆心坐标的列表
    for circle in circles:
        if circle[2] > Radius:
            center_list.append((circle[0],circle[1]))
    center_list = filter_centers(center_list,rectangles)
    return center_list,result_img  # 返回圆心坐标列表和图片

def detect_circles1(image,Radius = 50):
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

##############################

def find_parallel_rotation_angle(centers_original, centers_Actual, rotationCenter):
    rotation_angle = 0.0
    step_size = 0.0005

    while rotation_angle <= 90.0:
        rotated_centers_original = rotate_line(centers_original, rotationCenter, -rotation_angle)
        if are_lines_parallel(rotated_centers_original, centers_Actual):
            return rotation_angle
        rotation_angle += step_size

    rotation_angle = 0.0
    step_size = 0.0005
    while rotation_angle <= 90.0:
        rotated_centers_original = rotate_line(centers_original, rotationCenter, rotation_angle)
        if are_lines_parallel(rotated_centers_original, centers_Actual):
            return -rotation_angle
        rotation_angle += step_size

    return None

def find_parallel_rotation_angle1(centers_original, centers_Actual, rotationCenter):
    # calculate the slopes of the lines
    line1 = np.array([centers_original[0], centers_original[1]])
    line2 = np.array([centers_Actual[0], centers_Actual[1]])
    m1 = (line1[1, 1] - line1[0, 1]) / (line1[1, 0] - line1[0, 0])
    m2 = (line2[1, 1] - line2[0, 1]) / (line2[1, 0] - line2[0, 0])

    # calculate the angle between the lines
    theta_rad = np.arctan(abs((m2 - m1) / (1 + m1 * m2)))

    # convert the angle to degrees
    theta_deg = np.rad2deg(theta_rad)
    return theta_deg



# 旋转直线
def rotate_line(line_points, rotationCenter, rotation_angle_degrees):
    rotated_points = []
    rotation_angle_radians = np.radians(rotation_angle_degrees)

    for point in line_points:
        x, y = point
        rotationCenter_x, rotationCenter_y = rotationCenter

        # 以rotationCenter为中心旋转点
        x_rotated = rotationCenter_x + (x - rotationCenter_x) * np.cos(rotation_angle_radians) - (y - rotationCenter_y) * np.sin(rotation_angle_radians)
        y_rotated = rotationCenter_y + (x - rotationCenter_x) * np.sin(rotation_angle_radians) + (y - rotationCenter_y) * np.cos(rotation_angle_radians)

        rotated_points.append((x_rotated, y_rotated))

    return rotated_points


# 判断两条直线是否平行
def are_lines_parallel(line1_points, line2_points, tolerance_degrees=0.001):
    #slope1 = abs(calculate_slope(line1_points[0], line1_points[1]))
    #slope2 = abs(calculate_slope(line2_points[0], line2_points[1]))
    slope1 = calculate_angle_horizontal(line1_points[0], line1_points[1])
    slope2 = calculate_angle_horizontal(line2_points[0], line2_points[1])

    return np.abs(slope1 - slope2) < tolerance_degrees


# 计算斜率
def calculate_slope(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return (y2 - y1) / (x2 - x1)

##############################


def rotate_points(centers_original, centers_Actual, rotation_center):
    # 判断是否垂直
    x1, y1 = centers_original[0]
    x2, y2 = centers_original[1]
    if x1 == x2:
        # 两个点在同一条垂直线上，选择一个固定的旋转角度（例如 90 度或 -90 度）
        rotation_angle = 90  # 或者选择 -90
    else:
        # 计算centers_original两个点相对于rotation_center的坐标偏移量
        dx_original = x1 - rotation_center[0]
        dy_original = y1 - rotation_center[1]

        # 计算centers_Actual两个点相对于rotation_center的坐标偏移量
        x1, y1 = centers_Actual[0]
        x2, y2 = centers_Actual[1]
        dx_actual = x1 - rotation_center[0]
        dy_actual = y1 - rotation_center[1]

        # 计算旋转角度
        rotation_angle = math.degrees(math.atan2(dy_actual, dx_actual) - math.atan2(dy_original, dx_original))

    # 进行旋转变换
    rotated_centers_original = []
    for point in centers_original:
        x, y = point
        # 将坐标平移到以rotation_center为原点
        x -= rotation_center[0]
        y -= rotation_center[1]
        # 进行旋转变换
        rotated_x = x * math.cos(math.radians(rotation_angle)) - y * math.sin(math.radians(rotation_angle))
        rotated_y = x * math.sin(math.radians(rotation_angle)) + y * math.cos(math.radians(rotation_angle))
        # 将坐标平移回原来的位置
        rotated_x += rotation_center[0]
        rotated_y += rotation_center[1]
        rotated_centers_original.append((rotated_x, rotated_y))

    return rotated_centers_original, rotation_angle


#################################


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


#框选矩形框，获得在矩形框里的圆心，其他不选择
def filter_centers_GUI(ImgPath):
    #rectangles.clear
    #global rectangles
    #rectangles = []
    image = cv2.imread(ImgPath, cv2.IMREAD_COLOR)
    scale_factor = 0.5
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


def read_rotation_center(filename):
    with open(filename, 'r') as f:
        line = f.readline()
        # 提取坐标信息
        coordinates = line.strip().split(':')[1].strip().split(',')
        # 转换为整数类型
        rotationCenter = (int(coordinates[0]), int(coordinates[1]))
        return rotationCenter

def rotate_point(centers_original,rotationCenter, F_Angle):
    """
    Rotate a point clockwise by a given angle around a given origin.
    The angle should be given in degrees.
    """
    # Convert angle to radians
    theta = -np.radians(F_Angle)  # negative sign for clockwise rotation
    
    # Shift the point so that the rotation center is at the origin
    x_shifted = centers_original[0] - rotationCenter[0]
    y_shifted = centers_original[1] - rotationCenter[1]

    # Apply the rotation about the origin
    x_rotated = x_shifted * np.cos(theta) - y_shifted * np.sin(theta)
    y_rotated = x_shifted * np.sin(theta) + y_shifted * np.cos(theta)

    # Shift the point back to its original location
    x_new = x_rotated + rotationCenter[0]
    y_new = y_rotated + rotationCenter[1]
    
    return x_new, y_new



#获取参数
# arg1 = sys.argv[1]
# arg2 = sys.argv[2]
# arg3 = sys.argv[3]

# arg1 = '2'
# arg2 = r'D:\Image\25mm\Image__2023-06-19__17-04-07.bmp'
# arg2 = r'D:\Image\25mm\ActualPic.jpg'


arg1 = sys.argv[1]
arg2 = sys.argv[2]
arg3 = sys.argv[3]
arg4 = sys.argv[4]

# arg1 = "2"
# arg2 = "D:\\Image\\25mm\\Actual.bmp"
# arg3 = "D:\\Image\\25mm\\"
# arg4 = "Step1"

#img = cv2.imread(r'D:\Image\25mm\Image_20230601095800134.bmp')
#Image_20230601095843150.bmp
#打开原始图像
#RootPath = "D:\\Image\\25mm\\"
#originalImgPath = RootPath + 'Image__2023-06-19__17-04-07.bmp'
#originalImgPath = arg2


CheckMode = arg1
RootPath = arg3
StepName = arg4

Camera1calibrationFilePath = [RootPath + "image_coords_O.txt",RootPath + "arm_coords_O.txt"]
Camera2calibrationFilePath = [RootPath + "image_coords_fixture.txt",RootPath + "arm_coords_fixture.txt"]
#originalImgPath = arg2
#ActualPicPath = arg2
#sys.exit(1)
#获得旋转中心
rotationCenter = read_rotation_center(RootPath +'rotation_center_Camera1.txt')
#destinationPos = [-555.40555,91.560426]



#offset = [1551, 647]
CenterArmPosition = transformCoordinatePoint(Camera1calibrationFilePath[0],Camera1calibrationFilePath[1], rotationCenter)
print(CenterArmPosition)
CenterArmPosition = map_pixel_to_space(Camera1calibrationFilePath[1],Camera1calibrationFilePath[0],rotationCenter)
print(CenterArmPosition)
ActualArmPosition = [-510.040754, 160.754379]

OffsetArmCenterValue = getlArmPositionOffset(Camera1calibrationFilePath[1],Camera1calibrationFilePath[0],rotationCenter,ActualArmPosition)
#OffsetImgCenterValue = getlImgPositionOffset(Camera1calibrationFilePath[1],Camera1calibrationFilePath[0],ActualArmPosition,rotationCenter)

#ActualArmPosition = [-75.527,-449.735]
#ActualArmPosition = [-78.651, -416.377]
#OffsetArmCenterValue = [ ActualArmPosition[0] - CenterArmPosition[0], ActualArmPosition[1] - CenterArmPosition[1]]
#OffsetArmCenterValue = getlArmPositionOffset(Camera1calibrationFilePath[1],Camera1calibrationFilePath[0],rotationCenter,ActualArmPosition)
CenterArmPosition = map_pixel_to_space(Camera1calibrationFilePath[1],Camera1calibrationFilePath[0],rotationCenter,OffsetArmCenterValue)
print(CenterArmPosition)

#第一次用于纠偏机械臂拍照位置
step1_Position = [-540.915615, 103.134551]
#step1_Position = [-75.527,-449.735]
step1_ImageCenterPoint = map_space_to_pixel(Camera1calibrationFilePath[1],Camera1calibrationFilePath[0],step1_Position,OffsetArmCenterValue)
print(step1_ImageCenterPoint)

ttt = map_pixel_to_space(Camera1calibrationFilePath[1],Camera1calibrationFilePath[0],step1_ImageCenterPoint,OffsetArmCenterValue)

# step1_ArmCenterPoint = [step1_Position[0] - OffsetArmCenterValue[0],step1_Position[1] - OffsetArmCenterValue[1]]
# step1_ImageCenterPoint = transformCoordinateToImage(Camera1calibrationFilePath[1],Camera1calibrationFilePath[0], step1_ArmCenterPoint)
#print(step1_ImageCenterPoint)
# step1_ImageCenterPoint = map_space_to_pixel(Camera1calibrationFilePath[1],Camera1calibrationFilePath[0],step1_Position,OffsetArmCenterValue)
# print(step1_ImageCenterPoint)
rotationCenter = step1_ImageCenterPoint


#print("ImageCenterPoint :",rotationCenter)
#rotationCenter = [-1000,500]



rotationCenter2 = read_rotation_center(RootPath +'rotation_center_Camera2.txt')

CenterArmPosition2 = transformCoordinatePoint(Camera2calibrationFilePath[0],Camera2calibrationFilePath[1], rotationCenter2)
print(CenterArmPosition2)

ActualArmPosition2 = [-75.527,-449.735]

OffsetArmCenterValue2 = getlArmPositionOffset(Camera2calibrationFilePath[1],Camera2calibrationFilePath[0],rotationCenter2,ActualArmPosition2)
#OffsetImgCenterValue2 = getlImgPositionOffset(Camera2calibrationFilePath[1],Camera2calibrationFilePath[0],ActualArmPosition2,rotationCenter2)


#OffsetArmCenterValue2 = getlArmPositionOffset(Camera2calibrationFilePath[1],Camera2calibrationFilePath[0],rotationCenter2,ActualArmPosition2)
CenterArmPosition2 = map_pixel_to_space(Camera2calibrationFilePath[1],Camera2calibrationFilePath[0],rotationCenter2,OffsetArmCenterValue2)
print(CenterArmPosition2)

#第二次用于纠偏机械臂拍照位置
step2_Position = [-18.2342,-458.4167]
step2_ImageCenterPoint = map_space_to_pixel(Camera2calibrationFilePath[1],Camera2calibrationFilePath[0],step2_Position,OffsetArmCenterValue2)
print(step2_ImageCenterPoint)
rotationCenter2 = step2_ImageCenterPoint

def fixtureToCamera1_IMGPorint(point):
    Std_IMG_FixtureCirclePoint = point
    Std_ARM_FixtureCirclePoint = map_pixel_to_space(Camera2calibrationFilePath[1],Camera2calibrationFilePath[0],Std_IMG_FixtureCirclePoint,OffsetArmCenterValue2)
    #夹具位置标定点的pixel位置转换成Camera1的像素坐标
    Std_IMG_FixtureCirclePoint_Camera = map_space_to_pixel(Camera1calibrationFilePath[1],Camera1calibrationFilePath[0],Std_ARM_FixtureCirclePoint,OffsetArmCenterValue2)
    return Std_IMG_FixtureCirclePoint_Camera

#夹具位置标定点的pixel位置
# Original CircleCenter: (1830, 744)
# Original CircleCenter: (1363, 433)
#######################################################################


#Std_ARM_FixtureCirclePoint1_Actual_New = rotate_point(Std_ARM_FixtureCirclePoint1_Actual,step1_Position,-F_Angle)
#print(Std_ARM_FixtureCirclePoint1_Actual_New)
#xyOffsert = [Std_ARM_FixtureCirclePoint1[0] - Std_ARM_FixtureCirclePoint1_Actual_New[0],Std_ARM_FixtureCirclePoint1[1] - Std_ARM_FixtureCirclePoint1_Actual_New[1]]

#######################################################################

# Std_IMG_FixtureCirclePoint1 = [1830, 744]
# Std_IMG_FixtureCirclePoint2 = [1363, 433]
# #夹具位置标定点的pixel位置转换成机械臂坐标
# # Std_ARM_FixtureCirclePoint1 = map_pixel_to_space(Camera2calibrationFilePath[1],Camera2calibrationFilePath[0],rotationCenter2,OffsetArmCenterValue2)
# # Std_ARM_FixtureCirclePoint2 = map_pixel_to_space(Camera2calibrationFilePath[1],Camera2calibrationFilePath[0],rotationCenter2,OffsetArmCenterValue2)
# #夹具位置标定点的pixel位置转换成Camera1的像素坐标
# # Std_IMG_FixtureCirclePoint1_Camera1 = map_space_to_pixel(Camera1calibrationFilePath[1],Camera1calibrationFilePath[0],Std_ARM_FixtureCirclePoint1,OffsetArmCenterValue)
# # Std_IMG_FixtureCirclePoint1_Camera1 = map_space_to_pixel(Camera1calibrationFilePath[1],Camera1calibrationFilePath[0],Std_ARM_FixtureCirclePoint2,OffsetArmCenterValue)
# Std_IMG_FixtureCirclePoint1_Camera1 = fixtureToCamera1_IMGPorint(Std_IMG_FixtureCirclePoint1)
# Std_IMG_FixtureCirclePoint2_Camera1 = fixtureToCamera1_IMGPorint(Std_IMG_FixtureCirclePoint2)

# # Original CircleCenter: (1732, 941)
# # Original CircleCenter: (1260, 637)
# Std_IMG_FixtureCirclePoint1_Actual = [1732, 941]
# Std_IMG_FixtureCirclePoint2_Actual = [1260, 637]
# Std_IMG_FixtureCirclePoint1_Actual_Camera1 = fixtureToCamera1_IMGPorint(Std_IMG_FixtureCirclePoint1_Actual)
# Std_IMG_FixtureCirclePoint2_Actual_Camera1 = fixtureToCamera1_IMGPorint(Std_IMG_FixtureCirclePoint2_Actual)
# centers_original = [Std_IMG_FixtureCirclePoint1_Camera1,Std_IMG_FixtureCirclePoint2_Camera1]
# centers_Actual =[Std_IMG_FixtureCirclePoint1_Actual_Camera1,Std_IMG_FixtureCirclePoint2_Actual_Camera1]
# F_Angle = find_parallel_rotation_angle(centers_original,centers_Actual,rotationCenter)
# #img = rotate_image_to_horizontal(img, rotationCenter, -F_Angle)
# x_new, y_new = rotate_point(centers_Actual[0],rotationCenter,-F_Angle)
# xyOffsert = [centers_original[0][0] - x_new,centers_original[0][1] - y_new]
# print(xyOffsert)
# #计算旋转中心加上偏移后机械臂的xy坐标

# #ActualOffsetArmXY = transformCoordinateOffset(Camera1calibrationFilePath[0],Camera1calibrationFilePath[1], xyOffsert)
# ActualOffsetArmXY = [xyOffsert[0]/32.2,xyOffsert[1]/32.2]
# print(ActualOffsetArmXY)
# step1_ArmCenterPoint = [step1_Position[0] - OffsetArmCenterValue[0],step1_Position[1] - OffsetArmCenterValue[1]]
# step1_ImageCenterPoint = transformCoordinatePoint(Camera1calibrationFilePath[0],Camera1calibrationFilePath[1], step1_ArmCenterPoint)
# print(step1_ImageCenterPoint)

# StdArmPosition = map_pixel_to_space(Camera1calibrationFilePath[1],Camera1calibrationFilePath[0],[rotationCenter[0] - xyOffsert[0], rotationCenter[1] - xyOffsert[1]],OffsetArmCenterValue)
# ActualArmXY = [StdArmPosition[0] - step1_Position[0] ,StdArmPosition[1] - step1_Position[1] ]
# print(ActualArmXY)


ActualPicPath = RootPath + "Actual.bmp"

if CheckMode == '1':
    originalImgPath = arg2
    img = cv2.imread(originalImgPath)
    # 增加对比度
    #alpha = 3  # 对比度增益因子，大于1增加对比度，小于1减小对比度
    #beta = -200   # 亮度调整参数，可以为0
    #img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 显示图片
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 绘制点
    #cv2.circle(img, rotationCenter, radius=10, color=(0, 255, 255), thickness=-1)
    #print(RootPath + "output.bmp")
    #cv2.imwrite(RootPath + "output.bmp", img,[cv2.IMWRITE_JPEG_QUALITY, 100])
    #originalImgPath = RootPath + "output.bmp"
    
    #旋转图像
    #img = rotate_image_to_horizontal(img, rotationCenter, -2.467)
    #放大图像
    #scale_factor = 1.05
    #img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
    #平移图像
    #img = translate_image(img,10,30)
    # 保存图像
    #cv2.imwrite(ActualPicPath, img,[cv2.IMWRITE_JPEG_QUALITY, 100])
    #print(ActualPicPath)
    ######################################################################################

    ######################################################################################
    #读取标准图像
    #rectangles = []
    filter_centers_GUI(originalImgPath)
    #保存框选的定位点坐标信息
    with open(RootPath + 'PointRectangularArea_' + StepName + '.txt', 'w') as f:
        for rect in rectangles:
            f.write(f'{rect[0]},{rect[1]},{rect[2]},{rect[3]}\n')

    #rotationCenter = read_rotation_center(RootPath +'rotation_center.txt')
    img = cv2.imread(originalImgPath)
    
    centers_original,original_img = detect_circles(img)
   
        #text = f"Original CircleCenter: {x}, {y}"
        #cv2.putText(img, text, (10, TestLoation), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #TestLoation = TestLoation + 30

    # 打开文件以写入模式
    with open(RootPath +'Originalcenters_' + StepName + '.txt', 'w') as f:
        # 遍历中心点坐标列表
        for center_original in centers_original:
            x, y = center_original
            #x, y = rotate_point([x,y],rotationCenter,-0.8765)
            #x, y = x - (4.3102619470433865 * 32.2),y + (3.043215099502908 * 32.2)
            # 将坐标写入文件
            f.write(f"{x},{y}\n")
            print(f"Original CircleCenter: ({x}, {y})")
    sys.exit(0)

ActualPicPath = arg2
rectangles = []
with open(RootPath + 'PointRectangularArea_' + StepName + '.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        # 解析每一行数据，假设数据格式为 'x,y,width,height'
        x, y, width, height = map(int, line.strip().split(','))
        rect = (x, y, width, height)
        rectangles.append(rect)


# 打开文件以读取模式
with open(RootPath +'Originalcenters_' + StepName + '.txt', 'r') as f:
    # 逐行读取文件内容
    lines = f.readlines()


# 创建空列表用于存储坐标数据
centers_original = []

# 遍历文件中的每一行
for line in lines:
    # 移除行末尾的换行符
    line = line.strip()
    # 将坐标字符串拆分为x和y
    x, y = line.split(',')
    # 将x和y转换为整数或浮点数（根据需要）
    x = float(x)
    y = float(y)
    # 将坐标添加到列表中
    centers_original.append((x, y))

# 打印读取的坐标数据
for center_original in centers_original:
    x, y = center_original
    print(f"OriginalCircleCenter: ({x}, {y})")

#print(f"############")
#旋转中心

#filter_centers_GUI(ActualPicPath)

img = cv2.imread(ActualPicPath)
#alpha = 3  # 对比度增益因子，大于1增加对比度，小于1减小对比度
#beta = -200   # 亮度调整参数，可以为0
#img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# 显示图片
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

centers_Actual,img = detect_circles(img)
TestLoation = 30
PointAngle = 0

for center_Actual in centers_Actual:
    x, y = center_Actual
    print(f"Actual CircleCenter: ({x}, {y})")
    #text = f"Actual CircleCenter: {x}, {y}"
    #cv2.putText(img, text, (10, TestLoation), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #TestLoation = TestLoation + 30
#print(f"############")


if len(centers_Actual) >= 2:
    #F_Angle = calculate_rotation_angle(centers_Actual[0], centers_Actual[1], centers_original[0], centers_original[1], rotationCenter)
    #F_Angle = calculate_rotation_angle( centers_original[0], centers_original[1], centers_Actual[0], centers_Actual[1],rotationCenter)
    #print("tttt")

    #F_Angle = find_parallel_rotation_angle(centers_original,centers_Actual,rotationCenter)
    F_Angle = find_parallel_rotation_angle1(centers_original,centers_Actual,rotationCenter)
    #F_Angle = 0    
    #New_Centers_original,F_Angle = rotate_points(centers_original,centers_Actual,rotationCenter)
    #F_Angle = 180 - F_Angle
    #########################
    #旋转样本和实际图片标定平行
    #original_img = cv2.imread(RootPath + "LeftTop_0_2023-06-06 14_55_37_255.jpg")
    #original_img = rotate_image_to_horizontal(original_img, rotationCenter, F_Angle)
    #cv2.imwrite(RootPath + "output.jpg", original_img)

    #original_centers = rotate_line(centers_original, rotationCenter, F_Angle)
    #original_centers = centers_original
    #original_centers,original_img = detect_circles(original_img)
    # 计算 original_centers 两个点之间的距离
    distance_original = calculate_distance(centers_original[0], centers_original[1])
    # 计算 centers_Actual 两个点之间的距离
    distance_Actual = calculate_distance(centers_Actual[0], centers_Actual[1])
    # 计算缩放因子
    scaling_factor = calculate_scaling_factor(distance_Actual, distance_original)
    scale_factor = scaling_factor
    img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
    centers_Actual_scaling = centers_Actual
    #centers_Actual_scaling = centers_Actual
    for i in range(len(centers_Actual)):
        x, y = centers_Actual[i]
        centers_Actual_scaling[i] = (x*scale_factor, y*scale_factor)
    centers_Actual_scaling = rotate_line(centers_Actual_scaling, rotationCenter, F_Angle)
    #########################
    #F_Angle = calculate_rotation_angle(centers_Actual[0], centers_Actual[1], centers_original[0], centers_original[1], rotationCenter)
    #旋转+平移恢复图像
    img = rotate_image_to_horizontal(img, rotationCenter, -F_Angle)
    x_new, y_new = rotate_point(centers_Actual[0],rotationCenter,-F_Angle)
    xyOffsert = [centers_original[0][0] - x_new,centers_original[0][1] - y_new]
    img = translate_image(img,centers_original[0][0] - xyOffsert[0],centers_original[0][1] - xyOffsert[1])
    #xyOffsert = [xyOffsert[0]/32.2,xyOffsert[1]/32.2]

#保存最终图像并打印结果
#file_path = RootPath + "PointDistance_" + StepName + ".txt"
# 读取文件内容
#with open(file_path, 'r') as file:
#    content = file.read()

# 按逗号分隔内容，得到两个数值
#data = content.split(',')
#num1 = float(data[0].strip())
#num2 = float(data[1].strip())

# 打印提取的数值
#print("Num1:", num1)
#print("Num2:", num2)

#记录下相机第一次纠偏后xy的偏差
centers_ActualARM = map_pixel_to_space(Camera1calibrationFilePath[1],Camera1calibrationFilePath[0],centers_original[0],OffsetArmCenterValue)
New_centers_ActualARM = map_pixel_to_space(Camera1calibrationFilePath[1],Camera1calibrationFilePath[0],[x_new, y_new],OffsetArmCenterValue)
xyOffsert_forfixture = [New_centers_ActualARM[0] - centers_ActualARM[0],New_centers_ActualARM[1] - centers_ActualARM[1]]
First_Angle = F_Angle

#第一次用于纠偏机械臂拍照位置 机械臂的旋转中心
step1_Position = [-540.915615, 103.134551]

#[(1763, 1039), (1540, 607)]
Std_IMG_FixtureCirclePoint1 = [1628, 452]
Std_IMG_FixtureCirclePoint2 = [1840, 891]

#[(1665, 1207), (1441, 772)]
Std_IMG_FixtureCirclePoint1_Actual = [1441, 772]
Std_IMG_FixtureCirclePoint2_Actual = [1665, 1207]


FixtureLocation = [-18.2342,-458.4167]
FixtureLocation = [FixtureLocation[0] + xyOffsert_forfixture[0],FixtureLocation[1] + xyOffsert_forfixture[1]]

def LocationOffsetCalc(stdLocation,offsetLocation):
    return stdLocation
    #return [stdLocation[0] + offsetLocation[0],stdLocation[1] + offsetLocation[1] ]

#把贴在夹具上的两个定位点标准位置坐标转换成机械臂坐标
Std_ARM_FixtureCirclePoint1 = map_pixel_to_space(Camera2calibrationFilePath[1],Camera2calibrationFilePath[0],Std_IMG_FixtureCirclePoint1,OffsetArmCenterValue2)
Std_ARM_FixtureCirclePoint2 = map_pixel_to_space(Camera2calibrationFilePath[1],Camera2calibrationFilePath[0],Std_IMG_FixtureCirclePoint2,OffsetArmCenterValue2)
Std_ARM_FixtureCirclePoint1 = LocationOffsetCalc(Std_ARM_FixtureCirclePoint1,xyOffsert_forfixture)
Std_ARM_FixtureCirclePoint2 = LocationOffsetCalc(Std_ARM_FixtureCirclePoint2,xyOffsert_forfixture)
Std_ARM_FixturePorintLine =  [Std_ARM_FixtureCirclePoint1,Std_ARM_FixtureCirclePoint2]
print(Std_ARM_FixturePorintLine)
#把贴在夹具上的两个定位点实际位置坐标转换成机械臂坐标
Std_ARM_FixtureCirclePoint1_Actual = map_pixel_to_space(Camera2calibrationFilePath[1],Camera2calibrationFilePath[0],Std_IMG_FixtureCirclePoint1_Actual,OffsetArmCenterValue2)
Std_ARM_FixtureCirclePoint2_Actual = map_pixel_to_space(Camera2calibrationFilePath[1],Camera2calibrationFilePath[0],Std_IMG_FixtureCirclePoint2_Actual,OffsetArmCenterValue2)
Std_ARM_FixtureCirclePoint1_Actual = LocationOffsetCalc(Std_ARM_FixtureCirclePoint1_Actual,xyOffsert_forfixture)
Std_ARM_FixtureCirclePoint2_Actual = LocationOffsetCalc(Std_ARM_FixtureCirclePoint2_Actual,xyOffsert_forfixture)
Std_ARM_FixturePorintLine_Actual =  [Std_ARM_FixtureCirclePoint1_Actual,Std_ARM_FixtureCirclePoint2_Actual]
print(Std_ARM_FixturePorintLine_Actual)
#第一次纠偏后，新的夹具纠偏点坐标来计算角度
FixtureActual_Angle = find_parallel_rotation_angle1(Std_ARM_FixturePorintLine,Std_ARM_FixturePorintLine_Actual,FixtureLocation)

#Std_ARM_FixtureCirclePoint1 = rotate_point(Std_ARM_FixtureCirclePoint1,FixtureLocation,-FixtureActual_Angle)
#Std_ARM_FixtureCirclePoint1_Actual = rotate_point(Std_ARM_FixtureCirclePoint1_Actual,FixtureLocation,-FixtureActual_Angle)



Fixture_Angle = First_Angle + FixtureActual_Angle - 0.7
Std_ARM_FixtureCirclePoint1 = rotate_point(Std_ARM_FixtureCirclePoint1,FixtureLocation,-FixtureActual_Angle)
# Std_ARM_FixtureCirclePoint2 = rotate_point(Std_ARM_FixtureCirclePoint2,FixtureLocation,-Fixture_Angle)
# Std_ARM_FixtureCirclePoint1_Actual = rotate_point(Std_ARM_FixtureCirclePoint1_Actual,FixtureLocation,-Fixture_Angle)
# Std_ARM_FixtureCirclePoint2_Actual = rotate_point(Std_ARM_FixtureCirclePoint2_Actual,FixtureLocation,-Fixture_Angle)

#Std_ARM_FixtureCirclePoint1 = rotate_point(Std_ARM_FixtureCirclePoint1,FixtureLocation,FixtureActual_Angle)
# Std_ARM_FixtureCirclePoint1 = rotate_point(Std_ARM_FixtureCirclePoint1,FixtureLocation,Fixture_Angle)
# Std_ARM_FixtureCirclePoint1_Actual = rotate_point(Std_ARM_FixtureCirclePoint1_Actual,FixtureLocation,Fixture_Angle)
ARM_FixtureCirclePoint1_Offset = [Std_ARM_FixtureCirclePoint1_Actual[0] - Std_ARM_FixtureCirclePoint1[0] , Std_ARM_FixtureCirclePoint1_Actual[1] - Std_ARM_FixtureCirclePoint1[1]]


# Std_ARM_FixtureCirclePoint1_New = rotate_point(Std_ARM_FixtureCirclePoint1,FixtureLocation,-Fixture_Angle)
# Std_ARM_FixtureCirclePoint1_Actual_New = rotate_point(Std_ARM_FixtureCirclePoint1_Actual,FixtureLocation,-Fixture_Angle)



#ARM_FixtureCirclePoint1_Offset = [Std_ARM_FixtureCirclePoint1_Actual_New[0] - Std_ARM_FixtureCirclePoint1_New[0],Std_ARM_FixtureCirclePoint1_Actual_New[1] - Std_ARM_FixtureCirclePoint1_New[1]]




cv2.imwrite(RootPath + "final.jpg", img,[cv2.IMWRITE_JPEG_QUALITY, 100])
if CheckMode == str(2):
    print(f"Rotation Angle: {Fixture_Angle}")
    x,y = rotationCenter
    print(f"Center of Rotation: {x},{y}")
    x,y = xyOffsert
    print(f"Offset: {x},{y}")
    print(f"RotationCenter:{rotationCenter}")
    x,y = transformCoordinateOffset(Camera1calibrationFilePath[0],Camera1calibrationFilePath[1], (x, y))
    x,y = x + ARM_FixtureCirclePoint1_Offset[0], y + ARM_FixtureCirclePoint1_Offset[1] 
    print(f"ArmLocation:{x},{y}")
    #print(f"ArmLocation1:{destinationPos[0] + x},{destinationPos[1] + y}")
    print(f"Scaling Ratio: {scale_factor}")
    print(f"Image Saved in " + RootPath + "final.jpg")
#else:
    #print(f"{destinationPos[0] + x},{destinationPos[1] + y},{F_Angle}")

#显示图像
#scale_factor = 0.5
#img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
#cv2.imshow('Image with Rotation Center', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# 显示标注后的图像
#cv2.imshow('Circles', img)
#cv2.resizeWindow('Circles', int(big_width / 4), int(big_height/4))
#cv2.waitKey(0)
#cv2.destroyAllWindows()

def getParaList(stepNo):
    return 1,2,3