import cv2
import numpy as np
from tkinter import filedialog
from tkinter import Tk
from CircleDetect import find_circles
import sys
import os
# Global variables
ix, iy = -1, -1
rectangles = []
drawing = False

# Mouse callback function for drawing rectangles
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
        #rect = (x, y, width, height)
        rectangles.append(rect)
        #cv2.destroyAllWindows()
        #imgshowFlag = True


def filter_centers(center_list, rectangles_List):
    filtered_centers = []
    if len(rectangles_List) == 0:
        return center_list
    
    for rect in rectangles_List: 
        # 检查每个圆心坐标是否在任何矩形区域内
        inside_rectangle = False
        for center in center_list:
            x, y = center
            rect_x, rect_y, rect_width, rect_height = rect
            if rect_x <= x <= rect_x + rect_width and rect_y <= y <= rect_y + rect_height:
                inside_rectangle = True
                break
        # 如果圆心在矩形区域内，则将其添加到过滤后的列表中
        if inside_rectangle:
            filtered_centers.append(center)
    
    return filtered_centers

# Function to find circles in an image
def detect_circles(image,RadiusMin = 50,RadiusMax = 65,BinarizationP = 170):
    circles, result_img = find_circles(image,BinarizationP)
    center_list = []  # 存储圆心坐标的列表
    for circle in circles:
        if circle[2] > RadiusMin and circle[2] < RadiusMax:
            center_list.append((circle[0],circle[1]))
    center_list = filter_centers(center_list,rectangles)
    for center in center_list:
        cv2.circle(result_img, center, 1, (255, 0, 0), 10)
    
    return center_list,result_img  # 返回圆心坐标列表和图片

def read_xy_coordinates_from_file(file_path):
    xy_coordinates = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                x, y = map(int, line.split(","))
                xy_coordinates.append((x, y))

    return xy_coordinates


#D:\\Image\\25mm\\ActualBoardFixturePosCheck1.bmp -1 D:\\Image\\25mm\\StdBoardOnFixturePointPos1.txt 191

arg1 = sys.argv[1]  #加载初始图像
arg2 = sys.argv[2]  #需要检测的点的数量
arg3 = sys.argv[3]  #检测到的中心点后保存的文件路径，包括路径和文件名
arg4 = sys.argv[4]  #二值化图像分界数

# arg1 = "D:\\Image\\25mm\\ActualBoardFixturePosCheck1.bmp"
# arg2 = -1
# arg3 = "D:\\Image\\25mm\\StdBoardOnFixturePointPos1.txt"
# arg4 = 191


# print(arg1)
# print(arg2)
# print(arg3)
# print(arg4)

# arg1 = r"D:\Image\25mm\Std_Fixture.bmp"
# arg2 = 2
# arg3 = r"D:\Image\25mm\output.txt"
# arg4 = 200

# Create a Tk root widget
root = Tk()
root.withdraw() # Hide the main window

# Open file dialog to select the image file
#file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.bmp")])
file_path = arg1
pointQty = arg2
SaveFilePath = arg3
BinarizationPara = int(arg4)

#如果是check模式，只需要从StdBoardOnFixturePointPos1.txt
#和StdBoardOnFixturePointPos2.txt中读取坐标信息，画一个框
#赋值给rectangles，不需要弹出图片区域选择框
if(int(pointQty) == -1):
    xy_coordinates = read_xy_coordinates_from_file(SaveFilePath)
    x,y = xy_coordinates[0]
    print(f"DetectCenter: ({x}, {y})")
    rect = (x - 100, y - 100, 200, 200)
    #rect = (x, y, width, height)
    rectangles.append(rect)

    img = cv2.imread(file_path)
    circles,result_img = detect_circles(img,30,65,BinarizationPara)
    for center_original in circles:
        x, y = center_original
        print(f"CircleCenter: ({x}, {y})")
        break
    sys.exit(0)

# Load the image
img = cv2.imread(file_path)

# Reduce the size of the image
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

# Create a window and bind the mouse callback function
param = {'image': img}
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rect,param)

# Show the image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
img = cv2.imread(file_path)
#img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
circles,result_img = detect_circles(img,30,65,BinarizationPara)

# Process the rectangles to find circles
#with open(RootPath +'Originalcenters_' + StepName + '.txt', 'w') as f:
        # 遍历中心点坐标列表
try:
    os.remove(SaveFilePath)
    print("File deleted successfully.")
except OSError as e:
    print("Error deleting the file:", e)

if(len(circles) != int(pointQty)):
    sys.exit(1)

with open(SaveFilePath, 'w') as f:
    for center_original in circles:
        x, y = center_original
        #x, y = rotate_point([x,y],rotationCenter,-0.8765)
        #x, y = x - (4.3102619470433865 * 32.2),y + (3.043215099502908 * 32.2)
        # 将坐标写入文件
        f.write(f"{x},{y}\n")
        print(f"CircleCenter: ({x}, {y})")
f.close        
# result_img = cv2.resize(result_img, (img.shape[1] // 2, img.shape[0] // 2))
# cv2.imshow('image', result_img)
folder_path = os.path.dirname(arg1)
new_file_name = "output.bmp"
new_file_path = os.path.join(folder_path, new_file_name)
cv2.imwrite(new_file_path, result_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
