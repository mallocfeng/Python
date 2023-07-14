import cv2
import numpy as np
from tkinter import filedialog
from tkinter import Tk
from CircleDetect import find_circles

# Global variables
ix, iy = -1, -1
rectangles = []
drawing = False

# Mouse callback function for drawing rectangles
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rectangles

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img_temp = img.copy()
            cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', img_temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rectangles.append(((ix, iy), (x, y)))
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow('image', img)


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

# Function to find circles in an image
def detect_circles(image,Radius = 20):
    circles, result_img = find_circles(image,90)
    center_list = []  # 存储圆心坐标的列表
    for circle in circles:
        if circle[2] > Radius:
            center_list.append((circle[0],circle[1]))
    center_list = filter_centers(center_list,rectangles)
    return center_list,result_img  # 返回圆心坐标列表和图片

# Create a Tk root widget
root = Tk()
root.withdraw() # Hide the main window

# Open file dialog to select the image file
file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.bmp")])

# Load the image
img = cv2.imread(file_path)

# Reduce the size of the image
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

# Create a window and bind the mouse callback function
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

# Show the image
cv2.imshow('image', img)
cv2.waitKey(0)

circles,result_img = find_circles(img)

# Process the rectangles to find circles


cv2.destroyAllWindows()
