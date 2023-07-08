import cv2
import numpy as np
from scipy.optimize import minimize
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

def draw_circle(img1, values):
    min = 100000
    max = 0
    global flagMax
    global flagmin
    flag = 0
    for i in values[0, :]:
        #print(str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]))
        #cv.circle(img1, (int(i[0]),int(i[1])),int(i[2]),(255,0,255),2)
        #cv.circle(img1, (int(i[0]),int(i[1])),2,(0,255,0),3)
        if(i[1] > max):
            max = i[1]
            flagMax = i
        if(i[1] < min):
            min = i[1]
            flagmin = i
        flag = flag + 1

    cv2.circle(img1, (int(flagMax[0]),int(flagMax[1])),int(flagMax[2]),(255,0,255),2)
    cv2.circle(img1, (int(flagMax[0]),int(flagMax[1])),2,(0,255,0),3)

    cv2.circle(img1, (int(flagmin[0]),int(flagmin[1])),int(flagmin[2]),(255,0,255),2)
    cv2.circle(img1, (int(flagmin[0]),int(flagmin[1])),2,(0,255,0),3)




# 读取图像
img = cv2.imread(r'D:\Image\Image__2023-06-12__13-50-23.bmp')

# 将图片转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对灰度图进行高斯模糊以去除噪声
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用霍夫圆变换来检测圆形
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                           param1=80, param2=80, minRadius=0, maxRadius=0)

# 将检测到的圆形转换为整数
circles = circles.astype(int)

# 在原始图像中标注圆形
if circles is not None:
    # 将圆形坐标和半径转换为整数
    circles = np.round(circles[0, :]).astype('int')

    # 遍历所有检测到的圆形
    circle_centers1 = []
    for (x, y, r) in circles:
        # 绘制圆形
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
        circle_centers1.append((x, y))
        # 在圆心位置添加文本标签，显示圆心坐标
        cv2.putText(img, f'({x}, {y})', (x - 30, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

for center in circle_centers1:
    print(center)

# 显示标注后的图像
cv2.imshow('Circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


img = cv2.imread(r'E:\Golden\PointTest\Image1_90.jpg')

# 将图片转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对灰度图进行高斯模糊以去除噪声
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用霍夫圆变换来检测圆形
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                           param1=80, param2=80, minRadius=0, maxRadius=0)

# 将检测到的圆形转换为整数
circles = circles.astype(int)

# 在原始图像中标注圆形
if circles is not None:
    # 将圆形坐标和半径转换为整数
    circles = np.round(circles[0, :]).astype('int')

    # 遍历所有检测到的圆形
    circle_centers2 = []
    for (x, y, r) in circles:
        # 绘制圆形
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
        circle_centers2.append((x, y))
        # 在圆心位置添加文本标签，显示圆心坐标
        cv2.putText(img, f'({x}, {y})', (x - 30, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

for center in circle_centers2:
    print(center)

# 显示标注后的图像
cv2.imshow('Circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#points = [(860, 574), (384, 953), (868, 1088)]
#rotated_points = [(351, 639), (729, 1113), (216, 1120)]

points = [circle_centers1[0], circle_centers1[1], circle_centers1[2]]
rotated_points = [circle_centers2[0], circle_centers2[1], circle_centers2[2]]

x = 2

if x == 1:
    theta = np.pi
    X = np.array([circle_centers1[0], circle_centers1[1], circle_centers1[2]])
    Y = np.array([circle_centers2[0], circle_centers2[1], circle_centers2[2]])
    
    U, _, V = np.linalg.svd(X.T @ Y)
    R = V.T @ U.T

    # 将x点坐标绕180度旋转
    x = np.array([x, y])
    x0_optimal = x @ R
else:
    # Initial guess for the center of rotation
    x0_initial = (0, 0)

    # Minimize the objective function
    result = minimize(objective_func, x0_initial, args=(points, rotated_points))

    # The optimal center of rotation
    x0_optimal = result.x

print("The center of rotation is at: ", x0_optimal)
img = cv2.imread(r'E:\Golden\PointTest\Image1.jpg')

cv2.circle(img, (int(x0_optimal[0]), int(x0_optimal[1])), 5, (0, 255, 255), -1)
cv2.putText(img, f"Rotation Center: ({int(x0_optimal[0])}, {int(x0_optimal[1])})",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


cv2.imshow('Image with Rotation Center', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
