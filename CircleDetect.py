import cv2
import numpy as np


def find_circles(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    _, binary_img = cv2.threshold(img, 158, 255, cv2.THRESH_BINARY)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    
    # 找轮廓
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建结果图像
    result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 存储圆的信息
    circles = []
    
    # 遍历轮廓
    for contour in contours:
        if len(contour) < 5:
            continue
        area = cv2.contourArea(contour)
        if area < 2000:
            continue
        arc_length = cv2.arcLength(contour, True)
        radius = arc_length / (2 * np.pi)
        if not (10 < radius < 300):
            continue
        rect = cv2.fitEllipse(contour)
        ratio = float(rect[1][0]) / float(rect[1][1])
        if 0.9 < ratio < 1.1:
            cv2.ellipse(result_img, rect, (0, 255, 255), 2)
            cv2.circle(result_img, (int(rect[0][0]), int(rect[0][1])), 2, (0, 255, 0), 2)
            cv2.putText(result_img, f'Radius: {radius:.2f}', (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            circles.append((int(rect[0][0]), int(rect[0][1]), radius))
    return circles, result_img

# 读取灰度图像
#src_img = cv2.imread(, cv2.IMREAD_GRAYSCALE)

# img = cv2.imread(r'D:\Image\25mm\Std.bmp', cv2.IMREAD_GRAYSCALE)
# circles, result_img = find_circles(img)
# # 缩小图像
# result_img = cv2.resize(result_img, (result_img.shape[1] // 2, result_img.shape[0] // 2))
# # 打印所有找到的圆的信息
# for circle in circles:
#     print(f'Center: {circle[0]}, Radius: {circle[1]}')

# # 显示结果图像
# cv2.imshow('Result', result_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()