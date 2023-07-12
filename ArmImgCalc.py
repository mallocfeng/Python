import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.optimize import minimize
import sys
from TransCoordsToArms import transformCoordinateOffset,transformCoordinatePoint,transformCoordinateToImage
from CircleDetect import find_circles
from ninepointCalibration import map_space_to_pixel,map_pixel_to_space,getlArmPositionOffset

def read_rotation_center(filename):
    with open(filename, 'r') as f:
        line = f.readline()
        # 提取坐标信息
        coordinates = line.strip().split(':')[1].strip().split(',')
        # 转换为整数类型
        rotationCenter = (int(coordinates[0]), int(coordinates[1]))
        return rotationCenter

arg1 = "2"
arg2 = "D:\\Image\\25mm\\Fixture\\Actual.bmp"
arg3 = "D:\\Image\\25mm\\Fixture\\"
arg4 = "Step1"

CheckMode = arg1
RootPath = arg3
StepName = arg4

#0下标是图像坐标，1小标是机械臂坐标
Camera1calibrationFilePath = [RootPath + "image_coords_O.txt",RootPath + "arm_coords_O.txt"]
Camera2calibrationFilePath = [RootPath + "image_coords_fixture.txt",RootPath + "arm_coords_fixture.txt"]

#第二次用于纠偏机械臂拍照位置
step2_Position = [-18.2342,-458.4167]
#第二次纠偏的图像机械臂中心点
rotationCenter2 = read_rotation_center(RootPath +'rotation_center_Camera2.txt')

#把图像坐标转换成机械臂坐标
ARM_rotationCenter2 = map_pixel_to_space(Camera2calibrationFilePath[1], Camera2calibrationFilePath[0], rotationCenter2)
print(ARM_rotationCenter2)