# -*- coding: utf-8 -*-
import sys
# sys.path.append('C:\Yandle\Develop\Auto Testing\lib64')
#sys.path.append('C:\\Yandle\\Develop\\Auto Testing\\lib64-2.1.4')
import time    
import os

from util import get_dir_path

current_directory = get_dir_path()
lib_path_jeka = current_directory
sys.path.append(lib_path_jeka) 

import jkrc

import argparse
import configparser 

PI=3.1415926

#运动模式
ABS = 0
INCR= 1

#运动速度
SPEED_LOW = 10
SPEED_MID = 20
SPEED_HIGH = 80

NINE_MOVE_SLEEP_TIME = 1

parser = argparse.ArgumentParser()
# 配置文件完整路径 C:\\Yandle\\Develop\\Auto-Testing\\config\\model-A.ini'
parser.add_argument('-m', '--m', help='Model Config Path.')
# 相机编号  1/2
parser.add_argument('-c', '--c', help='Camera No.')
# 步骤名称
parser.add_argument('-title', '--title', help='Step Name.')
# 类型 1: 9点标定   2：旋转中心
parser.add_argument('-t', '--t', help='Rotate Offset')
# 点位 类型1时，1-9个点 类型2时，1：移动到初始位置，2：旋转
parser.add_argument('-p', '--p', help='Rotate Offset')
# 类型2时，需要传入角度(角度值)
parser.add_argument('-a', '--a', help='Rotate Angel')

args = parser.parse_args()



model_path = 'D:\\Image\\25mm\\model-A.ini'
# nine_move_name = 'Camera1_Nine'
# rotate_move_name = 'Camera1_Rotate'
# camera_arm_coords_name = 'camera1_arm_coords.txt'

# model_path = ''
nine_move_name = ''
rotate_move_name = ''
camera_arm_coords_name = ''

model_dir = os.path.dirname(model_path)
config_dir = 'config'


if args.m:
    model_path = args.m

if args.c == '1':
    camera_arm_coords_name = 'camera1_arm_coords.txt'
    nine_move_name = 'Camera1_Nine'
    rotate_move_name = 'Camera1_Rotate'
if args.c == '2':
    camera_arm_coords_name = 'camera2_arm_coords.txt'
    nine_move_name = 'Camera2_Nine'
    rotate_move_name = 'Camera2_Rotate'

# camera_image_coords_path = os.path.join(model_path, camera_image_coords_name)
camera_arm_coords_path = os.path.join(model_dir, camera_arm_coords_name)

# 九点标定初始位置 同时作为校准的标准位置
# step_nine_position = [-540.915615, 103.134551, 345.880005, 3.141593, 0.0, -0.950649]

def save_location_files():
    # 检查当前目录下是否存在配置文件
    if os.path.exists(camera_arm_coords_path) == False:
        location = get_step_location(nine_move_name)
        location = [float(num) for num in location.split(",")]
        list = get_grid_coordinates(location)
        with open(camera_arm_coords_path, "w") as file:
            for item in list:
                line = "[" +", ".join(str(x) for x in item) + "]\n"
                file.write(f"{line}")

def get_step_location(step_name): 
    # config_file = os.path.join(model_dir, config_dir)
    # 检查当前目录下是否存在配置文件
    if os.path.exists(model_path):
        # 创建配置解析器对象
        config = configparser.ConfigParser()
        config.read(model_path)

        # 对应步骤
        section_name = f'{step_name}'
        if config.has_section(section_name):
            return config.get(section_name, 'location')
        return ''
    else:
        print("Config file does not exist")
        return ''
    
def get_grid_coordinates(step_nine_position):
    x, y, z = step_nine_position[:3]  # 获取前三个坐标值
    grid_coordinates = []
    
    for i in range(-2, 1):
        for j in range(-2, 1):
            new_x = x + i * 7 + j * 3
            new_y = y + j * 7 + i * 3
            new_coordinate = [new_x, new_y, z, step_nine_position[3], step_nine_position[4], step_nine_position[5]]
            grid_coordinates.append(new_coordinate)

    order = [9, 8, 7, 5, 4, 1, 2, 3, 6]
    new_grid_coordinate = [grid_coordinates[i-1] for i in order]

    return new_grid_coordinate

def move_9_coordinates(index): 
    robot = jkrc.RC("10.5.5.100") #返回一个机器人对象
    robot.login()  	  #登录
    robot.power_on() #上电
    robot.enable_robot()
    
    location = get_step_location(nine_move_name)
    location = [float(num) for num in location.split(",")]
    list = get_grid_coordinates(location)
    # for coordinate, index in get_grid_coordinates():
    for idx, coordinate in enumerate(list):
        # print(index)
        if (index - 1) == idx:
            x, y, z, xr, yr, zr = coordinate
            # 在这里进行对每个坐标的进一步处理
            # print(f"Coordinate: {x}, {y}, {z}")
            ret = robot.linear_move(coordinate, ABS, True, SPEED_MID)
            time.sleep(0.5)
            robot.logout()
            return

def move_to_step_name(title): 
    robot = jkrc.RC("10.5.5.100") #返回一个机器人对象
    robot.login()  	  #登录
    robot.power_on() #上电
    robot.enable_robot()
    
    location = get_step_location(title)
    location = [float(num) for num in location.split(",")]
    ret = robot.linear_move(location, ABS, True, SPEED_MID)
    robot.logout()  

def move_to_rotate(point):
    robot = jkrc.RC("10.5.5.100") #返回一个机器人对象
    robot.login()  	  #登录
    robot.power_on() #上电
    robot.enable_robot()
    location = get_step_location(rotate_move_name)
    location = [float(num) for num in location.split(",")]
    #  单位：mm
    # 第一次拍照位置
    if point == 1:
        ret = robot.linear_move(location, ABS, True, SPEED_HIGH)
    elif point == 2:
        angel = int(args.a)
        if angel > 90 or angel < - 90:
            return
        angel_rad = angel / 180 * PI
        # 旋转后拍照位置-过度
        ret = robot.linear_move([0, 0, -40, 0, 0, angel_rad/2], INCR, True, SPEED_MID)
        ret = robot.linear_move([0, 0, 40, 0, 0, angel_rad/2], INCR, True, SPEED_MID)
    
    robot.logout()  

# save_location_files()
# args.title = 'Camera2_Board_Verify_1'
if args.title:
    step_title = args.title
    move_to_step_name(step_title)
else:
    type = int(args.t)
    point = int(args.p)
    if type == 1:
        move_9_coordinates(point)
    elif type == 2:
        move_to_rotate(point)

