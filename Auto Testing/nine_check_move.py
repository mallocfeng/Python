# -*- coding: utf-8 -*-
import sys
# sys.path.append('C:\Yandle\Develop\Auto Testing\lib64')
# sys.path.append('C:\Yandle\Develop\Auto Testing\lib64-2.1.4')
import time 

from util import get_dir_path

current_directory = get_dir_path()
lib_path_jeka = current_directory
sys.path.append(lib_path_jeka) 
import jkrc

import get_robot_status
import argparse


parser = argparse.ArgumentParser()
# model 型号
parser.add_argument('-p', '--p', help='Point No.')

args = parser.parse_args()

PI=3.1415926

#运动模式
ABS = 0
INCR= 1

#运动速度
SPEED_LOW = 10
SPEED_MID = 20
SPEED_HIGH = 80

NINE_MOVE_SLEEP_TIME = 1

robot = jkrc.RC(get_robot_status.get_jeka_ip()) #返回一个机器人对象
robot.login()  	  #登录

# 九点标定初始位置
step_nine_position = [-552.40555, 101.560426, 345.884097, -3.141593, -0.0, -0.409395]

def get_grid_coordinates():
    x, y, z = step_nine_position[:3]  # 获取前三个坐标值
    grid_coordinates = []
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            new_x = x + i * 10 + j * 3
            new_y = y + j * 10 + i * 3
            new_coordinate = [new_x, new_y, z, step_nine_position[3], step_nine_position[4], step_nine_position[5]]
            grid_coordinates.append(new_coordinate)

    order = [5, 4, 1, 2, 3, 6, 9, 8, 7]
    new_grid_coordinate = [grid_coordinates[i-1] for i in order]
    point_10 = [step_nine_position[0] - 5 - 7,step_nine_position[1] - 5 - 7,step_nine_position[2],step_nine_position[3],step_nine_position[4],step_nine_position[5]]
    new_grid_coordinate.insert(9, point_10)

    return new_grid_coordinate

def move_9_coordinates(index): 
    
    robot.power_on() #上电
    robot.enable_robot()
    # for coordinate, index in get_grid_coordinates():
    for idx, coordinate in enumerate(get_grid_coordinates()):
        # print(index)
        if index == idx:
            x, y, z, xr, yr, zr = coordinate
            # 在这里进行对每个坐标的进一步处理
            # print(f"Coordinate: {x}, {y}, {z}")
            ret = robot.linear_move(coordinate, ABS, True, SPEED_MID)
            time.sleep(0.5)
        
print(args.p)
if (args.p):
    index = int(args.p)
    move_9_coordinates(index)
    

robot.logout()


