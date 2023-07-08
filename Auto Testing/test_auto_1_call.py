# -*- coding: utf-8 -*-
import sys
# sys.path.append('C:\Yandle\Develop\Auto Testing\lib64')
#sys.path.append('C:\\Yandle\\Develop\\Auto Testing\\lib64-2.1.4')
import time    
import tkinter as tk

from util import get_dir_path

current_directory = get_dir_path()
lib_path_jeka = current_directory
sys.path.append(lib_path_jeka) 

import jkrc

import argparse


parser = argparse.ArgumentParser()
# model 型号
parser.add_argument('-s', '--s', help='Step No.')
# x
parser.add_argument('-x', '--x', help='X Offset')
# y
parser.add_argument('-y', '--y', help='Y Offset')
# r
parser.add_argument('-r', '--r', help='Rotate Offset')

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

robot = jkrc.RC("10.5.5.100") #返回一个机器人对象
robot.login()  	  #登录

# 移动之前，需要移动到的高度
# position_move_top_ = 410
position_move_top = 170
# position_pick_bottom = 1.97500
position_pick_standby = 60
position_pick_bottom = 11.873173
# 目标吸取位置
step_pick_position = [-594.561328, -599.410647, 60, 3.141593, 0.0, -0.950649]
position_put_top = 170
position_put_bottom = 95.802602
# position_put_bottom = 92.802602
# 目标放置位置
# step_put_position = [-536.670292 - 5.6104, -595.610338 + 2.018, 63.511211, 3.141593, 0.0, -0.409395 - (2.062 / 180 * PI)]
step_put_position = [-18.234192, -458.416719, position_put_bottom, 3.141593, 0.0, -0.950649]
# 放置时的x,y, rz偏移量 mm,mm,rad
step_put_offset = [0, 0, 0]

# 第一次拍照位置 旋转中心
step_rotate_position = [-510.040754, 160.754379, 345.884097, -3.141593, 0.0, 1.161401]

# 九点标定初始位置 同时作为校准的标准位置
step_nine_position = [-540.915615, 103.134551, 345.880005, 3.141593, 0.0, -0.950649]

IO_CABINET = 0 #控制柜面板 IO
# IO_TOOL = 1 #工具 IO
# IO_EXTEND = 2 #扩展 IO

# 固定帘子
FIX_IO_INDEX = 0
# 吸取
INHALING_IO_INDEX = 1
# 吹出帘子
BLOW_IO_INDEX = 2


def open_io(index):
    robot.set_digital_output(IO_CABINET, index, 1)#设置 DO2 的引脚输出值为 1
    time.sleep(0.2)
    # ret = robot.get_digital_output(0, 2)
    # if ret[0] == 0:
    #     print("2the DO2 is :",ret[1])
    # else:
    #     print("some things happend,the errcode is: ",ret[0])

def close_io(index):
    robot.set_digital_output(IO_CABINET, index, 0)#设置 DO2 的引脚输出值为 1
    time.sleep(0.2)
    # ret = robot.get_digital_output(0, 2)
    # if ret[0] == 0:
    #     print("2the DO2 is :",ret[1])
    # else:
    #     print("some things happend,the errcode is: ",ret[0])

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

def get_current_location():
    ret = robot.get_robot_status()
    # robot.logout()
    if ret[0] == 0:
        position = ret[1][18]
        rounded_position = [round(x, 6) for x in position]
        print(rounded_position)
        return rounded_position
    else:
        print("!!EORROR!! something wrong happend , errorcode is :",ret[0])
        return []
    
def get_replace_locations(locations, point):
    locations[2] = point
    return [locations[0], locations[1], point, locations[3], locations[4], locations[5]]
    
def move_to_top():
    locations = get_current_location()
    if len(locations) == 6 :
        locations = get_replace_locations(locations, position_move_top)
        print(locations)

        #  单位：mm
        # robot.login()#登录
        robot.power_on() #上电
        robot.enable_robot()
        # print("move1")
        #阻塞 沿z轴负方向 以20mm/s 运动60mm
        ret = robot.linear_move(locations,ABS,True,SPEED_HIGH)
        # print(ret[0])
        time.sleep(1)
    
def move_to_pick():
    # move_to_top()
    #  单位：mm
    # robot.login()#登录
    robot.power_on() #上电
    robot.enable_robot()
    # print("move1")
    #阻塞 沿z轴负方向 以20mm/s 运动60mm

    step_pick_position_up = get_replace_locations(step_pick_position, position_move_top)
    ret = robot.linear_move(step_pick_position_up,ABS,True,SPEED_HIGH)
    time.sleep(0.5)
    
    step_pick_position_standby = get_replace_locations(step_pick_position, position_pick_standby)
    ret = robot.linear_move(step_pick_position_standby,ABS,True,SPEED_HIGH)
    # print(ret[0])
    time.sleep(1)
    open_io(BLOW_IO_INDEX)
    time.sleep(0.5)
    close_io(BLOW_IO_INDEX)
    time.sleep(0.5)

    step_pick_position_down = get_replace_locations(step_pick_position, position_pick_bottom)
    ret = robot.linear_move(step_pick_position_down,ABS,True,SPEED_LOW)
    # print(ret[0])
    time.sleep(1)
    
    open_io(FIX_IO_INDEX)
    time.sleep(0.5)
    
    open_io(INHALING_IO_INDEX)
    time.sleep(0.5)

    step_pick_position_up = get_replace_locations(step_pick_position, position_move_top)
    ret = robot.linear_move(step_pick_position_up,ABS,True,SPEED_HIGH)
    time.sleep(1)
    

def move_to_rotate():
    #  单位：mm
    # robot.login()#登录
    robot.power_on() #上电
    robot.enable_robot()
    #阻塞 沿z轴负方向 以20mm/s 运动60mm
    # 第一次拍照位置
    ret = robot.linear_move(step_rotate_position, ABS, True, SPEED_HIGH)
    time.sleep(1)
    # 旋转后拍照位置-过度
    # ret = robot.linear_move([0, 0, -40, 0, 0, 0.785398], INCR, True, SPEED_MID)
    # time.sleep(0.1)
    # # 旋转后拍照位置
    # ret = robot.linear_move([0, 0, 40, 0, 0, 0.785398], INCR, True, SPEED_MID)
    # time.sleep(0.5)
    # # 旋转后拍照位置
    # # ret = robot.linear_move(step_rotate_position_rotate_90, ABS, True, SPEED_HIGH)
    # # ret = robot.linear_move([0, 0, 0, 0, 0, 1.570796],INCR,True,0.1)
    time.sleep(1)

def move_9_coordinates(): 
    robot.power_on() #上电
    robot.enable_robot()
    # for coordinate, index in get_grid_coordinates():
    for index, coordinate in enumerate(get_grid_coordinates()):
        # print(index)
        x, y, z, xr, yr, zr = coordinate
        # 在这里进行对每个坐标的进一步处理
        # print(f"Coordinate: {x}, {y}, {z}")
        ret = robot.linear_move(coordinate, ABS, True, SPEED_MID)
        time.sleep(NINE_MOVE_SLEEP_TIME)

def move_to_validate(): 
    robot.power_on() #上电
    robot.enable_robot()
    ret = robot.linear_move(step_nine_position, ABS, True, SPEED_HIGH * 2)
    time.sleep(0.5)
    # global step_put_offset

    # input_rotate = float(input("请输入旋转角度（度数）："))
    # # 创建 step_put_offset 列表并填入用户输入的数值
    # step_put_offset = [0, 0, input_rotate / 180 * PI]
    # ret = robot.linear_move([0, 0, 0, 0, 0, step_put_offset[2]], INCR, True, SPEED_HIGH)
    
    # input_offset_x = float(input("请输入X轴偏移量："))
    # input_offset_y = float(input("请输入Y轴偏移量："))
    # step_put_offset = [input_offset_x, input_offset_y, input_rotate / 180 * PI]

def move_to_validate_rotate(input_rotate): 
    robot.power_on() #上电
    robot.enable_robot()

    global step_put_offset
    # 创建 step_put_offset 列表并填入用户输入的数值
    step_put_offset = [0, 0, input_rotate / 180 * PI]
    ret = robot.linear_move([0, 0, 0, 0, 0, step_put_offset[2]], INCR, True, SPEED_HIGH)
        

def move_to_put(input_offset_x, input_offset_y, input_rotate):
    
    global step_put_offset
    step_put_offset = [input_offset_x, input_offset_y, input_rotate / 180 * PI]
    #  单位：mm
    # robot.login()#登录
    robot.power_on() #上电
    robot.enable_robot()
    # print("move1")
    #阻塞 沿z轴负方向 以20mm/s 运动60mm

    offset_step_put_position = [step_put_position[0] - step_put_offset[0],
                                step_put_position[1] - step_put_offset[1],
                                step_put_position[2],
                                step_put_position[3],
                                step_put_position[4],
                                step_put_position[5] + step_put_offset[2],
                                ]
    # 移动到夹具上方
    step_put_position_top = get_replace_locations(offset_step_put_position, position_put_top)
    ret = robot.linear_move(step_put_position_top,ABS,True,SPEED_HIGH)
    time.sleep(1)
    
    # step_put_position_down = get_replace_locations(offset_step_put_position, position_put_bottom)
    # ret = robot.linear_move(step_put_position_down,ABS,True,SPEED_LOW)
    # # print(ret[0])
    # time.sleep(1)
    
    # close_io(INHALING_IO_INDEX)
    # time.sleep(0.5)

    # step_put_position_up = get_replace_locations(step_put_position, position_move_top)
    # ret = robot.linear_move(step_put_position_up,ABS,True,SPEED_MID)
    # time.sleep(0.5)
    
    # close_io(FIX_IO_INDEX)

    # step_pick_position_up = get_replace_locations(step_pick_position, position_move_top)
    # ret = robot.linear_move(step_pick_position_up,ABS,True,SPEED_HIGH)
    # time.sleep(0.5)
    

if (args.s):
    step = int(args.s)
    if step == 1:
        move_to_pick()
    elif step == 2:
        move_to_validate()  
    elif step == 3:
        r = float(args.r)
        if(r < 30 and r > -30):
            move_to_validate_rotate(r)  
    elif step == 4:
        x = float(args.x)
        y = float(args.y)
        r = float(args.r)
        #if((x < 15 and x > -15) and (y < 15 and y > -15) (r < 15 and r > -15)):
        move_to_put(x, y, r)
       

robot.logout()


