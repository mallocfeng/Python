# -*- coding: utf-8 -*-
import sys
sys.path.append('C:\Yandle\Develop\Auto Testing\lib64-2.1.4')
import time      
import jkrc

PI=3.1415926
#运动模式
ABS = 0
INCR= 1
#  单位：mm
position = []
robot = jkrc.RC("10.5.5.100") #返回一个机器人对象
robot.login()#登录
ret = robot.get_robot_status()
if ret[0] == 0:
    position = ret[1][18]
    position = [round(x, 6) for x in position]

    if len(position) == 6:
        xr = (PI - position[3]) 
        yr = (- position[4]) 
        tcp_pos=[0,0,0,xr,yr,0]
        robot.power_on() #上电
        robot.enable_robot()
        print("move1")
        #阻塞式运行，将末端运动至水平状态
        ret=robot.linear_move(tcp_pos,INCR,True,180)
        print(ret[0])
        # time.sleep(1)

robot.logout()
