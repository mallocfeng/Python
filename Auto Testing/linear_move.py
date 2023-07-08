# -*- coding: utf-8 -*-
import sys
sys.path.append('C:\Yandle\Develop\Auto Testing\lib64')
import time      
import jkrc
PI=3.1415926
#运动模式
ABS = 0
INCR= 1
#  单位：mm
tcp_pos=[0,0,0,0,0,0]
robot = jkrc.RC("10.5.5.100") #返回一个机器人对象
robot.login()#登录
robot.power_on() #上电
robot.enable_robot()
print("move1")
#阻塞 沿z轴负方向 以20mm/s 运动60mm
ret=robot.linear_move(tcp_pos,INCR,True,180)
print(ret[0])
time.sleep(3)
robot.logout()
