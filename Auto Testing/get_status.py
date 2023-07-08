# -*- coding: utf-8 -*-
import sys
# sys.path.append('C:\Yandle\Develop\Auto Testing\lib64')
sys.path.append('C:\Yandle\Develop\Auto Testing\lib64-2.1.4')
import time      
import jkrc
PI=3.1415926

robot = jkrc.RC("10.5.5.100") #返回一个机器人对象
# robot.logout()
robot.login()  	  #登录


# robot.clear_error()
# robot.motion_abort()


# ret = robot.is_in_drag_mode()
# print(ret[1])

# robot.logout()

# time.sleep(5)
ret = robot.get_robot_status()
# robot.logout()
if ret[0] == 0:
    #print("robot_status : "+len(ret[1]))
    # print("    errcode : "+str(ret[1][0]))
    # print("    inpos : "+str(ret[1][1]))
    # print("    powered_on : "+str(ret[1][2]))
    # print("    enabled : "+str(ret[1][3]))
    # print("    rapidrate : "+str(ret[1][4]))
    # print("    15 : "+str(ret[1][15]))
    # print("    16 : "+str(ret[1][16]))
    # print("    17 : "+str(ret[1][17]))
    print("    18 : "+str(ret[1][18]))
    print("    xr : "+str(round(ret[1][18][3], 6)))
    print("    yr : "+str(round(ret[1][18][4], 6)))
    print("    19 : "+str(ret[1][19]))
    # print("    20 : "+str(ret[1][20]))
    # print('    is net connect: '+str(ret[1][len(ret[1])-1]))
    ret_end_pos = ret [1]
else:
    print("!!EORROR!! something wrong happend , errorcode is :",ret[0])
    
robot.logout()


