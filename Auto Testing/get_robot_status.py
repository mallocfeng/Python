# -*- coding: utf-8 -*-
import sys
import os
# sys.path.append('C:\Yandle\Develop\Auto Testing\lib64')

from util import get_dir_path

current_directory = get_dir_path()
# lib_path_jeka = os.path.join(current_directory, 'lib64-2.1.4')
lib_path_jeka = current_directory
sys.path.append(lib_path_jeka)



print(lib_path_jeka)
# print(os.path.join(current_directory,'config', 'config.ini'))

import jkrc
import configparser

PI=3.1415926

# 定义服务器地址和端口号
def get_jeka_ip():
    jeka_ip_config_path = os.path.join(current_directory,'config', 'config.ini')
    jeka_ip_config_name = 'Jeka_IP_Address'
    config = configparser.ConfigParser()
    if config.read(jeka_ip_config_path):
        if config.has_section(jeka_ip_config_name):
            ip = config.get(jeka_ip_config_name, 'ip')
            return ip
    return '10.5.5.100'

def get_robot_position():

    position = [-478.56059399657624,-499.81895400798004,699.6924181972526,3.14,0.16405997606335193,3.99260355205516]
    rounded_position = [round(x, 6) for x in position]
    return rounded_position
    robot = jkrc.RC(get_jeka_ip()) #返回一个机器人对象
    robot.login()  	  #登录
    # time.sleep(1)
    ret = robot.get_robot_status()
    robot.logout()
    if ret[0] == 0:
        position = ret[1][18]
        # position = list(position)  # 将元组转换为列表
        # position[3] = position[3] * 180 / PI
        # position[4] = position[4] * 180 / PI
        # position[5] = position[5] * 180 / PI
        # position = tuple(position)
        rounded_position = [round(x, 6) for x in position]
        return rounded_position
        #print("robot_status : "+len(ret[1]))
        print("    errcode : "+str(ret[1][0]))
        print("    inpos : "+str(ret[1][1]))
        print("    powered_on : "+str(ret[1][2]))
        print("    enabled : "+str(ret[1][3]))
        print("    rapidrate : "+str(ret[1][4]))
        print("    position : "+str(ret[1][19]))
        print('    is net connect: '+str(ret[1][len(ret[1])-1]))
        ret_end_pos = ret [1]
    else:
        return []
        print("!!EORROR!! something wrong happend , errorcode is :",ret[0])


