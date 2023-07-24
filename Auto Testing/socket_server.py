import socket
import argparse
import configparser 
import os
import signal
import sys
import select
import subprocess


import sys
sys.path.append(r'D:\Python')

# from CirclePos_V2 import getParaList

config_dir = 'config'


from util import get_dir_path

current_directory = get_dir_path()
socket_config_path = os.path.join(current_directory, config_dir, 'config.ini')
socket_config_name = 'Socket_Server'
current_model_name = 'Current_Model'

# 定义服务器地址和端口号
def read_socket_config():
    config = configparser.ConfigParser()
    if config.read(socket_config_path):
        if config.has_section(socket_config_name):
            host = config.get(socket_config_name, 'HOST')
            port = config.getint(socket_config_name, 'PORT')
            return host, port
    return None, None

def generate_socket_config(host='127.0.0.1', port=8888):
    config = configparser.ConfigParser()
    config.read(socket_config_path)  # 读取现有的配置文件
    if not config.has_section(socket_config_name):
        config.add_section(socket_config_name)
    # config[socket_config_name] = {'current_model': model_no}
    config.set(socket_config_name, 'HOST', host)  # 设置新的参数值
    config.set(socket_config_name, 'PORT', str(port))  # 设置新的参数值
    # config[socket_config_name] = {'HOST': host, 'PORT': str(port)}
    with open(socket_config_path, 'w') as configfile:
        config.write(configfile)

# HOST = '10.5.5.115'
# PORT = 8888
HOST, PORT = read_socket_config()

if HOST is None or PORT is None:
    generate_socket_config()
    HOST, PORT = read_socket_config()

# HOST = '127.0.0.1'

def get_location(branch): 
    model_no = ''
    config = configparser.ConfigParser()
    if config.read(socket_config_path):
        if config.has_section(current_model_name):
            model_no = config.get(current_model_name, 'model')
    if model_no == '':
        print('pls set current model first!')
        return
    # config_file = os.path.join(current_directory, config_dir, f'{model_no}.ini')
    config_file = os.path.join(model_no, 'config.ini')
    # 检查当前目录下是否存在配置文件
    if os.path.exists(config_file):
        # 创建配置解析器对象
        config = configparser.ConfigParser()
        config.read(config_file)

        # 获取步骤数量
        step_count = len(config.sections())

        # 根据分支选择读取的步骤范围
        if branch == 1:
            start_step = 1
            end_step = min(4, step_count)
        elif branch == 2:
            start_step = 5
            end_step = step_count
        else:
            print("Invalid branch")
            return ''

        # 拼接对应步骤的数值
        location_list = []
        for step_no in range(start_step, end_step + 1):
            section_name = f'{step_no}'
            if config.has_section(section_name):
                location = config.get(section_name, 'location')
                location_list.append(location)
            else:
                print(f"{step_no} does not exist")

        # 将数值拼接成字符串
        location_string = ','.join(location_list)
        location_string = f"{branch},{location_string}"
        print(location_string)
        return location_string
    else:
        print("Config file does not exist")
        return ''
    
def get_model_dir():
    model_no = ''
    config = configparser.ConfigParser()
    if config.read(socket_config_path):
        if config.has_section(current_model_name):
            model_no = config.get(current_model_name, 'model')
    if model_no == '':
        print('pls set current model first!')
    return model_no



def get_step_location(step_no): 
    model_no = get_model_dir()
    # config_file = os.path.join(current_directory, config_dir, f'{model_no}.ini')
    config_file = os.path.join(model_no, 'config.ini')
    # 检查当前目录下是否存在配置文件
    if os.path.exists(config_file):
        # 创建配置解析器对象
        config = configparser.ConfigParser()
        config.read(config_file)

        # 拼接对应步骤的数值
        section_name = f'{step_no}'
        if config.has_section(section_name):
            return  config.get(section_name, 'location')
        return ''
    else:
        print("Config file does not exist")
        return ''

def get_step_offset():
    model_path = get_model_dir()
    actual_image_path = os.path.join(model_path, 'ActualPic.jpg')
    # 构建调用A.py的命令
    command = ['py', 'CirclePos_V2.py', '3', actual_image_path, model_path + '\\']
    # 执行命令并捕获输出结果
    output = subprocess.check_output(command).decode()
    print( output)
    return float(output.split(',')[0].strip()),float(output.split(',')[1].strip()),float(output.split(',')[2].strip())



# 创建 TCP 服务器套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
def handle_interrupt(signal, frame):
    # 中断信号处理代码
    print("Exiting...")
    server_socket.close()
    sys.exit(0)

try:
    # 绑定服务器地址和端口号
    server_socket.bind((HOST, PORT))
    # 监听来自客户端的连接
    server_socket.listen()

    # 注册中断信号处理函数
    signal.signal(signal.SIGINT, handle_interrupt)

    print("Server started, waiting for client connection...")
    while True:
        # 使用 select 检查是否有客户端连接，设置超时时间为1秒
        ready_to_read, _, _ = select.select([server_socket], [], [], 1)
        
        if server_socket in ready_to_read:
            # 有客户端连接
            client_socket, client_address = server_socket.accept()
            # # 接受客户端连接请求
            # client_socket, client_address = server_socket.accept()
            print(f"Client connected: {client_address}")

            while True:

                # 接收客户端发送的数据
                data = client_socket.recv(1024).decode()
                print(f"Received from client: {data}")
                if data == "get <Recv_Data_Str_1>":
                    # 发送消息给客户端
                    location_string = get_location(1)
                    client_socket.send(f'<Recv_Data_Str_1><"{location_string}">'.encode())
                    print("Sent 'location 1-4' to client")

                if data == "get <Recv_Data_Str_2>":
                    # 发送消息给客户端
                    location_string = get_location(2)
                    client_socket.send(f'<Recv_Data_Str_2><"{location_string}">'.encode())
                    print("Sent 'location 4-7' to client")

                # 判断是否是获取具体步骤的偏差坐标
                if "get <Recv_Data_Step" in data:
                    start_index = data.find("<Recv_Data_Step") + len("<Recv_Data_Step")
                    end_index = data.find(">", start_index)
                    
                    #offset_x, offset_y, offset_rz = 1,2,3
                    if start_index != -1 and end_index != -1:
                        step_no = data[start_index:end_index]
                        offset_x, offset_y, offset_rz = get_step_offset()
                        # offset_x, offset_y, offset_rz = getParaList(step_no)
                        # offset_x, offset_y, offset_rz = 1,2,3
                        location_string = get_step_location(step_no)
                        # 拆解字符串并转换为浮点数列表
                        values = location_string.split(',')
                        values = [float(value) for value in values]

                        # 添加偏移量
                        values[0] += offset_x
                        values[1] += offset_y
                        values[5] += offset_rz

                        # 将更新后的值转换为字符串
                        updated_location_string = ','.join(str(value) for value in values)
                        client_socket.send(f'<Recv_Data_{step_no}><"{updated_location_string}">'.encode())
                
                # 如果客户端断开连接，则退出内部循环
                if not data:
                    break
                
                # 关闭与客户端的连接
                # client_socket.close()
        
except KeyboardInterrupt:
    # 中断信号处理代码
    # 捕获 KeyboardInterrupt 异常
    handle_interrupt(signal.SIGINT, None)
