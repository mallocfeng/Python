import argparse
import configparser 
import os

from get_robot_status import get_robot_position

from util import get_dir_path

PI=3.1415926
TOLERANCE = 0.1

parser = argparse.ArgumentParser()
# model 型号
parser.add_argument('-model', '--model', help='Model No.')
# step 步骤序号
parser.add_argument('-step', '--step', help='Step No.')
# read 读取对应步骤的坐标
parser.add_argument('-read', '--read', action='store_true', help='Read Config')
# write 保存当前步骤对应的机械臂坐标
parser.add_argument('-write', '--write', action='store_true', help='Write Config')

args = parser.parse_args()
read_flag = False
write_flag = False
step_no = ''
model_no = ''
config_dir = 'config'
config_file = ''
current_directory = get_dir_path()
socket_config_path = os.path.join(current_directory, config_dir, 'config.ini')
socket_config_name = 'Current_Model'


if args.read:
    # print(f'read: {args.read}')
    read_flag = args.read

if args.write:
    # print(f'read: {args.write}')
    write_flag = args.write

if args.step:
    # print(f'step: {args.step}')
    step_no = args.step

if args.model:
    # print(f'step: {args.step}')
    model_no = args.model

def set_current_model():
    config = configparser.ConfigParser()
    config.read(socket_config_path)  # 读取现有的配置文件

    if not config.has_section(socket_config_name):
        config.add_section(socket_config_name)
    config[socket_config_name] = {'model': model_no}
    
    with open(socket_config_path, 'w') as configfile:
        config.write(configfile)

def get_location():
    # 检查当前目录下是否存在配置文件
    if os.path.exists(config_file):
        # 创建配置解析器对象
        config = configparser.ConfigParser()
        config.read(config_file)

        # 检查步骤号对应的节是否存在于配置中
        section_name = f'{step_no}'
        if config.has_section(section_name):
            location = config.get(section_name, 'location')
            # print(location)
            return location
        else:
            print("Step does not exist")
            return ''
    else:
        print("Config file does not exist")
        return ''

def save_params():
    ensure_config_dir()
    location = get_robot_position()
    if len(location) != 6 :
        print("error")
        return

    location = list(location)  # 将元组转换为列表
    location[3] = location[3] * 180 / PI
    location[4] = location[4] * 180 / PI
    location[5] = location[5] * 180 / PI
    location = tuple(location)
    
    final_location = [round(x, 3) for x in location]
    
    # 检测是否水平
    # if check_position(final_location) == False:
        # print('Please check the horizontal state of the robotic arm.')
    # 创建配置解析器对象
    config = configparser.ConfigParser()

    # 检查当前目录下是否存在配置文件
    if os.path.exists(config_file):
        # 如果文件存在，则读取其中的数据
        config.read(config_file)

    # 将步骤号和位置信息添加到配置对象中
    section_name = f'{step_no}'
    config[section_name] = {'location': ','.join(str(coord) for coord in final_location)}

    # 将配置对象写入配置文件
    with open(config_file, 'w') as file:
        config.write(file)
        print(f"Configuration saved to{{aaaa}} {config_file}.")
    

def read_params():
    location = get_location()
    # print(f"Location for Step {step_no}: {location}")
    print(f"{location}")


def ensure_config_dir():
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

def check_position(position):
    x_error = round(abs(position[3] - 180.00), 4)
    y_error = round(abs(position[4]), 4)
    
    if x_error <= TOLERANCE and y_error <= TOLERANCE:
        return True
    else:
        return False
    
# step_no = 1
# model_no = 'CalcConfig'
# write_flag = True
if model_no != '' and step_no == '':
    
    print('set model')
    set_current_model()
# elif step_no != '' and model_no != '':
elif step_no != '' and model_no == '':
    if read_flag == False and write_flag == False:
        read_flag = True
        
    # config_file = os.path.join(os.path.join(current_directory, config_dir), f'{model_no}.ini')
    print(socket_config_path)
    config = configparser.ConfigParser()
    if config.read(socket_config_path):
        if config.has_section(socket_config_name):
            model_no = config.get(socket_config_name, 'model')
    if model_no == '':
        print('pls set current model first!')

    config_file = os.path.join(model_no, 'CalcConfig.ini')
    print(config_file)
    if read_flag:
        read_params()
    if write_flag:
        save_params()
else:
    print("Please input the correct params")

# 先初始化项目路径
# py .\location.py -model D:\Image\25mm

# 读取配置文件中对应step的location
# py .\location.py -step {step名字} -read

# 将当前机械臂坐标信息，写入对应step中
# py .\location.py -step {step名字} -write

