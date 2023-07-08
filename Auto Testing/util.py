
import os
import sys

# 获取项目配置路径
def get_dir_path():

    executable_path = sys.argv[0]
    executable_directory = os.path.dirname(executable_path)
    # print("可执行文件目录：", executable_directory)
    return executable_directory


get_dir_path()