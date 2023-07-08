import ast

image_coords_path = ''
def get_image_coords(path):
    vision_coords = []
    with open(path, 'r') as f:
        for line in f:
            # 去除行尾的换行符
            line = line.strip()
            # 分割字符串
            parts = line.split(': ')
            if len(parts) == 2:
                # 获取坐标部分
                coords_str = parts[1]
                # 分割坐标
                coords_parts = coords_str.split(',')
                if len(coords_parts) == 2:
                    # 转换坐标为浮点数并添加到列表
                    x = float(coords_parts[0])
                    y = float(coords_parts[1])
                    vision_coords.append((x, y))
    return vision_coords



def get_arm_coords(path):
    arm_coords = []
    with open(path, 'r') as f:
        for line in f:
            # 去除行尾的换行符
            line = line.strip()
            # 将字符串转换为列表
            coords = ast.literal_eval(line)
            # 取前三个数字并添加到列表
            arm_coords.append(coords[:3])
    return arm_coords

def transformCoordinate(image_coords_path, arm_coords_path, points):
    srcPoints = get_image_coords(image_coords_path)
    dstPoints = get_arm_coords(arm_coords_path)
    # 计算矩阵变换系数
    x = 0
    y = 0
    for i in range(8):
        a = (dstPoints[i+1][1] - dstPoints[i][1]) / (srcPoints[i+1][1] - srcPoints[i][1])
        b = dstPoints[i][1] - a * srcPoints[i][1]
        c = (dstPoints[i+1][0] - dstPoints[i][0]) / (srcPoints[i+1][0] - srcPoints[i][0])
        d = dstPoints[i][0] - c * srcPoints[i][0]

        # 对每个点进行坐标系转换
        x = points[0] * c + d + x
        y = points[1] * a + b + y
        #x = round(x)
        #y = round(y)
    return x/8, y/8


def transformCoordinateToImage(arm_coords_path,image_coords_path, points):
    dstPoints = get_image_coords(image_coords_path)
    srcPoints = get_arm_coords(arm_coords_path)
    # 计算矩阵变换系数
    x = 0
    y = 0
    for i in range(8):
        a = (dstPoints[i+1][1] - dstPoints[i][1]) / (srcPoints[i+1][1] - srcPoints[i][1])
        b = dstPoints[i][1] - a * srcPoints[i][1]
        c = (dstPoints[i+1][0] - dstPoints[i][0]) / (srcPoints[i+1][0] - srcPoints[i][0])
        d = dstPoints[i][0] - c * srcPoints[i][0]

        # 对每个点进行坐标系转换
        x = points[0] * c + d + x
        y = points[1] * a + b + y
        #x = round(x)
        #y = round(y)
    return x/8, y/8

def transformCoordinateOffset(image_coords_path, arm_coords_path, offset):
    srcPoints = get_image_coords(image_coords_path)
    dstPoints = get_arm_coords(arm_coords_path)
    std_x, std_y, _ = dstPoints[0]
    offset_x, offset_y = transformCoordinate(image_coords_path, arm_coords_path, (srcPoints[0][0] + offset[0], srcPoints[0][1] + offset[1]))
    #print(std_x - offset_x, std_y - offset_y)
    return std_x - offset_x, std_y - offset_y


def transformCoordinatePoint(image_coords_path, arm_coords_path, point):
    offset_x, offset_y = transformCoordinate(image_coords_path, arm_coords_path, (point[0],point[1]))
    #print(std_x - offset_x, std_y - offset_y)
    return offset_x, offset_y

image_coords_path = 'D:\\Image\\25mm\\image_coords.txt'
arm_coords_path = 'D:\\Image\\25mm\\arm_coords.txt'
# x, y = transformCoordinate(image_coords_path, arm_coords_path, (1000,1000))
# print(x, y)


#ArmPoint = transformCoordinateToImage(arm_coords_path,image_coords_path, ArmPoint)
#print(ArmPoint)
#-540.915615, 103.134551