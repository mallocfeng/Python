import numpy as np

def read_space_coords_from_file(file_path):
    coords_str = []
    with open(file_path) as f:
        for line in f:
            if not line.startswith("#"):
                coords_str.append(line.strip().strip("[]"))
    return np.array([list(map(float, s.split(", ")[:3])) for s in coords_str])

def read_pixel_coords_from_file(file_path):
    coords_str = []
    with open(file_path) as f:
        for line in f:
            if not line.startswith("#"):
                coords_str.append(line.strip())
    return np.array([list(map(int, s.split(": ")[1].split(","))) for s in coords_str])

def getlArmPositionOffset(robot_coords_path, pixel_coords_path, ImgRotationCenterPos,ActualArmPos):
    CenterArmPosition = map_pixel_to_space(robot_coords_path,pixel_coords_path,ImgRotationCenterPos)
    #print(CenterArmPosition)
    #ActualArmPosition = [-510.040754, 160.754379]
    OffsetArmCenterValue = [ ActualArmPos[0] - CenterArmPosition[0], ActualArmPos[1] - CenterArmPosition[1]]
    return OffsetArmCenterValue


def map_space_to_pixel(robot_coords_path, pixel_coords_path, space_coord,offset_coord = [0,0]):
    # 读取数据
    robot_coords_str = []
    pixel_coords_str = []
    

    # 将字符串转化为 numpy 数组
    pixel_coords = read_pixel_coords_from_file(pixel_coords_path)
    space_coords = read_space_coords_from_file(robot_coords_path)

    # 建立一个 18x6 的矩阵 A_inv，每一行的形式是 [X, Y, 1, 0, 0, 0] 或者 [0, 0, 0, X, Y, 1]
    A_inv = np.zeros((18, 6))
    A_inv[:9, :3] = np.c_[space_coords[:, :2], np.ones((9, 1))]
    A_inv[9:, 3:] = np.c_[space_coords[:, :2], np.ones((9, 1))]

    # 建立一个 18x1 的向量 b_inv，前9个元素是 x 坐标，后9个元素是 y 坐标
    b_inv = np.concatenate([pixel_coords[:, 0], pixel_coords[:, 1]])

    # 使用最小二乘法求解映射参数
    params_inv, _, _, _ = np.linalg.lstsq(A_inv, b_inv, rcond=None)
    space_coord = [space_coord[0] - offset_coord[0],space_coord[1] - offset_coord[1]]
    # 计算像素坐标
    X, Y = space_coord
    return [round(params_inv[0]*X + params_inv[1]*Y + params_inv[2]), round(params_inv[3]*X + params_inv[4]*Y + params_inv[5])]

def map_pixel_to_space(robot_coords_path, pixel_coords_path, pixel_coord,offset_coord = [0,0]):
    # 读取数据
    robot_coords_str = []
    pixel_coords_str = []

    # 将字符串转化为 numpy 数组
    pixel_coords = read_pixel_coords_from_file(pixel_coords_path)
    space_coords = read_space_coords_from_file(robot_coords_path)

    # 建立一个 18x6 的矩阵 A，每一行的形式是 [x, y, 1, 0, 0, 0] 或者 [0, 0, 0, x, y, 1]
    A = np.zeros((18, 6))
    A[:9, :3] = np.c_[pixel_coords, np.ones(9)]
    A[9:, 3:] = np.c_[pixel_coords, np.ones(9)]

    # 建立一个 18x1 的向量 b，前9个元素是 x 坐标，后9个元素是 y 坐标
    b = np.concatenate([space_coords[:, 0], space_coords[:, 1]])

    # 使用最小二乘法求解映射参数
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # 计算空间坐标
    x, y = pixel_coord
    return [params[0]*x + params[1]*y + params[2] + offset_coord[0], params[3]*x + params[4]*y + params[5] + offset_coord[1]]


#print(map_space_to_pixel(r"D:\Image\25mm\arm_coords.txt", r"D:\Image\25mm\image_coords.txt", [-565.40555, 88.560426]))
#print(map_pixel_to_space(r"D:\Image\25mm\arm_coords.txt", r"D:\Image\25mm\image_coords.txt", [1038,995]))