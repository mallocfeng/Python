
def read_xy_coordinates_from_file(file_path):
    xy_coordinates = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                x, y = map(int, line.split(","))
                xy_coordinates.append((x, y))

    return xy_coordinates

# file_path = "Actual_FixtureTwoLocationPoints.txt"
# coordinates = read_xy_coordinates_from_file(file_path)

# if len(coordinates) >= 2:
#     Std_IMG_FixtureCirclePoint1_Actual, Std_IMG_FixtureCirclePoint2_Actual = coordinates[:2]
# else:
#     print("Error: File does not contain enough coordinates.")
