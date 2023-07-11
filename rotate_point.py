import numpy as np

def rotate_point(rotationCenter, centers_original, F_Angle):
    """
    Rotate a point clockwise by a given angle around a given origin.
    The angle should be given in degrees.
    """
    # Convert angle to radians
    theta = -np.radians(F_Angle)  # negative sign for clockwise rotation
    
    # Shift the point so that the rotation center is at the origin
    x_shifted = centers_original[0] - rotationCenter[0]
    y_shifted = centers_original[1] - rotationCenter[1]

    # Apply the rotation about the origin
    x_rotated = x_shifted * np.cos(theta) - y_shifted * np.sin(theta)
    y_rotated = x_shifted * np.sin(theta) + y_shifted * np.cos(theta)

    # Shift the point back to its original location
    x_new = x_rotated + rotationCenter[0]
    y_new = y_rotated + rotationCenter[1]
    
    return x_new, y_new


# Test with some data
rotationCenter = [0, 0]
centers_original = [-18087.365521756696, 17251.578869262]
#centers_original = [-1, 1]
F_Angle = 1  # degrees

x_new, y_new = rotate_point(rotationCenter, centers_original, F_Angle)

print(f"New coordinates after rotation are: ({x_new}, {y_new})")
