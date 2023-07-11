import matplotlib.pyplot as plt
import numpy as np

# Define the rotation function
def rotate_point(rotationCenter, centers_original, F_Angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in degrees.
    """
    # Convert angle to radians
    theta = np.radians(F_Angle)  # positive sign for counterclockwise rotation
    
    # Shift the point so that the rotation center is at the origin
    x_shifted = centers_original[0] - rotationCenter[0]
    y_shifted = centers_original[1] - rotationCenter[1]

    # Apply the rotation about the origin
    x_rotated = x_shifted * np.cos(theta) - y_shifted * np.sin(theta)
    y_rotated = x_shifted * np.sin(theta) + y_shifted * np.cos(theta)

    # Shift the point back to its original location
    x_new = x_rotated + rotationCenter[0]
    y_new = y_rotated + rotationCenter[1]
    
    return [x_new, y_new]

# Define the points
rotationCenter = [0, 0]
step1_Position = [-540.915615, 103.134551]
StdARM = [57.44133073090249, -428.21138614268955]
StdARMActual = [53.49370202233297, -421.09205350134323]

# Define the rotation angles
rotation_angle = 2  # degrees
rotation_angle2 = -0.87  # degrees

# Calculate the new positions after the first rotation
rotationCenter_rotated = rotate_point(step1_Position, rotationCenter, rotation_angle)
StdARM_rotated = rotate_point(step1_Position, StdARM, rotation_angle)
StdARMActual_rotated = rotate_point(step1_Position, StdARMActual, rotation_angle)

# Calculate the new position of StdARMActual_rotated after the second rotation
StdARMActual_rotated2 = rotate_point(step1_Position, StdARMActual_rotated, rotation_angle2)

# Calculate the differences in x and y coordinates between StdARMActual_rotated2 and StdARM_rotated
x_diff = StdARMActual_rotated2[0] - StdARM_rotated[0]
y_diff = StdARMActual_rotated2[1] - StdARM_rotated[1]

# Create a scatter plot of the points
plt.figure(figsize=(8, 8))
plt.scatter(*rotationCenter_rotated, color='red', label='Rotation Center Rotated')
plt.scatter(*step1_Position, color='blue', label='Step1 Position')
plt.scatter(*StdARM_rotated, color='green', label='StdARM Rotated')
plt.scatter(*StdARMActual_rotated, color='purple', label='StdARM Actual Rotated')
plt.scatter(*StdARMActual_rotated2, color='orange', label='StdARM Actual Rotated2')

# Set the x and y axis limits to make the points more spread out
plt.xlim(-600, 100)
plt.ylim(-500, 200)

# Draw the x and y axes
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of points after second rotation')

# Add annotations for the coordinates of the points and the differences in x and y coordinates
plt.annotate(f'({StdARMActual_rotated2[0]:.2f}, {StdARMActual_rotated2[1]:.2f})', (StdARMActual_rotated2[0], StdARMActual_rotated2[1]), textcoords="offset points", xytext=(-10,10), ha='center')
plt.annotate(f'x_diff = {x_diff:.2f}', (0, -350), textcoords="offset points", xytext=(-10,10), ha='center')
plt.annotate(f'y_diff = {y_diff:.2f}', (0, -400), textcoords="offset points", xytext=(-10,10), ha='center')

plt.legend()

# Show the plot
plt.show()

# Print the differences
print(f"The difference in x coordinates between StdARMActual_rotated2 and StdARM_rotated is: {x_diff}")
print(f"The difference in y coordinates between StdARMActual_rotated2 and StdARM_rotated is: {y_diff}")
