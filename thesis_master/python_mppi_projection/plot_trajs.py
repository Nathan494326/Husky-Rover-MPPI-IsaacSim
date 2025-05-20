import pandas as pd
import matplotlib.pyplot as plt

# Load the robot's trajectory from the robot_position.csv
csv_file = 'python_mppi_projection/robot_position.csv'
try:
    robot_data = pd.read_csv(csv_file)
    print(f"Loaded data from {csv_file}")
except FileNotFoundError:
    print(f"File {csv_file} not found. Make sure the file exists.")
    exit()

# Extract X and Y columns for the robot's trajectory
x_robot = robot_data['X']
y_robot = robot_data['Y']

# Load the 2D trajectory from trajectory_2D.csv
trajectory_2d_file = 'python_mppi_projection/trajectory_2D.csv'
try:
    trajectory_2d_data = pd.read_csv(trajectory_2d_file)
    print(f"Loaded 2D trajectory from {trajectory_2d_file}")
except FileNotFoundError:
    print(f"File {trajectory_2d_file} not found. Make sure the file exists.")
    exit()

# Extract X and Y columns for the 2D trajectory
x_2D = trajectory_2d_data['X']
y_2D = trajectory_2d_data['Y']

# Load the 2.5D trajectory from trajectory_25D.csv
trajectory_25d_file = 'python_mppi_projection/trajectory_25D.csv'
try:
    trajectory_25d_data = pd.read_csv(trajectory_25d_file)
    print(f"Loaded 2.5D trajectory from {trajectory_25d_file}")
except FileNotFoundError:
    print(f"File {trajectory_25d_file} not found. Make sure the file exists.")
    exit()

# Extract X and Y columns for the 2.5D trajectory
x_25D = trajectory_25d_data['X']
y_25D = trajectory_25d_data['Y']

# Plot the robot's trajectory, 2D, and 2.5D trajectories on the same graph
plt.figure(figsize=(8, 6))

# Plot the robot's trajectory (real data from robot_position.csv)
plt.plot(x_robot, y_robot, color='blue', marker='o', linestyle='-', markersize=5)

# Plot the 2D trajectory
plt.plot(x_2D, y_2D, color='green', marker='x', linestyle='--')

# # Plot the 2.5D trajectory
plt.plot(x_25D, y_25D, color='red', marker='^', linestyle=':')
plt.xlim(-5, 14)  # Set x-axis range
plt.ylim(-13.5, 5.5) # Set y-axis range
# Adding labels and title
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Robot Trajectory Comparison in XY Plane')

# Display a legend
plt.legend()

# Display grid
plt.grid()

# Show the plot
plt.show()
