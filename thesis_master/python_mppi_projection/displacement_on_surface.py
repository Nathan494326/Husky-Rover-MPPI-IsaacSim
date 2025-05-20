import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R
import random 
import time
np.random.seed(42)

start_time = time.time()

# def create_surface(grid_size, half_width, bumps):
#     """
#     Create a 2.5D surface with a Gaussian hole in the middle.

#     Args:
#         grid_size (int): Number of grid points along each axis.
#         half_width (float): Half the width of the surface grid.
#         bumps (list of tuples): Each tuple contains parameters for a bump.
#                                Each tuple should be of the form 
#                                (bump_center, bump_height, bump_width),
#                                where:
#                                    bump_center (tuple): The (x, y) center of the bump.
#                                    bump_height (float): The height of the bump.
#                                    bump_width (float): The width of the bump.

#     Returns:
#         X (np.ndarray): X coordinates of the surface.
#         Y (np.ndarray): Y coordinates of the surface.
#         Z (np.ndarray): Z values (heights) of the surface.
#     """

#     # Create base grid
#     x = np.linspace(-half_width, half_width, grid_size)
#     y = np.linspace(-half_width, half_width, grid_size)
#     X, Y = np.meshgrid(x, y)
#     Z = np.zeros_like(X)  # Start with a flat surface

#     # Add a Gaussian hole at the center
#     hole_center = (0, 0)  # Center of the hole
#     hole_height = 4    # Depth of the hole (negative for a dip)
#     hole_width = 6.0      # Width of the hole

#     Z += hole_height * np.exp(-((X - hole_center[0])**2 + (Y - hole_center[1])**2) / (2 * hole_width**2))

#     return X, Y, Z

def create_surface(grid_size, half_width, bumps):
    """
    Create a 2.5D surface with multiple bumps.

    Args:
        grid_size (int): Number of grid points along each axis.
        half_width (float): Half the width of the surface grid.
        bumps (list of tuples): Each tuple contains parameters for a bump.
                               Each tuple should be of the form 
                               (bump_center, bump_height, bump_width),
                               where:
                                   bump_center (tuple): The (x, y) center of the bump.
                                   bump_height (float): The height of the bump.
                                   bump_width (float): The width of the bump.

    Returns:
        X (np.ndarray): X coordinates of the surface.
        Y (np.ndarray): Y coordinates of the surface.
        Z (np.ndarray): Z values (heights) of the surface.
    """

    # Create base grid
    x = np.linspace(-half_width, half_width, grid_size)
    y = np.linspace(-half_width, half_width, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)  # Start with a flat surface
    
    # Apply each bump
    for bump_center, bump_height, bump_width in bumps:
        Z += bump_height * np.exp(-((X - bump_center[0])**2 + (Y - bump_center[1])**2) / (2 * bump_width**2))
    Z += 0.85*np.arctan(0.5 * X)
    return X, Y, Z

def plot_trajectories_2D(trajectories):
    """
    Plot multiple 2D trajectories on a single graph with equal scaling for x and y axes.

    Args:
        trajectories (list of np.ndarray): A list where each element is a 2D array 
                                           of shape (n, 3) representing a trajectory 
                                           with columns (x, y, z).
                                           Each trajectory will be plotted on the same graph.
    """
    plt.figure(figsize=(8, 6))
    color = 'red'
    # Iterate over each trajectory in the list
    for idx, traj in enumerate(trajectories):
        # Extract the x and y coordinates for each trajectory
        x = traj[:, 0]
        y = traj[:, 1]

        # Plot the trajectory
        plt.plot(x, y, color=color)
        color = 'blue'
        

    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Trajectories')

    # Ensure the aspect ratio of the plot is equal
    plt.axis('equal')


    # Add legend to differentiate between trajectories
    plt.legend()

    # Show the plot
    plt.show()

def plot_surface_with_trajectory(X, Y, Z, trajectories, x_length=None, y_length=None, z_length=None):
    """
    Plot the 2.5D surface using a 3D surface plot with customizable axis lengths 
    and overlay multiple trajectory points connected by lines.

    Args:
        X (np.ndarray): X coordinates of the grid.
        Y (np.ndarray): Y coordinates of the grid.
        Z (np.ndarray): Z values (height) at each grid point.
        trajectories (list of np.ndarray): List of arrays of shape (n, 3) containing multiple trajectory points (x, y, z).
        x_length (tuple, optional): Min and max for the X axis, e.g., (min, max).
        y_length (tuple, optional): Min and max for the Y axis, e.g., (min, max).
        z_length (tuple, optional): Min and max for the Z axis, e.g., (min, max).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)  # Alpha for transparency
    
    # Plot all trajectories
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # Define a set of colors for each trajectory
    
    for i, trajectory in enumerate(trajectories):
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=colors[i % len(colors)], linewidth=2, label=f'Trajectory {i+1}')
        ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=colors[i % len(colors)], s=10)  # Scatter points
    
    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis (Height)')
    
    # Set title
    ax.set_title('Surface with Multiple Trajectories')
    
    # Set axis limits if provided
    if x_length:
        ax.set_xlim(-x_length / 2, x_length / 2)
    if y_length:
        ax.set_ylim(-y_length / 2, y_length / 2)
    if z_length:
        ax.set_zlim(0, z_length*2)
    
    # Add legend
    ax.legend()
    
    # Show the plot
    plt.show()

def coordinate_to_index(x, y, x_min, y_min, step_size):
    x_index = int((x - x_min) / step_size)
    y_index = int((y - y_min) / step_size)
    return x_index, y_index

def find_corners_heights(x, y, q, resolution, X, Y, Z):
    """
    Find the four corners of the grid cell containing the point (x, y), and compute their corresponding heights.
    
    Args:
        x (float): X-coordinate of the point.
        y (float): Y-coordinate of the point.
        grid_size (float): The size of each grid cell.
        X (np.ndarray): 2D array representing the x-coordinates of the surface grid.
        Y (np.ndarray): 2D array representing the y-coordinates of the surface grid.
        Z (np.ndarray): 2D array representing the z-heights of the surface at each (X, Y) point.
    
    Returns:
        np.ndarray: 4x2 array where each row is a corner (x, y) of the grid cell and the corresponding z-height.
    """

    # x0 = np.floor(x / resolution) * resolution
    # y0 = np.floor(y / resolution) * resolution

    i = np.max(np.searchsorted(X[0], x)-2, 0)  # x index
    j = np.max(np.searchsorted(Y[:, 0], y)-2, 0)  # y index

    i = np.int((x + 20) / resolution)-2
    j = np.int((y + 20) / resolution)-2
    
    q[0,0] = Z[j , i]
    q[0,1] = Z[j , i + 1]
    q[1,0] = Z[j+1, i]
    q[1,1] = Z[j+1, i + 1]
    
    return q

def normal_on_grid(q, grid_size):
    """
    Compute the normal of a quad on a regular grid.

    Args:
        q (np.ndarray): 2x2 array containing the z-values of the grid cell.
        grid_size (float): The grid size.

    Returns:
        np.ndarray: Normal vector (3D).
    """
    vec = np.array([
        -grid_size / 2.0 * (q[0, 1] - q[0, 0] - q[1, 0] + q[1, 1]),
        -grid_size / 2.0 * (q[1, 0] - q[0, 0] - q[0, 1] + q[1, 1]),
        grid_size * grid_size
    ])
    return vec / np.linalg.norm(vec)

def get_heading_tangent_vector(normal, previous_heading_vector):
    """
    Project the previous heading vector onto the new plane defined by the given normal vector.

    Args:
        normal (np.ndarray): The normal vector of the new plane (surface).
        previous_heading_vector (np.ndarray): The previous heading vector of the robot.

    Returns:
        np.ndarray: The new heading tangent vector after projection onto the plane.
    """
    
    projection = previous_heading_vector - np.dot(previous_heading_vector, normal) * normal
    tangent_vector = projection / np.linalg.norm(projection)
    return tangent_vector

def bilinear_interpolator(x, y, q, resolution):
    """
    Perform bilinear interpolation for a point.

    Args:
        x (float): X-coordinate.
        y (float): Y-coordinate.
        q (np.ndarray): 2x2 array containing the z-values of the grid cell.

    Returns:
        float: Interpolated height value.
    """
    x_normalized = x / resolution
    y_normalized = y / resolution

    x2 = x_normalized - np.floor(x_normalized)
    y2 = y_normalized - np.floor(y_normalized)

    return (
        (1.0 - x2) * (1.0 - y2) * q[0, 0] +
        x2 * (1.0 - y2) * q[1, 0] +
        (1.0 - x2) * y2 * q[0, 1] +
        x2 * y2 * q[1, 1]
    )

def update_position(x, y, heading_vector, linear_velocity, angular_velocity, normal_vector, dt):
    """
    Updates the position of the robot based on its current state.

    Args:
        x (float): Current x-coordinate of the robot.
        y (float): Current y-coordinate of the robot.
        z (float): Current z-coordinate of the robot (height).
        heading_vector (np.ndarray): 3D vector representing the heading direction of the robot.
        linear_velocity (float): Speed at which the robot is moving in the direction of the heading vector.
        angular_velocity (float): Angular velocity (in radians per second) for rotating the robot.
        normal_vector (np.ndarray): Normal vector to the surface on which the robot is standing.
        dt (float): Time step for the update.

    Returns:
        np.ndarray: Updated position (x, y, z) of the robot.
        np.ndarray: Updated heading vector of the robot after considering angular velocity.
    """

    heading_vector = heading_vector / np.linalg.norm(heading_vector)
    displacement = heading_vector * linear_velocity * dt

    new_x = x + displacement[0]
    new_y = y + displacement[1]

    angle = angular_velocity * dt  # Rotation angle in radians

    rotation_quat = R.from_rotvec(angle * normal_vector)
    new_heading_vector = rotation_quat.apply(heading_vector)
    new_heading_vector = new_heading_vector / np.linalg.norm(new_heading_vector)

    return new_x, new_y, new_heading_vector




    # cos_theta = np.cos(angle)
    # sin_theta = np.sin(angle)

    # # Rodrigues' rotation formula
    # new_heading_vector = (
    #     heading_vector * cos_theta
    #     + np.cross(normal_vector, heading_vector) * sin_theta
    #     + normal_vector * np.dot(normal_vector, heading_vector) * (1.0 - cos_theta)
    # )

    # # Normalize the new heading vector
    # new_heading_vector = new_heading_vector / np.sqrt(np.dot(new_heading_vector, new_heading_vector))

    # return new_x, new_y, new_heading_vector



def generate_trajectory_25D(
    x0, y0, heading_vector, linear_velocity, angular_velocity, dt,
    iterations, resolution, X, Y, Z
):
    """
    Generate the trajectory of a robot moving on a surface.

    Args:
        x0 (float): Initial x-coordinate of the robot.
        y0 (float): Initial y-coordinate of the robot.
        heading_vector (np.ndarray): Initial heading vector of the robot (3D).
        linear_velocity (np.ndarray): Array of the linear velocities at each time step (m/s).
        angular_velocity (np.ndarray): Array of the angular velocities at each time step (rad/s).
        dt (float): Time step for the simulation.
        iterations (int): Number of iterations for the simulation.
        resolution (float): Grid resolution for the surface.
        X (np.ndarray): 2D array of x-coordinates of the surface grid.
        Y (np.ndarray): 2D array of y-coordinates of the surface grid.
        Z (np.ndarray): 2D array of height values of the surface.

    Returns:
        np.ndarray: Trajectory array of shape (iterations, 3), where each row is (x, y, z).
    """
    # Initialize variables
    q = np.zeros((2, 2))  # Corner heights
    normal = np.zeros(3)  # Surface normal vector
    trajectory = np.zeros((iterations, 3))  # To store trajectory points

    # Compute initial conditions
    q = find_corners_heights(x0, y0, q, resolution, X, Y, Z)
    height = bilinear_interpolator(x0, y0, q, resolution)
    normal = normal_on_grid(q, resolution)
    heading_vector = get_heading_tangent_vector(normal, heading_vector)
    x, y = x0, y0

    # Store initial position
    # trajectory[0, :] = np.array([x, y, height])

    # Iterate to compute trajectory
    for k in range(iterations):
        x, y, heading_vector = update_position(x, y, heading_vector, linear_velocity[k], angular_velocity[k], normal, dt)

        q = find_corners_heights(x, y, q, resolution, X, Y, Z)
        height = bilinear_interpolator(x, y, q, resolution)
        normal = normal_on_grid(q, resolution)
        heading_vector = get_heading_tangent_vector(normal, heading_vector)
        
        if x >= 20 or x <= -20 or y >= 20 or y <=-20:
            return None
        
        trajectory[k, :] = np.array([x, y, height])

    return trajectory

def generate_trajectory_2D(x0, y0, heading_vector, linear_velocity, angular_velocity, dt, iterations):
    """
    Generate the trajectory of a robot moving on a flat 2D plane.

    Args:
        x0 (float): Initial x-coordinate of the robot.
        y0 (float): Initial y-coordinate of the robot.
        heading_vector (np.ndarray): Initial heading vector of the robot (2D).
        linear_velocity (np.ndarray): Array of the linear velocities at each time step (m/s).
        angular_velocity (np.ndarray): Array of the angular velocities at each time step (rad/s).
        dt (float): Time step for the simulation.
        iterations (int): Number of iterations for the simulation.

    Returns:
        np.ndarray: Trajectory array of shape (iterations, 3), where each row is (x, y, z),
                    with z always equal to 0.
    """
    # Initialize trajectory
    trajectory = np.zeros((iterations, 3))
    x, y = x0, y0

    # Normalize the heading vector
    heading_vector = heading_vector / np.linalg.norm(heading_vector)

    # Store initial position
    trajectory[0, :] = np.array([x, y, 0])

    # Compute trajectory on a flat 2D plane
    for k in range(iterations - 1):
        # Update position using linear velocity
        displacement = heading_vector * linear_velocity[k] * dt
        x += displacement[0]
        y += displacement[1]

        # Update heading vector using angular velocity
        angle = angular_velocity[k] * dt
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        heading_vector[:2] = np.dot(rotation_matrix, heading_vector[:2])

        # Store the updated position with z = 0
        trajectory[k + 1, :] = np.array([x, y, 0])
        if x >= 20 or x <= -20 or y >= 20 or y <=-20:
            return None

    return trajectory

def generate_linear_velocities(iterations, initial_velocity=1.0, std_dev=0.1, min_velocity=0.5, max_velocity=2.0):
    """
    Generate an array of linear velocity commands for the robot, where each value is drawn
    from a Gaussian distribution centered on the previous velocity.

    Args:
        iterations (int): Number of iterations (length of the output array).
        initial_velocity (float): Initial linear velocity.
        std_dev (float): Standard deviation for the Gaussian distribution.
        min_velocity (float): Minimum linear velocity allowed.
        max_velocity (float): Maximum linear velocity allowed.

    Returns:
        np.ndarray: Array of shape (iterations - 1,) containing velocity commands.
    """
    velocities = np.zeros(iterations)
    velocities[0] = initial_velocity

    for i in range(1, iterations):
        new_velocity = np.random.normal(velocities[i - 1], std_dev)
        velocities[i] = np.clip(new_velocity, min_velocity, max_velocity)

    return velocities

def generate_angular_velocities(iterations, initial_velocity=0.0, std_dev=0.05, min_velocity=-0.5, max_velocity=0.5):
    """
    Generate an array of angular velocity commands for the robot, where each value is drawn
    from a Gaussian distribution centered on the previous velocity.

    Args:
        iterations (int): Number of iterations (length of the output array).
        initial_velocity (float): Initial angular velocity.
        std_dev (float): Standard deviation for the Gaussian distribution.
        min_velocity (float): Minimum angular velocity allowed.
        max_velocity (float): Maximum angular velocity allowed.

    Returns:
        np.ndarray: Array of shape (iterations - 1,) containing velocity commands.
    """
    velocities = np.zeros(iterations)
    velocities[0] = initial_velocity

    for i in range(1, iterations):
        new_velocity = np.random.normal(velocities[i - 1], std_dev)
        velocities[i] = np.clip(new_velocity, min_velocity, max_velocity)

    return velocities


grid_size = 400
half_width = 20
resolution = 2*half_width/grid_size
bumps = [
    ((-18.32, -8.94), 2.48, 3.62),    # Hill (height = 2.48, width = 3.62)
    ((-13.01, 6.74), 4.45, 5.85),     # Hill (height = 2.45, width = 5.85)
    ((-8.64, -14.23), 1.12, 4.39),    # Hill (height = 1.12, width = 4.39)
    ((-3.57, 12.05), 2.39, 1.92),     # Hill (height = 2.39, width = 5.02)
    ((0.97, -17.81), 1.62, 2.91),     # Hill (height = 1.62, width = 2.91)
    ((3.15, -1.56), 3.23, 2.21),      # Hill (height = 2.63, width = 5.21)
    # ((6.13, 3.41), 2.14, 1.89),       # Hill (height = 2.14, width = 4.89)
    ((9.87, 16.38), 1.45, 3.74),      # Hill (height = 1.45, width = 3.74)
    ((14.94, 15.64), 2.89, 4.02),     # Hill (height = 2.89, width = 4.02)
    ((19.83, -9.56), 2.58, 1.72),     # Hill (height = 2.58, width = 6.72)
    ((-6.34, 5.56), 0.58, 4.55),      # Hill (height = 0.58, width = 4.55)
    ((-12.21, -13.32), 1.01, 3.89),   # Hill (height = 1.01, width = 3.89)
    ((-5.21, -5.32), 4.01, 3.89),     # Hill (height = 1.01, width = 3.89)
]

X, Y, Z = create_surface(grid_size, half_width, bumps=bumps)

iterations = 200



x0, y0 = 6, -6
heading_vector = np.array([-0.5, 4.0, 0])
linear_velocities = generate_linear_velocities(iterations, initial_velocity=1.5, std_dev=0.0, min_velocity=1.5, max_velocity=2.5)
angular_velocities = generate_angular_velocities(iterations, initial_velocity=0.0, std_dev=0.0, min_velocity=-0.4, max_velocity=0.4)
dt = 0.05

initialisation_time = time.time() - start_time
start_time = time.time()

# Generate 2.5D trajectory
trajectory_25D = generate_trajectory_25D(
    x0, y0, heading_vector, linear_velocities, angular_velocities, dt,
    iterations, resolution, X, Y, Z
)

traj25D_time = time.time()- start_time
start_time = time.time()

# Generate 2D trajectory
trajectory_2D = generate_trajectory_2D(
    x0, y0, heading_vector, linear_velocities, angular_velocities, dt, 
    iterations
)

traj2D_time = time.time()- start_time
start_time = time.time()

plot_surface_with_trajectory(X, Y, Z, [trajectory_25D, trajectory_2D], half_width*2, half_width*2, half_width) # trajectory_25D, trajectory_2D

plot_trajectories_2D([trajectory_25D, trajectory_2D])

plot_time = time.time()- start_time

print("initialisation_time:", initialisation_time)
print("traj25D_time:", traj25D_time)
print("traj2D_time:", traj2D_time)
print("plot_time:", plot_time)


# trajectories_2D = []
# trajectories_25D = []

# for i in range(20):
#     start_time = time.time()  

#     if i % 10 == 0:
#         print(i, 'th iteration')
#     # Randomize the start position (x0, y0) within a range
#     x0 = random.uniform(-20, 20)  # Example random x0 within a range
#     y0 = random.uniform(-20, 20)  # Example random y0 within a range

#     if x0 < 0 and y0 < 0:
#         heading_angle = random.uniform(0, np.pi/2)
#     elif x0 > 0 and y0 < 0:
#         heading_angle = random.uniform(np.pi/2, np.pi)
#     elif x0 < 0 and y0 > 0:
#         heading_angle = random.uniform(-np.pi/2, 0)
#     else:
#         heading_angle = random.uniform(np.pi, 3*np.pi/2)

#     heading_vector = np.array([np.cos(heading_angle), np.sin(heading_angle), 0])  # Heading in x, y (2D)

#     linear_velocities = generate_linear_velocities(
#         iterations, initial_velocity=random.uniform(1.5, 2.5), std_dev=0.1, min_velocity=1.5, max_velocity=2.5
#     )

#     angular_velocities = generate_angular_velocities(
#         iterations, initial_velocity=random.uniform(-0.4, 0.4), std_dev=0.1, min_velocity=-0.4, max_velocity=0.4
#     )

#     dt = 0.01

#     trajectory_25D = generate_trajectory_25D(
#         x0, y0, heading_vector, linear_velocities, angular_velocities, dt,
#         iterations, resolution, X, Y, Z
#     )
#     if trajectory_25D is None:
#         print("break")
#         continue

#     trajectory_2D = generate_trajectory_2D(
#         x0, y0, heading_vector, linear_velocities, angular_velocities, dt, 
#         iterations
#     )
#     if trajectory_2D is None:
#         print("break")
#         continue

#     # plot_surface_with_trajectory(X, Y, Z, [trajectory_25D, trajectory_2D], half_width*2, half_width*2, half_width) # trajectory_25D, trajectory_2D

#     # plot_trajectories_2D([trajectory_25D, trajectory_2D])

#     trajectories_25D.append(trajectory_25D)
#     trajectories_2D.append(trajectory_2D)

#     end_time = time.time()
#     loop_duration = end_time - start_time
#     print(f"Iteration {i} took {loop_duration:.4f} seconds.")

# iteration_points = [499, 999, 1499]
# differences = {499: [], 999: [], 1499: []}

# for i in range(len(trajectories_25D)):  # Loop through both sets of trajectories
#     traj_25D = trajectories_25D[i]
#     traj_2D = trajectories_2D[i]
    
#     for point in iteration_points:
#         point_25D = traj_25D[point]
#         point_2D = traj_2D[point]
        
#         diff = np.linalg.norm(np.array([point_25D[0], point_25D[1]]) - np.array([point_2D[0], point_2D[1]]))
#         differences[point].append(diff)

# mean_differences = {}
# std_differences = {}

# for point in iteration_points:
#     mean_differences[point] = np.mean(differences[point])
#     std_differences[point] = np.std(differences[point])

# print("Mean Differences:")
# for point in iteration_points:
#     print(f"At {point} iterations: Mean Difference = {mean_differences[point]}, Std Dev = {std_differences[point]}")



#######################################################################################################################
#######################################################################################################################
# FOR THE STATISTICS
#######################################################################################################################
#######################################################################################################################

# trajectories_2D = []
# trajectories_25D = []

# for i in range(100):
#     if i % 10 == 0:
#         print(i, 'th iteration')
#     # Randomize the start position (x0, y0) within a range
#     x0 = random.uniform(-20, 20)  # Example random x0 within a range
#     y0 = random.uniform(-20, 20)  # Example random y0 within a range

#     if x0 < 0 and y0 < 0:
#         heading_angle = random.uniform(0, np.pi/2)
#     elif x0 > 0 and y0 < 0:
#         heading_angle = random.uniform(np.pi/2, np.pi)
#     elif x0 < 0 and y0 > 0:
#         heading_angle = random.uniform(-np.pi/2, 0)
#     else:
#         heading_angle = random.uniform(np.pi, 3*np.pi/2)

#     heading_vector = np.array([np.cos(heading_angle), np.sin(heading_angle), 0])  # Heading in x, y (2D)

#     linear_velocities = generate_linear_velocities(
#         iterations, initial_velocity=random.uniform(1.5, 2.5), std_dev=0.1, min_velocity=1.5, max_velocity=2.5
#     )

#     angular_velocities = generate_angular_velocities(
#         iterations, initial_velocity=random.uniform(-0.4, 0.4), std_dev=0.1, min_velocity=-0.4, max_velocity=0.4
#     )

#     dt = 0.01

#     trajectory_25D = generate_trajectory_25D(
#         x0, y0, heading_vector, linear_velocities, angular_velocities, dt,
#         iterations, resolution, X, Y, Z
#     )
#     if trajectory_25D is None:
#         print("break")
#         continue

#     trajectory_2D = generate_trajectory_2D(
#         x0, y0, heading_vector, linear_velocities, angular_velocities, dt, 
#         iterations
#     )
#     if trajectory_2D is None:
#         print("break")
#         continue

#     # plot_surface_with_trajectory(X, Y, Z, [trajectory_25D, trajectory_2D], half_width*2, half_width*2, half_width) # trajectory_25D, trajectory_2D

#     # plot_trajectories_2D([trajectory_25D, trajectory_2D])

#     trajectories_25D.append(trajectory_25D)
#     trajectories_2D.append(trajectory_2D)


# iteration_points = [499, 999, 1499]
# differences = {499: [], 999: [], 1499: []}

# for i in range(len(trajectories_25D)):  # Loop through both sets of trajectories
#     traj_25D = trajectories_25D[i]
#     traj_2D = trajectories_2D[i]
    
#     for point in iteration_points:
#         point_25D = traj_25D[point]
#         point_2D = traj_2D[point]
        
#         diff = np.linalg.norm(np.array([point_25D[0], point_25D[1]]) - np.array([point_2D[0], point_2D[1]]))
#         differences[point].append(diff)

# mean_differences = {}
# std_differences = {}

# for point in iteration_points:
#     mean_differences[point] = np.mean(differences[point])
#     std_differences[point] = np.std(differences[point])

# print("Mean Differences:")
# for point in iteration_points:
#     print(f"At {point} iterations: Mean Difference = {mean_differences[point]}, Std Dev = {std_differences[point]}")



#######################################################################################################################
#######################################################################################################################
# BASIC RUN
#######################################################################################################################
#######################################################################################################################

# x0, y0 = -14, -4
# heading_vector = np.array([1.0, 0.0, 0])
# linear_velocities = generate_linear_velocities(iterations, initial_velocity=1.5, std_dev=0.1, min_velocity=1.5, max_velocity=2.5)
# angular_velocities = generate_angular_velocities(iterations, initial_velocity=0.0, std_dev=0.1, min_velocity=-0.4, max_velocity=0.4)
# dt = 0.01


# # Generate 2.5D trajectory
# trajectory_25D = generate_trajectory_25D(
#     x0, y0, heading_vector, linear_velocities, angular_velocities, dt,
#     iterations, resolution, X, Y, Z
# )

# # Generate 2D trajectory
# trajectory_2D = generate_trajectory_2D(
#     x0, y0, heading_vector, linear_velocities, angular_velocities, dt, 
#     iterations
# )


# plot_surface_with_trajectory(X, Y, Z, [trajectory_25D, trajectory_2D], half_width*2, half_width*2, half_width) # trajectory_25D, trajectory_2D

# plot_trajectories_2D([trajectory_25D, trajectory_2D])




#######################################################################################################################
#######################################################################################################################
# COMPARISON WITH REAL TRAJ
#######################################################################################################################
#######################################################################################################################

# import csv


# # Function to save a trajectory to a CSV file
# def save_trajectory_to_csv(filename, trajectory):
#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         # Write header
#         writer.writerow(['X', 'Y', 'Z'])
#         # Write data
#         writer.writerows(trajectory)

# # Save the trajectories
# save_trajectory_to_csv('trajectory_25D.csv', trajectory_25D)
# save_trajectory_to_csv('trajectory_2D.csv', trajectory_2D)



#######################################################################################################################
#######################################################################################################################
# DEBUGGING TOOL FOR NORMAL PLANS
#######################################################################################################################
#######################################################################################################################

# q = np.zeros((2, 2))  # Corner heights
# normal = np.zeros(3)  # Surface normal vector

# # Compute initial conditions
# q = find_corners_heights(1, -18, q, resolution, X, Y, Z)
# print(q)
# normal = normal_on_grid(q, resolution)

# x_index, y_index = coordinate_to_index(x0, y0, -20, -20, 0.1)

# print(normal_map[400 - y_index][x_index])
# print(normal)
