from projection_warp import _generate_velocities_kernel, \
                            _find_corners_heights, \
                            _bilinear_interpolation, \
                            _compute_normal, \
                            _compute_tangent_vector, \
                            _compute_next_waypoint, \
                            _generate_trajectories_kernel

from critics_warp import _evaluate_trajectories_kernel, _compute_weighted_sum
import warp as wp
import numpy as np
import matplotlib.pyplot as plt
import time


def plot_surface_with_trajectory(X, Y, Z, trajectories, x_length=None, y_length=None, z_length=None):
    """
    Plot the 2.5D surface using a 3D surface plot with customizable axis lengths 
    and overlay selected trajectory points connected by lines.

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
    color = 'blue'
    for i, trajectory in enumerate(trajectories):
 
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=color, linewidth=2, label=f'Trajectory {i+1}')
        ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=color, s=5)  # Show one point out of 10
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis (Height)')
    
    # Set title
    ax.set_title('Surface with Selected Trajectories')
    
    # Set axis limits if provided
    if x_length:
        ax.set_xlim(-x_length / 2, x_length / 2)
    if y_length:
        ax.set_ylim(-y_length / 2, y_length / 2)
    if z_length:
        ax.set_zlim(0, z_length)
    
    # Add legend
    ax.legend()
    
    # Show the plot
    plt.show()

def split_trajectory(trajectories, number_of_trajectories, chunk_size):

    chunks = [trajectories[i*chunk_size:(i+1)*chunk_size] for i in range(number_of_trajectories)]  

    return chunks

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

# Set the parameters
grid_size = 600
half_width = 30.0
resolution = 2*half_width/grid_size
bumps = [
    ((-18.32, -8.94), 2.48, 3.62),    # Hill (height = 2.48, width = 3.62)
    ((-13.01, 6.74), 4.45, 5.85),     # Hill (height = 2.45, width = 5.85)
    ((-8.64, -14.23), 1.12, 4.39),    # Hill (height = 1.12, width = 4.39)
    ((-3.57, 12.05), 2.39, 1.92),     # Hill (height = 2.39, width = 5.02)
    ((0.97, -17.81), 1.62, 2.91),     # Hill (height = 1.62, width = 2.91)
    ((3.15, -1.56), 2.63, 2.21),      # Hill (height = 2.63, width = 5.21)
    ((6.13, 3.41), 2.14, 1.89),       # Hill (height = 2.14, width = 4.89)
    ((9.87, 16.38), 1.45, 3.74),      # Hill (height = 1.45, width = 3.74)
    ((14.94, 15.64), 2.89, 4.02),     # Hill (height = 2.89, width = 4.02)
    ((19.83, -9.56), 2.58, 1.72),     # Hill (height = 2.58, width = 6.72)
    ((-6.34, 5.56), 0.58, 4.55),      # Hill (height = 0.58, width = 4.55)
    ((-12.21, -13.32), 1.01, 3.89),   # Hill (height = 1.01, width = 3.89)
    ((-5.21, -5.32), 4.01, 3.89),     # Hill (height = 1.01, width = 3.89)
]
X, Y, Z = create_surface(grid_size, half_width, bumps=bumps)

x_robot = [15.0]
y_robot = [-15.0]
z_robot = [0]
hv_robot = np.array([-0.0, 0.5, 0.0])
x_goal = -15.0
y_goal = 20.0

dt = 0.05
iterations = 80 

initial_linear_velocity = 2.0
std_dev_linear = 0.1
min_linear_velocity = 0.5
max_linear_velocity = 2.5
initial_angular_velocity = 0.00
std_dev_angular = 1.5
min_angular_velocity = -0.8
max_angular_velocity = 0.8
x_min = -half_width
y_min = -half_width
radius = 0.01
obstacles_data = []
obstacles_wp = wp.array(obstacles_data, dtype=wp.vec3f)
horizon = iterations*dt*max_linear_velocity
number_of_trajectories = 50

# Initialise the warp framework
wp.init()
start_time = time.time()
loop = 0
goal = wp.vec2f(x_goal, y_goal)
optimal_lin_vel_wp = wp.array([initial_linear_velocity] * iterations, dtype=wp.float32, device="cuda")
optimal_ang_vel_wp = wp.array([initial_angular_velocity] * iterations, dtype=wp.float32, device="cuda")
Z_wp = wp.array(Z.flatten(), dtype=wp.float32, device="cuda")
goal_orientation = 2.2
linear_velocities = wp.zeros((number_of_trajectories*iterations), dtype=float)  
angular_velocities = wp.zeros((number_of_trajectories*iterations), dtype=float) 

while (abs(x_robot[-1]-x_goal) > 1.5 or abs(y_robot[-1]-y_goal) > 1.5) and y_robot[-1] < 30.0 and loop <3000:
    number_of_trajectories = 50

    position = wp.array([wp.vec2f(x_robot[-1], y_robot[-1])] * number_of_trajectories, dtype=wp.vec2f)
    q = wp.zeros(number_of_trajectories, dtype=wp.mat22f) 
    height = wp.zeros(number_of_trajectories, dtype=float)
    normal = wp.zeros(number_of_trajectories, dtype=wp.vec3f)
    heading_vectors = wp.zeros(iterations*number_of_trajectories, dtype=wp.vec3f)
    previous_heading_vector = wp.vec3f(hv_robot/np.linalg.norm(hv_robot))
    trajectories = wp.empty((number_of_trajectories*iterations), dtype=wp.vec3f) 
    costs_wp = wp.zeros(number_of_trajectories, dtype=float)

    # Generate random inputs sequences
    wp.launch(
        _generate_velocities_kernel,
        dim=number_of_trajectories*iterations,  
        inputs=[
            iterations, np.random.randint(iterations+1, 1000),  
            optimal_lin_vel_wp, optimal_ang_vel_wp, 
            std_dev_linear, min_linear_velocity, max_linear_velocity, 
            std_dev_angular, min_angular_velocity, max_angular_velocity,
            linear_velocities, angular_velocities
        ],
        device="cuda",
    )

    # Generate the trajectories based on the random inputs sequences
    wp.launch(
        _generate_trajectories_kernel,
        dim=number_of_trajectories,
        inputs=[
            position,
            -half_width,
            -half_width,
            grid_size,
            q,
            resolution,
            Z_wp,
            height,
            normal,
            heading_vectors,
            previous_heading_vector,
            iterations,
            linear_velocities,
            angular_velocities,
            dt,
            trajectories
        ],
        device="cuda"
    )

    # sampled_trajectories = trajectories.numpy()

    # Evaluate the cost of each trajectory
    wp.launch(
        kernel=_evaluate_trajectories_kernel,
        dim= number_of_trajectories,
        inputs=[x_robot[-1], y_robot[-1], goal, goal_orientation, trajectories, number_of_trajectories, iterations, obstacles_wp, radius, horizon, costs_wp],
    )
    

    costs = costs_wp.numpy()
    temperature = 1.0  
    normalized_scores = costs - np.min(costs)
    exponents = np.exp(-normalized_scores / temperature)
    weights = exponents / np.sum(exponents)
    weights_wp = wp.array(weights, dtype=float, device="cuda")
    optimal_lin_vel_wp = wp.zeros(iterations, dtype=float, device="cuda")
    optimal_ang_vel_wp = wp.zeros(iterations, dtype=float, device="cuda")


    wp.launch(
        kernel=_compute_weighted_sum,
        dim= number_of_trajectories,
        inputs=[
            weights_wp, 
            iterations,
            linear_velocities, 
            angular_velocities,
            optimal_lin_vel_wp,
            optimal_ang_vel_wp
        ],
        device="cuda"
    )

    # Reset the warp parameter to compute the optimal trajectory
    number_of_trajectories = 1
    position = wp.array([wp.vec2f(x_robot[-1], y_robot[-1])] * number_of_trajectories, dtype=wp.vec2f) 
    q = wp.zeros(number_of_trajectories, dtype=wp.mat22f) 
    height = wp.zeros(number_of_trajectories, dtype=float) 
    normal = wp.zeros(number_of_trajectories, dtype=wp.vec3f) 
    heading_vectors = wp.zeros(iterations*number_of_trajectories, dtype=wp.vec3f) 
    previous_heading_vector = wp.vec3f(hv_robot/np.linalg.norm(hv_robot))
    optimal_trajectory = wp.empty((number_of_trajectories*iterations), dtype=wp.vec3f) 

    # Compute the optimal trajectory
    wp.launch(
        _generate_trajectories_kernel,
        dim=number_of_trajectories,
        inputs=[
            position,
            -half_width,
            -half_width,
            grid_size,
            q,
            resolution,
            Z_wp,
            height,
            normal,
            heading_vectors,
            previous_heading_vector,
            iterations,
            optimal_lin_vel_wp,
            optimal_ang_vel_wp,
            dt,
            optimal_trajectory
        ],
        device="cuda"
    )


    # hv_robot = np.array([optimal_trajectory.numpy()[0][0] - x_robot[-1], optimal_trajectory.numpy()[0][1] - y_robot[-1], optimal_trajectory.numpy()[0][2] - z_robot[-1]])
    # if loop % 10 == 0:
    #     print(optimal_trajectory.numpy()[0][0])
    #     print(x_robot[-1])
    #     print("bad one:", hv_robot/np.linalg.norm(hv_robot))





    hv_robot = heading_vectors.numpy()[0]

    x_robot.append(optimal_trajectory.numpy()[0][0])
    y_robot.append(optimal_trajectory.numpy()[0][1])
    z_robot.append(optimal_trajectory.numpy()[0][2])

    
    loop += 1



print("Duration:", time.time()-start_time)
print("Number of loop:", loop)
print("Average time duration of one loop", (time.time()-start_time)/loop)


trajectory = np.array(list(zip(x_robot, y_robot, z_robot)))

plot_surface_with_trajectory(X, Y, Z, [trajectory], half_width*2, half_width*2, half_width) 





