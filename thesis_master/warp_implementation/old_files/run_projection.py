from projection_warp import _find_corners_heights, \
                            _bilinear_interpolation, \
                            _compute_normal, \
                            _compute_tangent_vector, \
                            _compute_next_waypoint, \
                            _generate_trajectories_kernel

from sampling_warp import _generate_velocities_kernel
import warp as wp
import numpy as np
import matplotlib.pyplot as plt



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
    
    # Colors for different trajectories
    colors = plt.cm.get_cmap('tab10', len(trajectories))  # 'tab10' colormap for distinct colors
    
    for i, trajectory in enumerate(trajectories):
        if i % 3 == 0:  # Only show one trajectory out of every 5
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=colors(i), linewidth=2, label=f'Trajectory {i+1}')
            ax.scatter(trajectory[::10, 0], trajectory[::10, 1], trajectory[::10, 2], color=colors(i), s=10)  # Show one point out of 10
    
    # Set labels
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





########################################################
# Trajectories generations
########################################################


# Set the parameters

grid_size = 400
half_width = 20
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

dt = 0.05
number_of_trajectories = 100
iterations = 100 
initial_linear_velocity = 2.0
std_dev_linear = 0.1
min_linear_velocity = 0.0
max_linear_velocity = 2.5
initial_angular_velocity = 0.00
std_dev_angular = 1.0
min_angular_velocity = -0.8
max_angular_velocity = 0.8


# Initialise the warp framework

wp.init()
linear_velocities = wp.empty((number_of_trajectories*iterations), dtype=float)  
angular_velocities = wp.empty((number_of_trajectories*iterations), dtype=float) 
Z_wp = wp.array(Z.flatten(), dtype=wp.float32, device="cuda")
position = wp.array([wp.vec2f(6, 0)] * number_of_trajectories, dtype=wp.vec2f)

x_min = -20.0
y_min = -20.0
q = wp.zeros(number_of_trajectories, dtype=wp.mat22f) 
height = wp.zeros(number_of_trajectories, dtype=float)
normal = wp.zeros(number_of_trajectories, dtype=wp.vec3f)
heading_vectors = wp.zeros(number_of_trajectories, dtype=wp.vec3f)
previous_heading_vector = wp.vec3f(-1.0, 1.0, 0.0)
trajectories = wp.empty((number_of_trajectories*iterations), dtype=wp.vec3f) 
optimal_lin_vel_wp = wp.array([initial_linear_velocity] * iterations, dtype=wp.float32, device="cuda")
optimal_ang_vel_wp = wp.array([initial_angular_velocity] * iterations, dtype=wp.float32, device="cuda")

wp.launch(
            _generate_velocities_kernel,
            dim=number_of_trajectories*iterations,  
            inputs=[
                iterations, 
                np.random.randint(iterations+1, 1000),  
                optimal_lin_vel_wp, 
                optimal_ang_vel_wp, 
                std_dev_linear, 
                min_linear_velocity, 
                max_linear_velocity, 
                std_dev_angular, 
                min_angular_velocity, 
                max_angular_velocity,
                linear_velocities, 
                angular_velocities
            ],
            device="cuda"
        )


wp.launch(
    _generate_trajectories_kernel,
    dim=number_of_trajectories,
    inputs=[
        position,
        x_min,
        y_min,
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


trajectories = trajectories.numpy()
trajectories_chunks = split_trajectory(trajectories, number_of_trajectories, iterations)
plot_surface_with_trajectory(X, Y, Z, trajectories_chunks, half_width*2, half_width*2, half_width) 








########################################################
# TESTING
########################################################

# q_result = wp.zeros(2, dtype=wp.mat22f) 
# height_result = wp.zeros(2, dtype=float) 
# normal_result = wp.zeros(2, dtype=wp.vec3f)
# heading_vector = wp.zeros(2, dtype=wp.vec3f)

# np_hv = np.array([[-1.0, 4.0, 0.0], [-1.0, 4.0, 0.0]], dtype=np.float32)
# previous_heading_vector = wp.array([wp.vec3f(*v) for v in np_hv], dtype=wp.vec3f)

# waypoint_np = np.array([[-7.0, 4.0], [1.0, -18.0]], dtype=np.float32)
# waypoint = wp.array([wp.vec2f(*v) for v in waypoint_np], dtype=wp.vec2f)

# next_waypoint_position = wp.zeros(2, dtype=wp.vec2f)
# next_heading_vector = wp.zeros(2, dtype=wp.vec3f)

# np_lv = np.array([-1.0,-1.0], dtype=np.float32)
# linear_velocities = wp.array(np_lv, dtype=float)

# np_av = np.array([-0.2, -0.2], dtype=np.float32)
# angular_velocities = wp.array(np_av, dtype=float)


# wp.launch(
#     _find_corners_heights,
#     dim=2,
#     inputs=[
#         test_x, test_y, -half_width, -half_width, q_result, grid_size, resolution, Z_wp
#     ],
#     device="cuda",
# )


# wp.launch(
#     _bilinear_interpolation,
#     dim=2,
#     inputs=[test_x, test_y, q_result, resolution, height_result],
#     device="cuda",
# )


# wp.launch(
#     _compute_normal,
#     dim=2,
#     inputs=[q_result, resolution, normal_result],
#     device="cuda",
# )


# wp.launch(
#     _compute_tangent_vector,
#     dim=2,
#     inputs=[previous_heading_vector, normal_result, heading_vector],
#     device="cuda",
# )


# wp.launch(
#     _compute_next_waypoint,
#     dim=2,
#     inputs=[waypoint, heading_vector, linear_velocities, angular_velocities, normal_result, dt, next_waypoint_position, next_heading_vector],
#     device="cuda",
# )