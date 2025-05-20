from thesis_master.warp_implementation.projection_warp import _find_corners_heights, \
                            _bilinear_interpolation, \
                            _compute_normal, \
                            _compute_tangent_vector, \
                            _generate_trajectories_kernel, \
                            _generate_trajectories_2D_kernel

from thesis_master.warp_implementation.sampling_warp import _generate_velocities_kernel, _generate_inputs_kernel, _convert_inputs_to_velocities

from thesis_master.warp_implementation.critics_warp import _evaluate_trajectories_kernel, _compute_weights, _compute_sum, _compute_weighted_sum
import sys
import cv2
# import importlib.util
# import os
# module_path = '/isaac-sim/kit/python/lib/python3.10/site-packages/warp'
# print(f"Does the file exist? {os.path.exists(module_path)}")
# # Load the module explicitly using importlib
# spec = importlib.util.spec_from_file_location("warp", module_path)
# wp = importlib.util.module_from_spec(spec)
# sys.modules["warp"] = wp
# spec.loader.exec_module(wp)

# if 'warp' in sys.modules:
#     del sys.modules['warp']

# # Now insert the path where you want to load warp from
# sys.path = [path for path in sys.path if "extscache" not in path]
# sys.path.insert(0, '/isaac-sim/kit/python/lib/python3.10/site-packages/warp')

# print(sys.path)
import warp as wp

# print(wp.__file__)
import numpy as np
import matplotlib.pyplot as plt
import time
import yaml
import os

def plot_surface_with_trajectory(X, Y, Z, costmap, trajectories, x_length=None, y_length=None, z_length=None, frame_idx=0, save_path="frames"):
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

    os.makedirs(save_path, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)  # Alpha for transparency

    above_threshold = costmap > 0.5
    ax.scatter(X[above_threshold], Y[above_threshold], Z[above_threshold], color='red', s=1)

    color = 'blue'

    for i, trajectory in enumerate(trajectories):
        if i < len(trajectories) - 1:  
            color = 'red'  # All except the last one are blue
            s = 3
            alpha = 0.4  # Make blue points more transparent
        else:
            color = 'blue'   # The last trajectory is red
            s = 20
            alpha = 1.0  # Keep red fully visible

        if i % 50 == 0 or i == (len(trajectories) - 1):
            ax.scatter(trajectory[::5, 0], trajectory[::5, 1], trajectory[::5, 2], 
                    color=color, s=s, alpha=alpha)  # Adjusted transparency

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

    # if x_length:
    #     ax.set_xlim(-15, 15)
    # if y_length:
    #     ax.set_ylim(-15, 15)
    # if z_length:
    #     ax.set_zlim(0, 30)
    

    # ax.view_init(elev=90, azim=-90)  # Top-down view
    # ax.dist = 5

    # # plt.savefig(f"{save_path}/frame_{frame_idx:04d}.png", dpi=300)
    # # plt.close(fig)
    plt.show()

def plot_2d_surface_with_trajectory(X, Y, Z, costmap, trajectories, goal, show_live, frame, frame_folder='frame_folder'):
    """
    Plot the 2D surface using a heatmap and overlay selected trajectory points.

    Args:
        X (np.ndarray): X coordinates of the grid.
        Y (np.ndarray): Y coordinates of the grid.
        Z (np.ndarray): Z values (height) at each grid point.
        costmap (np.ndarray): Costmap used for obstacle visualization.
        trajectories (list of np.ndarray): List of arrays of shape (n, 3) containing multiple trajectory points (x, y, z).
        save_path (str, optional): Directory to save frames (if needed).
    """

    if Z.size == 0:
        raise ValueError("The Z array is empty, cannot plot.")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Check if X, Y are also non-empty and properly shaped
    if X.size == 0 or Y.size == 0:
        raise ValueError("The X or Y array is empty, cannot plot.")
    

    # Plot the heightmap using 'jet' colormap
    img = ax.imshow(Z, cmap="jet", origin="upper", extent=[X.min(), X.max(), Y.min(), Y.max()]) #, vmin=6370, vmax=6430)
    plt.colorbar(img, label="Height")

    # # Overlay costmap points (thresholded)
    # below_threshold = costmap < 0.01
    # ax.scatter(X[below_threshold], Y[below_threshold], color='red', s=1, label="Rocks")


    # x_lin = np.linspace(X.min(), X.max(), costmap.shape[0])
    # y_lin = np.linspace(Y.min(), Y.max(), costmap.shape[1])
    # Xc, Yc = np.meshgrid(x_lin, y_lin)
    # below_threshold = costmap < 0.01
    # ax.scatter(Xc[below_threshold], Yc[below_threshold], color='red', s=1, label="Rocks")



    # Plot trajectory points
    for i, trajectory in enumerate(trajectories):
        if i < len(trajectories) - 1:
            color = 'red'  # Past trajectories
            s = 3
            alpha = 0.1 
        else:
            color = 'black'  # Last trajectory
            s = 20
            alpha = 1.0  

        if i % 1 == 0 or i == (len(trajectories) - 1):
            ax.scatter(trajectory[:, 0], trajectory[:, 1], color=color, s=s, alpha=alpha)

    ax.scatter(goal[0], goal[1], color='green', s=50, alpha=1.0)
    # Labels and title
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_title("2D Surface with Selected Trajectories")

    plt.legend()

    if show_live:
        plt.show()
    else:
        if not os.path.exists(frame_folder):
            os.makedirs(frame_folder)

        # Save the plot as a PNG file inside the frame_folder
        filename = os.path.join(frame_folder, f'frame_{frame}.png')

        plt.savefig(filename)

        # Close the plot to free up memory
        plt.close()


def plot_costmap_with_frames(costmap, map_extent, show_live=False, frame=0, frame_folder='costmap_frame_folder'):
    """
    Plots the 2D costmap and optionally saves each frame to disk.

    Args:
        costmap (np.ndarray): 2D array representing the costmap.
        map_extent (tuple): (xmin, xmax, ymin, ymax) defining the real-world extent of the map.
        show_live (bool): Whether to show the plot interactively.
        frame (int): Frame number to save as (used in filename).
        frame_folder (str): Directory where frames are saved if show_live is False.
    """
    xmin, xmax, ymin, ymax = map_extent
    extent = [xmin, xmax, ymin, ymax]

    fig, ax = plt.subplots(figsize=(6, 6))

    img = ax.imshow(costmap, origin='upper', cmap='gray_r', extent=extent)
    plt.colorbar(img, label="Cost value")

    ax.set_title("Costmap")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    plt.tight_layout()

    if show_live:
        plt.show()
    else:
        os.makedirs(frame_folder, exist_ok=True)
        filename = os.path.join(frame_folder, f'frame_{frame:04d}.png')
        plt.savefig(filename)
        plt.close()





def split_trajectory(trajectories, number_of_trajectories, chunk_size):

    chunks = [trajectories[i*chunk_size:(i+1)*chunk_size] for i in range(number_of_trajectories)]  

    return chunks

def compute_path_metrics(trajectory):
    total_length = 0.0
    total_angle_up = 0.0
    total_angle_down = 0.0
    total_distance_up = 0.0
    k = 20

    for i in range(1, len(trajectory) - k, k):
        # Compute the Euclidean distance between consecutive points
        segment_vector = trajectory[i + k] - trajectory[i - 1]
        segment_length = np.linalg.norm(segment_vector)
        total_length += segment_length

        # Compute the slope angle (in degrees)
        if segment_length > 0:  # Avoid division by zero
            angle = np.degrees(np.arctan2(segment_vector[2], np.linalg.norm(segment_vector[:2])))

            if angle > 0:
                total_angle_up += angle
            else:
                total_angle_down += abs(angle)
        
        if segment_vector[2] > 0:
            total_distance_up += segment_vector[2]

    return total_length, total_angle_up, total_angle_down, total_distance_up


class Surface:
    def __init__(self, which_map, filename, which_costmap, costmap_file, grid_size, half_width, origin, bumps, radius_robot, obstacles=[]):

        self.grid_size = grid_size
        self.r_robot = radius_robot
        self.half_width = half_width
        self.resolution = 2 * self.half_width / self.grid_size

        x = np.linspace(-self.half_width, self.half_width, grid_size)
        y = np.linspace(-self.half_width, self.half_width, grid_size)
        self.X, self.Y = np.meshgrid(x, y)

        self.costmap_size = int(self.grid_size/8)
        self.costmap_resolution = 2 * self.half_width / self.costmap_size

        x_costmap = np.linspace(-self.half_width, self.half_width, self.costmap_size)
        y_costmap = np.linspace(-self.half_width, self.half_width, self.costmap_size)
        self.X_costmap, self.Y_costmap = np.meshgrid(x_costmap, y_costmap)
        self.Z = np.zeros_like(self.X)

        if which_map == "manual":
            self.r_robot = radius_robot
            self.X, self.Y, self.Z = self.create_surface(bumps)
            
        if which_map == "imported":
            start_index = 1000
            end_index = 2500
            self.X, self.Y, self.Z = self.import_surface(filename, start_index, end_index, bumps)

        if which_costmap == "manual":
            self.obstacles = obstacles
            print("Creating the costmap...")
            self.costmap = self.create_obstacles_costmap(obstacles, origin)
            print("Costmap created")

        if which_map == "imported":
            self.costmap = self.import_obstacles_costmap(costmap_file)
        
        # np.save("map_with_craters_without_rocks.npy", self.costmap)

    def import_surface(self, filename, start_index, end_index, bumps):
        x = np.linspace(-self.half_width, self.half_width, end_index-start_index)
        y = np.linspace(-self.half_width, self.half_width, end_index-start_index)
        X, Y = np.meshgrid(x, y)
        Z = np.load(filename) 

        return X, Y, Z

    def create_surface(self, bumps):
        x = np.linspace(-self.half_width, self.half_width, self.grid_size)
        y = np.linspace(-self.half_width, self.half_width, self.grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        # Z += 10.5 * np.exp(-((X + 55)**2 + (Y - 55)**2) / (2 * 4**2))
        # Z += 20 *np.exp(-((X-20)**2 + (Y-20)**2) / (2 * 20.0**2))  # Inverted Gaussian


        ### Craters
        for bump_center, bump_height, bump_width in bumps:
            Z += (bump_height - 0.5) *np.exp(-((X - bump_center[0])**2 + (Y - bump_center[1])**2) / (2 * (bump_width)**2)) 
            Z -= (bump_height + 0.5) *np.exp(-((X - bump_center[0])**2 + (Y - bump_center[1])**2) / (2 * (bump_width/2)**2))
            
        # Z *= 2
        # Z += 0.85 * np.arctan(0.5 * X)

        # Z += 3.5 *np.exp(-(X**2 + Y**2) / (2 * 3.0**2))  # Inverted Gaussian
        # Z -= 4.5* np.exp(-(X**2 + Y**2) / (2 * 1.5**2))  # Raised rim


        ## ONE BUMP
        # for bump_center, bump_height, bump_width in bumps:
        #     sigma_x = bump_width
        #     sigma_y = bump_width  # Example: making elongation in y twice that in x

        #     Z += bump_height * np.exp(-((X - bump_center[0])**2 / (2 * sigma_x**2) + (Y - bump_center[1])**2 / (2 * sigma_y**2)))

        # ### BUMPS
        # for bump_center, bump_height, bump_width in bumps:
        #     Z += bump_height * np.exp(-((X - bump_center[0])**2 + (Y - bump_center[1])**2) / (2 * bump_width**2))
        # Z += 0.85 * np.arctan(0.5 * X)
            
        # ### INCLINED SURFACE
        # Z = X.copy()
        # Z = Z / 3
        # for bump_center, bump_height, bump_width in bumps:
        #     sigma_x = bump_width
        #     sigma_y = 2 * bump_width  # Example: making elongation in y twice that in x

        #     Z += bump_height * np.exp(-((X - bump_center[0])**2 / (2 * sigma_x**2) + (Y - bump_center[1])**2 / (2 * sigma_y**2)))

        # Z += 1.0 * np.sin(0.5 * X + 2.5) * np.cos(0.5 * Y + 1.5)
        # Z -= 0.5 * np.sin(0.25 * X + 2.5) * np.cos(0.25 * Y + 1.5)
        # # Z += 1.2 * np.sin(5 * X) * np.sin(5 * Y)  # Higher frequency oscillations
        # # Z += 0.8 * np.sin(np.sin(0.5 * X) * np.cos(0.5 * Y))  # Nested sine functions
        # # Z += 0.3 * np.random.randn(*X.shape)  # Adding some random noise

        return X, Y, Z
    
    def import_obstacles_costmap(self, costmap_file):
        return np.load(costmap_file) 
    
    def create_obstacles_costmap(self, obstacles, origin):
        obs_costmap = 255 * np.ones((self.costmap_size, self.costmap_size), dtype=np.uint8)        
        x0, y0 = origin

        for x_global, y_global, r_obs in obstacles:
            x_local = y_global - y0
            y_local = (x_global - x0)

            total_radius = r_obs/2 + self.r_robot + 0.1
            mask = (self.X_costmap - x_local)**2 + (self.Y_costmap - y_local)**2 <= total_radius**2
            
            obs_costmap[mask] = 0.0

        distance_map = cv2.distanceTransform(obs_costmap.astype(np.uint8), cv2.DIST_L2, 5)
        distance_map = cv2.normalize(distance_map, None, 0, 1.0, cv2.NORM_MINMAX)
        costmap = (1 - distance_map)**20
        print("Costmap updated")
        return costmap


class Robot:
    def __init__(self, x, y, heading_vector, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self.x = [x]
        self.y = [y]
        self.z = [0]
        self.lin_vel = []
        self.ang_vel = []
        self.heading_vector = np.array(heading_vector) / np.linalg.norm(heading_vector)
        self.radius = config['frame_work']['robot_radius']
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0


    def update_position(self, new_x, new_y, new_z, new_heading):
        self.x.append(new_x)
        self.y.append(new_y)
        self.z.append(new_z)
        self.heading_vector = new_heading

class MPPI_Controller:

    def __init__(self, surface, robot, config_path, goal_x, goal_y, goal_orientation):
        # Load configuration from file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.rng = np.random.default_rng(seed=42)
        self.robot = robot
        self.surface = surface
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_orientation = goal_orientation
        self.loop = 0
        # self.total_cost = 0

        # General parameters of MPPI
        self.number_of_iterations = config['controller']['number_of_iterations']
        self.dt = config['controller']['dt']
        self.number_of_trajectories = config['controller']['number_of_trajectories']
        
        # Parameters for the random trajectories generation

        self.initial_linear_velocity = config['velocities']['initial_linear_velocity'] # 1.0
        self.std_dev_u1 = config['inputs']['std_dev_u1'] # 0.3
        self.min_u1 = config['inputs']['min_u1'] #-1.0
        self.max_u1 = config['inputs']['max_u1'] # 1.0
        self.std_dev_u2 = config['inputs']['std_dev_u2'] # 0.3
        self.min_u2 = config['inputs']['min_u2'] # -1.0
        self.max_u2 = config['inputs']['max_u2'] # 1.0
        self.initial_angular_velocity = config['velocities']['initial_angular_velocity'] # 0.4
        self.v_min_linear = config['velocities']['min_linear_velocity'] # 0.0
        self.v_max_linear = config['velocities']['max_linear_velocity'] # 2.0
        self.v_min_angular = config['velocities']['min_angular_velocity'] # -0.8
        self.v_max_angular = config['velocities']['max_angular_velocity'] # 0.8

        # Parameters for the cost evaluation
        self.temperature = config['cost_evaluation']['temperature']
        self.horizon = self.dt*self.v_max_linear*self.number_of_iterations

    def warp_setup(self):
        wp.init()        

        # Variables for the random inputs sequences generation
        self.optimal_u1_wp = wp.zeros(self.number_of_iterations, dtype=float, device="cuda")
        self.optimal_u2_wp = wp.zeros(self.number_of_iterations, dtype=float, device="cuda")
        self.u1 = wp.zeros((self.number_of_trajectories*self.number_of_iterations), dtype=float, device="cuda")  
        self.u2 =wp.zeros((self.number_of_trajectories*self.number_of_iterations), dtype=float, device="cuda")  

        # Variables regarding the kinematics state
        self.position = wp.array([wp.vec2f(self.robot.x[-1], self.robot.y[-1])] * self.number_of_trajectories, dtype=wp.vec2f, device="cuda")
        self.optimal_lin_vel_wp = wp.array([self.initial_linear_velocity] * self.number_of_iterations, dtype=wp.float32, device="cuda")
        self.optimal_ang_vel_wp = wp.array([self.initial_angular_velocity] * self.number_of_iterations, dtype=wp.float32, device="cuda")
        self.linear_velocities = wp.zeros((self.number_of_trajectories*self.number_of_iterations), dtype=float, device="cuda")  
        self.angular_velocities = wp.zeros((self.number_of_trajectories*self.number_of_iterations), dtype=float, device="cuda") 
        
        # Variables for the trajectories generation
        self.q = wp.zeros(self.number_of_trajectories, dtype=wp.mat22f, device="cuda") 
        self.Z_wp = wp.array(self.surface.Z.flatten(), dtype=wp.float32, device="cuda") 
        self.costmap_wp = wp.array(self.surface.costmap.flatten(), dtype=wp.float32, device="cuda") 
        self.height = wp.zeros(self.number_of_trajectories, dtype=float, device="cuda")
        self.normal = wp.zeros(self.number_of_trajectories, dtype=wp.vec3f, device="cuda")
        self.heading_vectors = wp.zeros(self.number_of_trajectories*self.number_of_iterations, dtype=wp.vec3f, device="cuda")
        self.previous_heading_vector = wp.vec3f(self.robot.heading_vector)
        self.trajectories = wp.zeros((self.number_of_trajectories*self.number_of_iterations), dtype=wp.vec3f, device="cuda") 
        
        # Variables for the cost evaluation
        self.goal = wp.vec2f(self.goal_x, self.goal_y)
        self.costs_wp = wp.zeros(self.number_of_trajectories, dtype=float, device="cuda")
        self.weights_wp = wp.zeros(self.number_of_trajectories, dtype=float, device="cuda")
        self.min_cost = wp.array([np.inf], dtype=float)
        self.weights_sum = wp.array([0.0], dtype=float)

        # Variables for the simulation of the robot
        self.position_sim = wp.array([wp.vec2f(self.robot.x[-1], self.robot.y[-1])], dtype=wp.vec2f, device="cuda")
        self.q_sim = wp.zeros(1, dtype=wp.mat22f, device="cuda") 
        self.height_sim = wp.zeros(1, dtype=float, device="cuda")
        self.normal_sim = wp.zeros(1, dtype=wp.vec3f, device="cuda")
        self.heading_vectors_sim = wp.zeros(self.number_of_iterations, dtype=wp.vec3f, device="cuda")
        self.trajectories_sim = wp.zeros((self.number_of_iterations), dtype=wp.vec3f, device="cuda") 

        # Part regarding new critic to evaluate the height of the wheels
        self.left_wheel_pos = wp.zeros((self.number_of_trajectories*self.number_of_iterations), dtype=wp.vec3f, device="cuda")  
        self.right_wheel_pos = wp.zeros((self.number_of_trajectories*self.number_of_iterations), dtype=wp.vec3f, device="cuda")  
        self.left_wheel_pos_sim = wp.zeros((self.number_of_iterations), dtype=wp.vec3f, device="cuda") 
        self.right_wheel_pos_sim = wp.zeros((self.number_of_iterations), dtype=wp.vec3f, device="cuda") 

    def reset(self, controller_or_sim):
        # Reset the warp variables before starting a new loop if necessary
        if controller_or_sim == "controller":
            self.position = wp.array([wp.vec2f(self.robot.x[-1], self.robot.y[-1])] * self.number_of_trajectories, dtype=wp.vec2f)
            self.previous_heading_vector = wp.vec3f(self.robot.heading_vector/np.linalg.norm(self.robot.heading_vector))
            self.costs_wp.zero_()
            self.weights_wp.zero_()
            self.min_cost = wp.array([np.inf], dtype=float)
            self.weights_sum = wp.array([0.0], dtype=float)

        else:
            self.position_sim = wp.array([wp.vec2f(self.robot.x[-1], self.robot.y[-1])], dtype=wp.vec2f)
            self.previous_heading_vector = wp.vec3f(self.robot.heading_vector/np.linalg.norm(self.robot.heading_vector))
            self.optimal_u1_wp.zero_()
            self.optimal_u2_wp.zero_()

    def MPPI_step(self, proj):
        # Run one MPPI step which consists in:
        # - sampling random inputs sequences
        # - computing the corresponding trajectories in the 2.5D space
        # - evaluating those trajectories
        # - computing the estimated optimal trajectory

        wp.launch(
            _generate_inputs_kernel,
            dim=self.number_of_trajectories*self.number_of_iterations,  
            inputs=[
                self.number_of_iterations, 
                self.rng.integers(self.number_of_iterations+1, 1000),  
                self.optimal_u1_wp, 
                self.optimal_u2_wp, 
                self.std_dev_u1, 
                self.min_u1, 
                self.max_u1, 
                self.std_dev_u2, 
                self.min_u2, 
                self.max_u2,
                self.u1, 
                self.u2
            ],
            device="cuda"
        )

        wp.launch(
            _convert_inputs_to_velocities,
            dim=self.number_of_trajectories,  
            inputs=[
                self.number_of_iterations, 
                self.robot.radius,  
                self.robot.left_wheel_speed, 
                self.robot.right_wheel_speed, 
                self.v_min_linear,
                self.v_max_linear,
                self.v_min_angular,
                self.v_max_angular,
                self.u1, 
                self.u2,
                self.linear_velocities,
                self.angular_velocities,
                3.5,
                0.96
            ],
            device="cuda"
        )

        if proj == "2d":
            wp.launch(
                _generate_trajectories_2D_kernel,
                dim=self.number_of_trajectories,
                inputs=[
                    self.position,
                    -self.surface.half_width,
                    -self.surface.half_width,
                    self.surface.grid_size,
                    self.q,
                    self.surface.resolution,
                    self.Z_wp,
                    self.height,
                    self.heading_vectors,
                    self.previous_heading_vector,
                    self.number_of_iterations,
                    self.linear_velocities,
                    self.angular_velocities,
                    self.dt,
                    self.trajectories
                ],
                device="cuda"
            )

        elif proj == "3d":
            wp.launch(
                _generate_trajectories_kernel,
                dim=self.number_of_trajectories,
                inputs=[
                    self.position,
                    -self.surface.half_width,
                    -self.surface.half_width,
                    self.surface.grid_size,
                    self.q,
                    self.surface.resolution,
                    self.Z_wp,
                    self.height,
                    self.normal,
                    self.heading_vectors,
                    self.previous_heading_vector,
                    self.number_of_iterations,
                    self.linear_velocities,
                    self.angular_velocities,
                    self.dt,
                    self.trajectories, 
                    self.left_wheel_pos,
                    self.right_wheel_pos
                ],
                device="cuda"
            )
        


        wp.launch(
            kernel=_evaluate_trajectories_kernel,
            dim= self.number_of_trajectories,
            inputs=[
                self.robot.x[-1], 
                self.robot.y[-1], 
                self.goal, 
                self.goal_orientation, 
                self.trajectories, 
                self.left_wheel_pos,
                self.right_wheel_pos,
                self.linear_velocities,
                self.v_max_linear,
                self.number_of_trajectories, 
                self.number_of_iterations, 
                self.surface.half_width,
                self.surface.costmap_resolution,
                self.surface.costmap_size,
                self.costmap_wp,
                self.horizon,
                self.costs_wp
            ],
            device="cuda"
        )

        wp.launch(
            kernel=_compute_weights,
            dim= self.number_of_trajectories,
            inputs=[
                self.costs_wp,
                self.min_cost,
                self.weights_wp,
                self.temperature
            ],
            device="cuda"
        )
        
        wp.launch(
            kernel=_compute_sum,
            dim= self.number_of_trajectories,
            inputs=[
                self.weights_wp,
                self.weights_sum
            ],
            device="cuda"
        )

        # Reset the warp parameter to compute the optimal trajectory
        self.reset("sim")

        wp.launch(
            kernel=_compute_weighted_sum,
            dim= self.number_of_trajectories,
            inputs=[
                self.weights_wp, 
                self.number_of_iterations,
                self.u1, 
                self.u2,
                self.weights_sum,
                self.optimal_u1_wp,
                self.optimal_u2_wp
            ],
            device="cuda"
        )

        wp.launch(
            _convert_inputs_to_velocities,
            dim=1,  
            inputs=[
                self.number_of_iterations, 
                self.robot.radius,  
                self.robot.left_wheel_speed, 
                self.robot.right_wheel_speed, 
                self.v_min_linear,
                self.v_max_linear,
                self.v_min_angular,
                self.v_max_angular,
                self.optimal_u1_wp, 
                self.optimal_u2_wp,
                self.optimal_lin_vel_wp,
                self.optimal_ang_vel_wp,
                3.0,
                0.92
            ],
            device="cuda"
        )



        wp.launch(
            kernel=_generate_trajectories_kernel,
            dim=1,
            inputs=[
                self.position_sim,
                -self.surface.half_width,
                -self.surface.half_width,
                self.surface.grid_size,
                self.q_sim,
                self.surface.resolution,
                self.Z_wp,
                self.height_sim,
                self.normal_sim,
                self.heading_vectors_sim,
                self.previous_heading_vector,
                self.number_of_iterations,
                self.optimal_lin_vel_wp,
                self.optimal_ang_vel_wp,
                self.dt,
                self.trajectories_sim,
                self.left_wheel_pos_sim,
                self.right_wheel_pos_sim
            ],
            device="cuda"
        )

        # print("optimal_lin_vel_wp:", self.optimal_lin_vel_wp)
        # print("optimal_ang_vel_wp:", self.optimal_ang_vel_wp)
        # print("trajectories_sim:", self.trajectories_sim)

        ### TEST TO REMOVE

        # self.cost.zero_()
        # wp.launch(
        #     kernel=_evaluate_trajectories_kernel,
        #     dim= 1,
        #     inputs=[
        #         self.robot.x[-1], 
        #         self.robot.y[-1], 
        #         self.goal, 
        #         self.goal_orientation, 
        #         self.trajectories_sim, 
        #         self.optimal_lin_vel_wp,
        #         self.v_max_linear,
        #         self.number_of_trajectories, 
        #         self.number_of_iterations, 
        #         self.surface.half_width,
        #         self.surface.resolution,
        #         self.surface.grid_size,
        #         self.costmap_wp,
        #         self.horizon,
        #         self.cost
        #     ],
        #     device="cuda"
        # )
        # current_cost = self.cost.numpy()[0]
        # self.total_cost += current_cost


    def run(self, proj):
        # Initialise all the warp arrays in the GPU memory
        self.warp_setup()

        # start_time = time.time()
        # frame_idx = 0  # Initialize frame counter

        # Main loop that keeps running as long as the robot has not reached the goal yet
        while (abs(self.robot.x[-1] - self.goal_x) > 0.5 or abs(self.robot.y[-1] - self.goal_y) > 0.5) and self.loop < 3500:

            # Reset the warp arrays at each loop
            self.reset("controller")
            self.MPPI_step(proj=proj)

            self.robot.update_position(self.trajectories_sim.numpy()[0][0], 
                                       self.trajectories_sim.numpy()[0][1], 
                                       self.trajectories_sim.numpy()[0][2], 
                                       self.heading_vectors_sim.numpy()[0])

            lin_vel = self.optimal_lin_vel_wp.numpy()[0]
            ang_vel = self.optimal_ang_vel_wp.numpy()[0]

            self.std_dev_u1 = np.maximum(0.4, 0.4 - ang_vel*ang_vel)
            self.std_dev_u2 = np.maximum(0.4, 0.4 + ang_vel*ang_vel)

            self.robot.lin_vel.append(lin_vel)
            self.robot.ang_vel.append(ang_vel)

            self.robot.left_wheel_speed = lin_vel - ang_vel * self.robot.radius/2
            self.robot.right_wheel_speed = lin_vel + ang_vel * self.robot.radius/2

            # if self.loop % 50 == 0:
            #     all_trajectories = np.concatenate((self.trajectories.numpy(), self.trajectories_sim.numpy()), axis=0)
            #     trajectories_chunks = split_trajectory(all_trajectories, self.number_of_trajectories+1, self.number_of_iterations) 
            #     plot_2d_surface_with_trajectory(surface.X, surface.Y, surface.Z, surface.costmap, trajectories_chunks) 
            
            #     print(self.robot.lin_vel[-1])
            #     print(self.optimal_u1_wp.numpy()[0])
            #     print(self.std_dev_u1)
            #     print(self.std_dev_u2)
            # # if self.cost.numpy()[0] > 2000:
                
            #     # print(self.cost.numpy()[0])
            #     print(ang_vel)

            self.loop += 1


        # elapsed_time = time.time()-start_time
        # print("Duration:", elapsed_time)
        print("Number of loops:", self.loop)
        # print("Average time duration of one loop", (elapsed_time/self.loop))



"""
Surface settings
"""

# bumps = []
# obstacles = []
# rng = np.random.RandomState(99)
# for i in range(750):
#     obstacles.append([rng.uniform(-50.0, 50.0), rng.uniform(-50.0, 50.0), rng.uniform(0.0, 0.4)])




# bumps = [
#     ((-2.7, -19.0), 3.4, 12.23),
#     ((-0.57, -0.05), 4.39, 11.52),
#     ((-48.56, 12.78), 3.6, 12.4),
#     ((-27.89, 38.56), 4.0, 12.7),
#     ((-50.12, 19.34), 3.7, 13.0),
#     ((20.45, -48.78), 4.4, 12.9),
#     ((-20.67, -40.12), 4.2, 12.9),
#     ((42.78, 21.56), 4.5, 12.7),
#     ((-36.12, -33.34), 3.9, 13.0),
# ]

# # surface = Surface("imported", "test_nathan.npy", "manual", "", grid_size=1500, half_width=75.0, bumps=bumps, radius_robot=0.3, obstacles=obstacles)
# surface = Surface("imported", "test_nathan.npy", "imported", "costmap_750_transformed.npy", grid_size=1500, half_width=75.0, bumps=bumps, radius_robot=0.3, obstacles=obstacles)
# plot_2d_surface_with_trajectory(surface.X, surface.Y, surface.Z, surface.costmap, []) 




# """
# Running multiple times in a row with random settings
# """

# for i in range(1, 60):
#     print("\n This is loop", i)
#     if i % 2 == 0:
#         x_start = np.random.uniform(-70.0, -55.0)
#         y_start = np.random.uniform(-50.0, 50.0)
#         x_goal = np.random.uniform(55.0, 70.0)
#         y_goal = np.random.uniform(-50.0, 50.0)
#         x_hr = np.random.uniform(0.0, 1.0)
#         y_hr = np.random.uniform(-1.0, 1.0)
#     else:
#         x_start = np.random.uniform(-50.0, 50.0)
#         y_start = np.random.uniform(-70.0, -55.0)   
#         x_goal = np.random.uniform(-50.0, 50.0)
#         y_goal = np.random.uniform(55.0, 70.0)
#         x_hr = np.random.uniform(-1.0, 1.0)
#         y_hr = np.random.uniform(0.0, 1.0)


#     robot_2d = Robot(x=x_start, y=y_start, heading_vector=[x_hr, y_hr, 0.0], config_file="warp_implementation/config.yaml")
#     robot_3d = Robot(x=x_start, y=y_start, heading_vector=[x_hr, y_hr, 0.0], config_file="warp_implementation/config.yaml")

#     controller_2d = MPPI_Controller(surface, robot_2d, "warp_implementation/config.yaml", goal_x=x_goal, goal_y=y_goal, goal_orientation=2.2)
#     controller_3d = MPPI_Controller(surface, robot_3d, "warp_implementation/config.yaml", goal_x=x_goal, goal_y=y_goal, goal_orientation=2.2)

#     # 2D
#     controller_2d.run("2d")
#     trajectory_2d = np.array(list(zip(robot_2d.x, robot_2d.y, robot_2d.z)))
#     # plot_2d_surface_with_trajectory(surface.X, surface.Y, surface.Z, surface.costmap, [trajectory_2d]) 

#     # 3D
#     controller_3d.run("3d")
#     trajectory_3d = np.array(list(zip(robot_3d.x, robot_3d.y, robot_3d.z)))
#     # if controller_3d.loop > 3600:
#     # plot_2d_surface_with_trajectory(surface.X, surface.Y, surface.Z, surface.costmap, [trajectory_3d]) 

#     # saving data
#     robot_2d.lin_vel.append(robot_2d.lin_vel[-1])  
#     robot_3d.lin_vel.append(robot_3d.lin_vel[-1])  

    # data_dict_2d = {
    #     'Step': [i*2 - 1]*len(trajectory_2d),  
    #     'X': [point[0] for point in trajectory_2d], 
    #     'Y': [point[1] for point in trajectory_2d], 
    #     'Z': [point[2] for point in trajectory_2d],  
    #     'Linear_Velocity': robot_2d.lin_vel,  
    #     'Start': [[x_start, y_start]] * len(trajectory_2d), 
    #     'Goal': [[x_goal, y_goal]] * len(trajectory_2d) 
    # }

    # data_dict_3d = {
    #     'Step': [i * 2]*len(trajectory_3d),  
    #     'X': [point[0] for point in trajectory_3d], 
    #     'Y': [point[1] for point in trajectory_3d], 
    #     'Z': [point[2] for point in trajectory_3d],  
    #     'Linear_Velocity': robot_3d.lin_vel,  
    #     'Start': [[x_start, y_start]] * len(trajectory_3d), 
    #     'Goal': [[x_goal, y_goal]] * len(trajectory_3d) 
    # }

    # df = pd.DataFrame(data_dict_2d)
    # df.to_csv('march21_350_slopes.csv', mode='a', index=False, header=False)

    # df = pd.DataFrame(data_dict_3d)
    # df.to_csv('march21_350_slopes.csv', mode='a', index=False, header=False)



"""
Manual launch of either an MPPI run or of one rollouts spread only
"""

# x_start = np.random.uniform(-70.0, -55.0)
# y_start = np.random.uniform(-50.0, 50.0)
# x_goal = np.random.uniform(55.0, 70.0)
# y_goal = np.random.uniform(-50.0, 50.0)
# x_hr = np.random.uniform(0.0, 1.0)
# y_hr = np.random.uniform(-1.0, 1.0)

# # x_start = -60.57
# # y_start = -60.23

# # x_goal = 65.80
# # y_goal = 65.4


# robot = Robot(x=x_start, y=y_start, heading_vector=[x_hr, y_hr, 0.0], config_file="warp_implementation/config.yaml")
# controller = MPPI_Controller(surface, robot, "warp_implementation/config.yaml", goal_x=x_goal, goal_y=y_goal, goal_orientation=2.2)


# command = "run_MPPI"
# proj = "2d"

# if command == "run_MPPI":
#     controller.run(proj)
#     trajectory = np.array(list(zip(robot.x, robot.y, robot.z)))
    
#     # plot_surface_with_trajectory(surface.X, surface.Y, surface.Z, surface.costmap, [trajectory], surface.half_width*2, surface.half_width*2, surface.half_width*2) 
#     plot_2d_surface_with_trajectory(surface.X, surface.Y, surface.Z, surface.costmap, [trajectory]) 


# elif command == "run_one_spread":
#     controller.warp_setup()
#     controller.MPPI_step(proj)
#     all_trajectories = np.concatenate((controller.trajectories.numpy(), controller.trajectories_sim.numpy()), axis=0)
#     trajectories_chunks = split_trajectory(all_trajectories, controller.number_of_trajectories+1, controller.number_of_iterations) 

#     # plot_surface_with_trajectory(surface.X, surface.Y, surface.Z, surface.costmap, trajectories_chunks, surface.half_width*2, surface.half_width*2, surface.half_width*2) 
#     plot_2d_surface_with_trajectory(surface.X, surface.Y, surface.Z, surface.costmap, trajectories_chunks) 


# length, angle_up, angle_down, distance_up = compute_path_metrics(trajectory)


# print(f"Total Path Length: {length:.2f} meters")
# print(f"Total Angle Climbed Up: {angle_up:.2f} degrees")
# print(f"Total Angle Climbed Down: {angle_down:.2f} degrees")
# print(f"Total Distance Climbed Up: {distance_up:.2f} m")





"""
Saving the data obtained
"""

# save_results = True

# if save_results:
#     ask_question = input("Do you want to erase the previous results and overwrite them? (yes/no): ").strip().lower()
    
#     if ask_question == "yes":
#         robot.lin_vel.append(robot.lin_vel[-1])  
    
#         data_dict = {
#             'Step': [1 for i in range(len(trajectory))],  
#             'X': [point[0] for point in trajectory], 
#             'Y': [point[1] for point in trajectory], 
#             'Z': [point[2] for point in trajectory],  
#             'Linear_Velocity': robot.lin_vel,  
#             'Start': [[x_start, y_start]] * len(trajectory), 
#             'Goal': [[x_goal, y_goal]] * len(trajectory) 
#         }

#         df = pd.DataFrame(data_dict)
#         df.to_csv('trajectories_data.csv', mode='a', index=False, header=False)





"""
Running statistics
"""

# average_ratio_of_loops_gained = 0
# average_ratio_of_distance_gained = 0
# average_ratio_of_climbed_degrees_gained = 0
# average_ratio_of_up_slope_gained = 0
# average_cost_gained = 0

# obstacles = []
# rng = np.random.RandomState(99)
# for i in range(325):
#     obstacles.append([rng.uniform(-50.0, 50.0), rng.uniform(-50.0, 50.0), rng.uniform(0.0, 0.4)])

# bumps = [
#     ((-2.7, -19.0), 3.4, 12.23),
#     ((-0.57, -0.05), 4.39, 11.52),
#     ((-48.56, 12.78), 3.6, 12.4),
#     ((-27.89, 38.56), 4.0, 12.7),
#     ((-50.12, 19.34), 3.7, 13.0),
#     ((20.45, -48.78), 4.4, 12.9),
#     ((-20.67, -40.12), 4.2, 12.9),
#     ((42.78, 21.56), 4.5, 12.7),
#     ((-36.12, -33.34), 3.9, 13.0),
# ]

# surface = Surface("imported", "test_nathan.npy", grid_size=1500, half_width=75.0, bumps=bumps, radius_robot=0.3, obstacles=obstacles)
# plot_2d_surface_with_trajectory(surface.X, surface.Y, surface.Z, surface.costmap, []) 

# for i in range(1, 100):
#     print("\n This is loop", i)


#     if i % 2 == 0:
#         x_start = np.random.uniform(-70.0, -55.0)
#         y_start = np.random.uniform(-50.0, 50.0)
#         x_goal = np.random.uniform(55.0, 70.0)
#         y_goal = np.random.uniform(-50.0, 50.0)
#         x_hr = np.random.uniform(0.0, 1.0)
#         y_hr = np.random.uniform(-1.0, 1.0)
#     else:
#         x_start = np.random.uniform(-50.0, 50.0)
#         y_start = np.random.uniform(-70.0, -55.0)   
#         x_goal = np.random.uniform(-50.0, 50.0)
#         y_goal = np.random.uniform(55.0, 70.0)
#         x_hr = np.random.uniform(-1.0, 1.0)
#         y_hr = np.random.uniform(0.0, 1.0)

#     x_hr = np.random.uniform(-1.0, 1.0)
#     y_hr = np.random.uniform(0.0, 1.0)

#     robot_2d = Robot(x=x_start, y=y_start, heading_vector=[x_hr, y_hr, 0.0], radius=0.3)
#     robot_3d = Robot(x=x_start, y=y_start, heading_vector=[x_hr, y_hr, 0.0], radius=0.3)

#     controller_2d = MPPI_Controller(surface, robot_2d, "warp_implementation/config.yaml", goal_x=x_goal, goal_y=y_goal, goal_orientation=2.2)
#     controller_3d = MPPI_Controller(surface, robot_3d, "warp_implementation/config.yaml", goal_x=x_goal, goal_y=y_goal, goal_orientation=2.2)

#     # 2D
#     controller_2d.run("2d")
#     trajectory = np.array(list(zip(robot_2d.x, robot_2d.y, robot_2d.z)))
#     # plot_2d_surface_with_trajectory(surface.X, surface.Y, surface.Z, surface.costmap, [trajectory]) 
#     length_2d, angle_up_2d, angle_down_2d, distance_up_2d = compute_path_metrics(trajectory)
#     # print(f"Total Path Length: {length_2d:.2f} meters")
#     # print(f"Total Angle Climbed Up: {angle_up_2d:.2f} degrees")
#     # print(f"Total Distance Climbed Up: {distance_up_2d:.2f} m")

#     # 3D
#     controller_3d.run("3d")
#     trajectory = np.array(list(zip(robot_3d.x, robot_3d.y, robot_3d.z)))
#     # plot_2d_surface_with_trajectory(surface.X, surface.Y, surface.Z, surface.costmap, [trajectory]) 
#     length_3d, angle_up_3d, angle_down_3d, distance_up_3d = compute_path_metrics(trajectory)
#     # print(f"Total Path Length: {length_3d:.2f} meters")
#     # print(f"Total Angle Climbed Up: {angle_up_3d:.2f} degrees")
#     # print(f"Total Distance Climbed Up: {distance_up_3d:.2f} m")

#     if controller_3d.loop == 5000:
#         print("oh ptn le 3d a foire")
#         plot_2d_surface_with_trajectory(surface.X, surface.Y, surface.Z, surface.costmap, [trajectory]) 

#         if controller_2d.loop == 5000:
#             print("wah dinguerie le 2d aussi")
#         continue
    
#     if controller_2d.loop == 5000:
#         print("hop la le 2d uniquement a fail")
#         continue
    
#     if controller_2d.total_cost >= controller_3d.total_cost:
#         current_cost_gained = (controller_2d.total_cost - controller_3d.total_cost)/controller_2d.total_cost
#         average_cost_gained = average_cost_gained * (i-1)/i + current_cost_gained * 1/i
#     else:
#         current_cost_gained = (controller_2d.total_cost - controller_3d.total_cost)/controller_3d.total_cost
#         average_cost_gained = average_cost_gained * (i-1)/i + current_cost_gained * 1/i

#     if controller_2d.loop >= controller_3d.loop:
#         current_loop_gained_ratio = (controller_2d.loop - controller_3d.loop)/controller_2d.loop
#         average_ratio_of_loops_gained = average_ratio_of_loops_gained * (i-1)/i + current_loop_gained_ratio * 1/i
#     else:
#         current_loop_gained_ratio = (controller_2d.loop - controller_3d.loop)/controller_3d.loop
#         average_ratio_of_loops_gained = average_ratio_of_loops_gained * (i-1)/i + current_loop_gained_ratio * 1/i

#     if length_2d >= length_3d:
#         current_ratio_of_distance_gained = (length_2d - length_3d)/length_2d
#         average_ratio_of_distance_gained = average_ratio_of_distance_gained * (i-1)/i + current_ratio_of_distance_gained * 1/i
#     else:
#         current_ratio_of_distance_gained = (length_2d - length_3d)/length_3d
#         average_ratio_of_distance_gained = average_ratio_of_distance_gained * (i-1)/i + current_ratio_of_distance_gained * 1/i

#     if angle_up_2d >= angle_up_3d:
#         current_ratio_of_climbed_degrees_gained = (angle_up_2d - angle_up_3d)/angle_up_2d
#         average_ratio_of_climbed_degrees_gained = average_ratio_of_climbed_degrees_gained * (i-1)/i + current_ratio_of_climbed_degrees_gained * 1/i
#     else:
#         current_ratio_of_climbed_degrees_gained = (angle_up_2d - angle_up_3d)/angle_up_3d
#         average_ratio_of_climbed_degrees_gained = average_ratio_of_climbed_degrees_gained * (i-1)/i + current_ratio_of_climbed_degrees_gained * 1/i
    
#     if distance_up_2d >= distance_up_3d:
#         current_ratio_of_up_slope_gained = (distance_up_2d - distance_up_3d)/distance_up_2d
#         average_ratio_of_up_slope_gained = average_ratio_of_up_slope_gained * (i-1)/i + current_ratio_of_up_slope_gained * 1/i
#     else:
#         current_ratio_of_up_slope_gained = (distance_up_2d - distance_up_3d)/distance_up_3d
#         average_ratio_of_up_slope_gained = average_ratio_of_up_slope_gained * (i-1)/i + current_ratio_of_up_slope_gained * 1/i

#     print("\n")

#     print("Cost 2d:", controller_2d.total_cost)
#     print("Cost 3d:", controller_3d.total_cost)

#     print("Average cost gained:", average_cost_gained)



