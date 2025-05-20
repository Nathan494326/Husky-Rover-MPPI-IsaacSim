from projection_warp import _find_corners_heights, \
                            _bilinear_interpolation, \
                            _compute_normal, \
                            _compute_tangent_vector, \
                            _generate_trajectories_kernel, \
                            _generate_trajectories_2D_kernel

from sampling_warp import _generate_velocities_kernel, _generate_inputs_kernel, _convert_inputs_to_velocities

from critics_warp import _evaluate_trajectories_kernel, _compute_weights, _compute_sum, _compute_weighted_sum
import warp as wp
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
            alpha = 0.8  # Make blue points more transparent
        else:
            color = 'blue'   # The last trajectory is red
            s = 20
            alpha = 1.0  # Keep red fully visible

        if i % 1 == 0 or i == (len(trajectories) - 1):
            ax.scatter(trajectory[::4, 0], trajectory[::4, 1], trajectory[::4, 2], 
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
    

    ax.view_init(elev=90, azim=-90)  # Top-down view
    ax.dist = 5

    # # plt.savefig(f"{save_path}/frame_{frame_idx:04d}.png", dpi=300)
    # # plt.close(fig)
    plt.show()


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
    def __init__(self, grid_size, half_width, bumps, radius_robot, obstacles=[]):
        self.grid_size = grid_size
        self.half_width = half_width
        self.r_robot = radius_robot
        self.resolution = 2 * half_width / grid_size
        self.X, self.Y, self.Z = self.create_surface(bumps)
        self.obstacles = obstacles
        self.costmap = self.create_obstacles_costmap(obstacles, self.X, self.Y)

    def create_surface(self, bumps):
        x = np.linspace(-self.half_width, self.half_width, self.grid_size)
        y = np.linspace(-self.half_width, self.half_width, self.grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        Z += 2 *np.exp(-((X-20)**2 + (Y-20)**2) / (2 * 20.0**2))  # Inverted Gaussian


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
    
    def create_obstacles_costmap(self, obstacles, X, Y):
        obs_costmap = np.zeros((self.grid_size, self.grid_size))

        for obs in obstacles:
            x_obs, y_obs, r_obs = obs
            
            # Compute the mask for the obstacle area
            mask = (X - x_obs) ** 2 + (Y - y_obs) ** 2 <= (r_obs + self.r_robot + 0.2) ** 2
            
            # Set obstacle regions to 1
            obs_costmap[mask] = 1.0

        return obs_costmap



class Robot:
    def __init__(self, x, y, heading_vector, radius):
        self.x = [x]
        self.y = [y]
        self.z = [0]
        self.heading_vector = np.array(heading_vector) / np.linalg.norm(heading_vector)
        self.radius = radius
        self.left_wheel_speed = 1.0
        self.right_wheel_speed = 1.0

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
        
        self.robot = robot
        self.surface = surface
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_orientation = goal_orientation
        self.loop = 0

        # General parameters of MPPI
        self.number_of_iterations = config['controller']['number_of_iterations']
        self.dt = config['controller']['dt']
        self.number_of_trajectories = config['controller']['number_of_trajectories']
        
        # Parameters for the random trajectories generation

        self.initial_linear_velocity = 1.0 # config['velocities']['initial_linear_velocity']
        self.std_dev_u1 = 0.1
        self.min_u1 = -1.0
        self.max_u1 = 1.0
        self.std_dev_u2 = 0.1
        self.min_u2 = -1.0
        self.max_u2 = 1.0
        self.initial_angular_velocity = 0.4 # config['velocities']['initial_angular_velocity']
        self.v_min_linear = 0.0
        self.v_max_linear = 2.0
        self.v_min_angular = -0.8
        self.v_max_angular = 0.8

        # Parameters for the cost evaluation
        self.temperature = config['cost_evaluation']['temperature']
        self.horizon = self.dt*self.v_max_linear*self.number_of_iterations
        
    def warp_setup(self):
        wp.init()

        # Variables for the random inputs sequences generation
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
        self.obstacles_wp = wp.array(self.surface.obstacles, dtype=wp.vec3f, device="cuda")
        self.costs_wp = wp.zeros(self.number_of_trajectories, dtype=float, device="cuda")
        self.weights_wp = wp.zeros(self.number_of_trajectories, dtype=float, device="cuda")
        self.min_cost = wp.array([np.inf], dtype=float)
        self.weights_sum = wp.array([0.0], dtype=float)

        # Variables for the optimal sequences computations
        self.linear_velocities_wp = wp.zeros(self.number_of_iterations, dtype=float, device="cuda")
        self.angular_velocities_wp = wp.zeros(self.number_of_iterations, dtype=float, device="cuda")

        # Variables for the simulation of the robot part
        self.position_sim = wp.array([wp.vec2f(self.robot.x[-1], self.robot.y[-1])], dtype=wp.vec2f, device="cuda")
        self.q_sim = wp.zeros(1, dtype=wp.mat22f, device="cuda") 
        self.height_sim = wp.zeros(1, dtype=float, device="cuda")
        self.normal_sim = wp.zeros(1, dtype=wp.vec3f, device="cuda")
        self.heading_vectors_sim = wp.zeros(self.number_of_iterations, dtype=wp.vec3f, device="cuda")
        self.trajectories_sim = wp.zeros((self.number_of_iterations), dtype=wp.vec3f, device="cuda") 


        """
        following is for integration of kinematics
        """

        self.optimal_u1_wp = wp.ones(self.number_of_iterations, dtype=float, device="cuda")
        self.optimal_u2_wp = wp.ones(self.number_of_iterations, dtype=float, device="cuda")
        self.u1 = wp.zeros((self.number_of_trajectories*self.number_of_iterations), dtype=float, device="cuda")  
        self.u2 =wp.zeros((self.number_of_trajectories*self.number_of_iterations), dtype=float, device="cuda")  

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
            # self.optimal_lin_vel_wp.zero_() 
            # self.optimal_ang_vel_wp.zero_()
            self.position_sim = wp.array([wp.vec2f(self.robot.x[-1], self.robot.y[-1])], dtype=wp.vec2f)
            self.previous_heading_vector = wp.vec3f(self.robot.heading_vector/np.linalg.norm(self.robot.heading_vector))
            self.optimal_u1_wp.zero_()
            self.optimal_u2_wp.zero_()

    def MPPI_step(self, proj, u1, u2):
        # Run one MPPI step which consists in:
        # - sampling random inputs sequences
        # - computing the corresponding trajectories in the 2.5D space
        # - evaluating those trajectories
        # - computing the estimated optimal trajectory

        # wp.launch(
        #     _generate_inputs_kernel,
        #     dim=self.number_of_trajectories*self.number_of_iterations,  
        #     inputs=[
        #         self.number_of_iterations, 
        #         np.random.randint(self.number_of_iterations+1, 1000),  
        #         self.optimal_u1_wp, 
        #         self.optimal_u2_wp, 
        #         self.std_dev_u1, 
        #         self.min_u1, 
        #         self.max_u1, 
        #         self.std_dev_u2, 
        #         self.min_u2, 
        #         self.max_u2,
        #         self.u1, 
        #         self.u2
        #     ],
        #     device="cuda"
        # )


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
                u1, 
                u2,
                self.linear_velocities,
                self.angular_velocities,
                2.0,
                0.95
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
                    # self.normal,
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
                    self.trajectories
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
                self.linear_velocities,
                self.v_max_linear,
                self.number_of_trajectories, 
                self.number_of_iterations, 
                self.surface.half_width,
                self.surface.resolution,
                self.surface.grid_size,
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
                u1, 
                u2,
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
                2.0,
                0.95
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
                self.trajectories_sim
            ],
            device="cuda"
        )

        
    def run(self, proj):
        # Initialise all the warp arrays in the GPU memory
        self.warp_setup()

        start_time = time.time()
        frame_idx = 0  # Initialize frame counter

        # Main loop that keeps running as long as the robot has not reached the goal yet
        while (abs(self.robot.x[-1] - self.goal_x) > 1.0 or abs(self.robot.y[-1] - self.goal_y) > 1.0) and self.loop < 3000:

            # Reset the warp arrays at each loop
            self.reset("controller")
            self.MPPI_step(proj=proj)

            self.robot.update_position(self.trajectories_sim.numpy()[0][0], 
                                       self.trajectories_sim.numpy()[0][1], 
                                       self.trajectories_sim.numpy()[0][2], 
                                       self.heading_vectors_sim.numpy()[0])

            lin_vel = self.optimal_lin_vel_wp.numpy()[0]
            ang_vel = self.optimal_ang_vel_wp.numpy()[0]
            self.robot.left_wheel_speed = lin_vel - ang_vel * self.robot.radius/2
            self.robot.right_wheel_speed = lin_vel + ang_vel * self.robot.radius/2
            # if self.loop % 50 == 0:

            #     all_trajectories = np.concatenate((self.trajectories.numpy(), self.trajectories_sim.numpy()), axis=0)
            #     trajectories_chunks = split_trajectory(all_trajectories, controller.number_of_trajectories+1, controller.number_of_iterations) 
            #     plot_surface_with_trajectory(surface.X, surface.Y, surface.Z, self.surface.costmap, trajectories_chunks, surface.half_width*2, surface.half_width*2, surface.half_width*2, frame_idx=frame_idx)

            #     frame_idx += 1  # Increment frame count
            self.loop += 1


        elapsed_time = time.time()-start_time
        print("Duration:", elapsed_time)
        print("Number of loop:", self.loop)
        print("Average time duration of one loop", (elapsed_time/self.loop))


bumps = [
    ((-18.32, -8.94), 3.48, 4.62),
    ((-15.01, 5.0), 5.45, 3.85),
    ((-8.64, -14.23), 2.12, 4.39),
    ((-3.57, 12.05), 3.39, 2.92),
    ((0.97, -8.81), 2.62, 2.5),
    ((3.15, -1.56), 3.63, 3.21),
    ((6.13, 3.41), 3.14, 2.89),
    ((9.87, 16.38), 2.45, 4.74),
    ((14.94, 15.64), 3.89, 3.02),
    ((19.83, -9.56), 3.58, 2.72),
    ((6.83, -13.56), 3.58, 2.72),
    ((-6.34, 5.56), 1.58, 3.55),
    ((-12.21, -13.32), 2.01, 4.89),
    ((-5.21, -5.32), 5.01, 4.89),
    ((-2.7, 1.0), 3.4, 2.23),
    ((0.7, -16.0), 2.9, 2.5),
    ((-2.7, -19.0), 3.4, 2.23),
    ((3.57, -18.05), 3.39, 2.92),
    ((-13.57, -15.05), 3.39, 2.92),
    ((-4.21, 16.32), 5.01, 4.89),
]

bumps = [
    ((-2.7, -19.0), 3.4, 2.23),
    ((-10.57, -4.05), 4.39, 3.92),
    ((-13.57, -15.05), 3.39, 1.52),
    ((-0.57, -0.05), 3.39, 1.52),
    ((-4.21, 16.32), 5.01, 2.89),
    ((11.37, -16.22), 3.4, 2.23),
    ((1.15, 9.97), 3.39, 1.92),
    ((-7.28, 12.45), 3.39, 1.52),
    ((4.56, -8.76), 5.01, 2.89),
    ((-11.11, 3.27), 3.4, 2.23),
    ((9.87, 6.14), 3.39, 1.92),
    ((-5.32, -12.59), 3.39, 1.52),
    ((12.04, -0.68), 4.01, 2.49)
    
]

obstacles = [
    [10.0, 12.0, 0.5],
    [0.0, 0.0, 2.0],
    [-10.0, 3.0, 1.5],
    [12.0, -7.0, 1.0],
    [-8.0, 0.0, 1.8],
    [9.0, -12.0, 0.5],
    [-7.0, -9.0, 1.2],
    [15.0, 3.0, 1.0],
    [-5.0, -15.0, 1.0],
    [-12.0, -8.0, 0.5],
    [5.0, 8.0, 1.3],
    [-6.0, -4.0, 1.7],
    [8.0, 10.0, 1.19],
    [-11.0, -7.0, 1.4],
    [3.0, 12.0, 1.6],
    [-13.0, -2.0, 0.8],
    [2.0, -6.0, 0.9],
    [7.0, 2.0, 1.2],
    [-9.0, 4.0, 1.0],
    [5.0, -9.0, 0.5]
]
bumps = [((3.0, 3.0), 5, 2.5),
]
obstacles = [
    [2.0, -2.0, 0.5],
    [4.8, -1.8, 0.6],
    [3.5, 1.3, 0.9]
]


surface = Surface(grid_size=600, half_width=30.0, bumps=bumps, radius_robot=0.3, obstacles=obstacles)
# plot_surface_with_trajectory(surface.X, surface.Y, surface.Z, surface.costmap, [], surface.half_width*2, surface.half_width*2, surface.half_width*2) 

######## GOOD RUN #########

x_start = np.random.uniform(-25.0, 25.0)
y_start = np.random.uniform(-25.0, 25.0)

x_goal = np.random.uniform(10.0, 22.0)
y_goal = np.random.uniform(-22.0, 22.0)

#robot = Robot(x=x_start, y=y_start, heading_vector=[0.4776196313015999, -0.7987740239716057, 0.0], radius=0.3)
robot = Robot(x=-2.0, y=0.5, heading_vector=[1.0, 0.0, 0.0], radius=0.3)

#controller = MPPI_Controller(surface, robot, "warp_implementation/config.yaml", goal_x=x_goal, goal_y=y_goal, goal_orientation=2.2)
controller = MPPI_Controller(surface, robot, "warp_implementation/config.yaml", goal_x=20.0, goal_y=0.8, goal_orientation=2.2)

command = "run_one_spread"
proj = "2d"



u1_list = []
u2_list = []

# Define input values for each trajectory
inputs_1 = [0.96, 0.98, 0.99, 1.0, 1.0, 1.0, 1.0]
inputs_2 = [1.0, 1.0, 1.0, 1.0, 0.99, 0.98, 0.96]

for k in range(7):
    u1_list.extend([inputs_1[k]] * 120)  # Corrected: Repeat the value for 80 timesteps
    u2_list.extend([inputs_2[k]] * 120)

# Convert to numpy arrays with the correct shape
u1_array = np.array(u1_list)
u2_array = np.array(u2_list)
print(u1_array.shape)
# Convert to Warp arrays for CUDA
u1 = wp.array(u1_array, dtype=float, device="cuda")
u2 = wp.array(u2_array, dtype=float, device="cuda")

controller.warp_setup()
controller.MPPI_step(proj, u1, u2)
all_trajectories = np.concatenate((controller.trajectories.numpy(), controller.trajectories_sim.numpy()), axis=0)
trajectories_chunks = split_trajectory(all_trajectories, controller.number_of_trajectories+1, controller.number_of_iterations) 

plot_surface_with_trajectory(surface.X, surface.Y, surface.Z, surface.costmap, trajectories_chunks, surface.half_width*2, surface.half_width*2, surface.half_width*2) 






### Case 1
# bumps = [((3.5, 2.6), 5, 2.5)]
# obstacles = [
#     [7.0, 3.5, 0.5],
#     [6.8, -1.5, 0.6],
#     [10.0, 3.0, 1.4]
# ]

# robot = Robot(x=1.6, y=-0.96, heading_vector=[0.55, 0.1, 0.0], radius=0.3)
# controller = MPPI_Controller(surface, robot, "warp_implementation/config.yaml", goal_x=-20.45, goal_y=-0.8, goal_orientation=2.2)




### Case 2
# bumps = [((3.5, 2.6), 5, 2.5),
#          ((2.0, 8.6), 4, 2.5)
# ]
# obstacles = [
#     [7.0, 5.5, 0.5],
#     [4.8, 5.2, 0.6],
#     [6.5, 8.9, 0.9]
# ]

# robot = Robot(x=-1.0, y=3.96, heading_vector=[0.55, 0.3, 0.0], radius=0.3)
# controller = MPPI_Controller(surface, robot, "warp_implementation/config.yaml", goal_x=20.45, goal_y=10.8, goal_orientation=2.2)



### Case 3
# bumps = [((3.0, 3.0), 5, 2.5),
# ]
# obstacles = [
#     [2.0, -2.0, 0.5],
#     [4.8, -1.8, 0.6],
#     [3.5, 1.3, 0.9]
# ]

# robot = Robot(x=-2.0, y=0.5, heading_vector=[1.0, 0.0, 0.0], radius=0.3)
# controller = MPPI_Controller(surface, robot, "warp_implementation/config.yaml", goal_x=20.0, goal_y=0.8, goal_orientation=2.2)
