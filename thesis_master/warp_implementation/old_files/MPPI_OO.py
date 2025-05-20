from projection_warp import _find_corners_heights, \
                            _bilinear_interpolation, \
                            _compute_normal, \
                            _compute_tangent_vector, \
                            _compute_next_waypoint, \
                            _generate_trajectories_kernel

from sampling_warp import _generate_velocities_kernel

from critics_warp import _evaluate_trajectories_kernel, _compute_weights, _compute_sum, _compute_weighted_sum
import warp as wp
import numpy as np
import matplotlib.pyplot as plt
import time
import yaml

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
        if i < len(trajectories) - 1:  
            color = 'blue'  # All except the last one are blue
            s = 3
            alpha = 0.3  # Make blue points more transparent
        else:
            color = 'red'   # The last trajectory is red
            s = 10
            alpha = 1.0  # Keep red fully visible

        if i % 2 == 0 or i == (len(trajectories) - 1):
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
    
    # Add legend
    ax.legend()
    
    # Show the plot
    plt.show()

class Surface:
    def __init__(self, grid_size, half_width, bumps, obstacles=[]):
        self.grid_size = grid_size
        self.half_width = half_width
        self.resolution = 2 * half_width / grid_size
        self.X, self.Y, self.Z = self.create_surface(bumps)
        self.obstacles = obstacles

    def create_surface(self, bumps):
        x = np.linspace(-self.half_width, self.half_width, self.grid_size)
        y = np.linspace(-self.half_width, self.half_width, self.grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for bump_center, bump_height, bump_width in bumps:
            Z += bump_height * np.exp(-((X - bump_center[0])**2 + (Y - bump_center[1])**2) / (2 * bump_width**2))
        Z += 0.85 * np.arctan(0.5 * X)
        return X, Y, Z

class Robot:
    def __init__(self, x, y, heading_vector, radius):
        self.x = [x]
        self.y = [y]
        self.z = [0]
        self.heading_vector = np.array(heading_vector) / np.linalg.norm(heading_vector)
        self.radius = radius

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
        self.initial_linear_velocity = config['velocities']['initial_linear_velocity']
        self.std_dev_linear = config['velocities']['std_dev_linear']
        self.min_linear_velocity = config['velocities']['min_linear_velocity']
        self.max_linear_velocity = config['velocities']['max_linear_velocity']
        self.initial_angular_velocity = config['velocities']['initial_angular_velocity']
        self.std_dev_angular = config['velocities']['std_dev_angular']
        self.min_angular_velocity = config['velocities']['min_angular_velocity']
        self.max_angular_velocity = config['velocities']['max_angular_velocity']

        # Parameters for the cost evaluation
        self.temperature = config['cost_evaluation']['temperature']
        self.horizon = self.dt*self.max_linear_velocity*self.number_of_iterations

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
            self.optimal_lin_vel_wp.zero_() 
            self.optimal_ang_vel_wp.zero_()
            self.position_sim = wp.array([wp.vec2f(self.robot.x[-1], self.robot.y[-1])], dtype=wp.vec2f)
            self.previous_heading_vector = wp.vec3f(self.robot.heading_vector/np.linalg.norm(self.robot.heading_vector))

    def MPPI_step(self):
        # Run one MPPI step which consists in:
        # - sampling random inputs sequences
        # - computing the corresponding trajectories in the 2.5D space
        # - evaluating those trajectories
        # - computing the estimated optimal trajectory

        wp.launch(
            _generate_velocities_kernel,
            dim=self.number_of_trajectories*self.number_of_iterations,  
            inputs=[
                self.number_of_iterations, 
                np.random.randint(self.number_of_iterations+1, 1000),  
                self.optimal_lin_vel_wp, 
                self.optimal_ang_vel_wp, 
                self.std_dev_linear, 
                self.min_linear_velocity, 
                self.max_linear_velocity, 
                self.std_dev_angular, 
                self.min_angular_velocity, 
                self.max_angular_velocity,
                self.linear_velocities, 
                self.angular_velocities
            ],
            device="cuda"
        )


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
                self.max_linear_velocity,
                self.number_of_trajectories, 
                self.number_of_iterations, 
                self.obstacles_wp, 
                self.robot.radius, 
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
                self.linear_velocities, 
                self.angular_velocities,
                self.weights_sum,
                self.optimal_lin_vel_wp,
                self.optimal_ang_vel_wp
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
        
    def run(self):
        # Initialise all the warp arrays in the GPU memory
        self.warp_setup()

        start_time = time.time()

        # Main loop that keeps running as long as the robot has not reached the goal yet
        while abs(self.robot.x[-1] - self.goal_x) > 0.5 or abs(self.robot.y[-1] - self.goal_y) > 0.5: # and self.loop < 2000:

            # Reset the warp arrays at each loop
            self.reset("controller")
            self.MPPI_step()
            self.robot.update_position(self.trajectories_sim.numpy()[0][0], 
                                       self.trajectories_sim.numpy()[0][1], 
                                       self.trajectories_sim.numpy()[0][2], 
                                       self.heading_vectors_sim.numpy()[0])
            self.loop += 1

        elapsed_time = time.time()-start_time
        print("Duration:", elapsed_time)
        print("Number of loop:", controller.loop)
        print("Average time duration of one loop", (elapsed_time/controller.loop))

bumps = [
    ((-18.32, -8.94), 2.48, 3.62),
    ((-13.01, 6.74), 4.45, 5.85),
    ((-8.64, -14.23), 1.12, 4.39),
    ((-3.57, 12.05), 2.39, 1.92),
    ((0.97, -17.81), 1.62, 2.91),
    ((3.15, -1.56), 2.63, 2.21),
    ((6.13, 3.41), 2.14, 1.89),
    ((9.87, 16.38), 1.45, 3.74),
    ((14.94, 15.64), 2.89, 4.02),
    ((19.83, -9.56), 2.58, 1.72),
    ((-6.34, 5.56), 0.58, 4.55),
    ((-12.21, -13.32), 1.01, 3.89),
    ((-5.21, -5.32), 4.01, 3.89),
]

surface = Surface(grid_size=600, half_width=30.0, bumps=bumps)
robot = Robot(x=15.0, y=-17.3, heading_vector=[0.0, 0.5, 0.0], radius=0.01)
controller = MPPI_Controller(surface, robot, "warp_implementation/config.yaml", goal_x=-15.6, goal_y=17.13, goal_orientation=2.2)

# command = "run_MPPI"
command = "run_one_spread"

if command == "run_MPPI":
    controller.run()
    trajectory = np.array(list(zip(robot.x, robot.y, robot.z)))
    plot_surface_with_trajectory(surface.X, surface.Y, surface.Z, [trajectory], surface.half_width*2, surface.half_width*2, surface.half_width) 

elif command == "run_one_spread":
    controller.warp_setup()
    controller.MPPI_step()
    plot_surface_with_trajectory(surface.X, surface.Y, surface.Z, [controller.trajectories.numpy(), controller.trajectories_sim.numpy()], surface.half_width*2, surface.half_width*2, surface.half_width) 


# start_time = time.time()
# elapsed_time = time.time()-start_time
# print("Duration:", elapsed_time)
# print("Number of loop:", controller.loop)
# print("Average time duration of one loop", (elapsed_time/controller.loop))

# 0.2ms for self.reset("sim")
# 0.25ms for wp.launch(_generate_trajectories_kernel,dim=1,...
# 0.5ms for self.robot.update_position()
# --> average time of a control loop: 1.8ms - 0.2 - 0.25 - 0.5 = <1ms