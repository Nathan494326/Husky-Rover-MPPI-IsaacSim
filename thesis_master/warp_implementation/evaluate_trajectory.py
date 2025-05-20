import pandas as pd
import warp as wp
from critics_warp import _evaluate_trajectories_kernel
import numpy  as np
import matplotlib.pyplot as plt
import heapq
import statistics



def compute_cost(trajectories, linear_velocities, robot_x_end, robot_y_end, goal, 
                 goal_orientation, v_max_linear, number_of_iterations, half_width, 
                 resolution, grid_size, costmap_wp, horizon):
        
    trajectories = wp.array(trajectories, dtype=wp.vec3f, device="cuda")
    linear_velocities = wp.array(linear_velocities, dtype=float, device="cuda")
    costs_wp = wp.zeros(1, dtype=float, device="cuda")
    wp.launch(
        kernel=_evaluate_trajectories_kernel,
        dim=1,
        inputs=[
            robot_x_end, 
            robot_y_end, 
            goal, 
            goal_orientation, 
            trajectories, 
            linear_velocities,
            v_max_linear,
            1, 
            number_of_iterations, 
            half_width,
            resolution,
            grid_size,
            costmap_wp,
            horizon,
            costs_wp
        ],
        device="cuda"
    )
    return costs_wp.numpy()[0]


def compute_length(trajectory):
    if len(trajectory) < 2:
        return 0.0  # No path to measure if fewer than 2 waypoints
    
    length = 0.0
    sampled_waypoints = trajectory[::5]  # Take one waypoint out of 10
    
    for i in range(len(sampled_waypoints) - 1):
        length += np.linalg.norm(sampled_waypoints[i + 1] - sampled_waypoints[i])
    
    return length     


df = pd.read_csv('march21_500trajs.csv')

robot_x_end = 200000.0
robot_y_end = -200000.0 
goal = 400000.0
goal_orientation = 0.0 
horizon = 10.0 

half_width = 75.0
grid_size = 1500
resolution = 2 * half_width / grid_size

v_max_linear = 2.0
costmap = np.load("costmap_750_transformed.npy") 
costmap_wp = wp.array(costmap.flatten(), dtype=wp.float32, device="cuda") 

current_traj_idx = 1
number_of_iterations = 0
trajectories = np.empty((0,3))
linear_velocities = np.array([])

costs_2d = []
costs_3d = []

for i, row in df.iterrows():
    if row['Step'] != current_traj_idx:
        if number_of_iterations > 3499:
            if row['Step'] % 2 == 0:
                print("Note that the 2D failed this one")
            else:
                print("Note that the 3D failed this one")

            number_of_iterations = 0
            trajectories = np.empty((0,3))
            linear_velocities = np.array([])
            current_traj_idx = current_traj_idx + 1
            continue
            
        # Remove the current position
        trajectories = trajectories[1:]
        linear_velocities = linear_velocities[1:]
        number_of_iterations -= 1

        # Compute the cost of the path (select the relevant critics in critics_warp.py)
        cost = compute_cost(trajectories, linear_velocities, robot_x_end, robot_y_end, goal, 
                            goal_orientation, v_max_linear, number_of_iterations, half_width, 
                            resolution, grid_size, costmap_wp, horizon)
        
        if cost > 99000000:
            print("Collision")
            number_of_iterations = 0
            trajectories = np.empty((0,3))
            linear_velocities = np.array([])
            current_traj_idx = current_traj_idx + 1
            continue

        # Compute the length of the path
        # cost = compute_length(trajectories)

        # Reset the parameters
        number_of_iterations = 0
        trajectories = np.empty((0,3))
        linear_velocities = np.array([])
        
        # Print the cost and add it ot the list of costs
        if row['Step'] % 2 == 0:
            # print(f"Cost for the trajectory {(current_traj_idx+1)/2} in 2D is:\n")
            costs_2d.append(cost)
        else:
            # print(f"Cost for the trajectory {current_traj_idx/2} in 3D is: \n")
            costs_3d.append(cost)
        # print(cost)
        current_traj_idx = current_traj_idx + 1

    # Reconstruct the data corresponding to one path
    trajectories = np.vstack((trajectories, [row['X'], row['Y'], row['Z']]))
    linear_velocities = np.append(linear_velocities,row['Linear_Velocity'])
    number_of_iterations += 1


trajectories = trajectories[1:]
linear_velocities = linear_velocities[1:]

cost = compute_cost(trajectories, linear_velocities, robot_x_end, robot_y_end, goal, 
                 goal_orientation, v_max_linear, number_of_iterations, half_width, 
                 resolution, grid_size, costmap_wp, horizon)
# cost = compute_length(trajectories)


# print(f"Cost for the trajectory {int(current_traj_idx / 2 + 0.25)} in 3D is: \n")
# print(cost)

costs_3d.append(cost)

# Compute the average cost for the 2D and 3D projection
print(sum(costs_2d) / len(costs_2d))
print(sum(costs_3d) / len(costs_3d))


# Plot the results
plt.plot(costs_2d, label="Costs 2D", marker='o')
plt.plot(costs_3d, label="Costs 3D", marker='s')

# Labels and title
plt.xlabel("Index")
plt.ylabel("Values")
plt.title("Plot of Costs in 2D and in 3D")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()



print(" \n and removing the five greatest elements \n")

largest_2d = heapq.nlargest(5, costs_2d)
largest_3d = heapq.nlargest(5, costs_3d)

# Remove them from the lists
costs_2d = [x for x in costs_2d if x not in largest_2d]
costs_3d = [x for x in costs_3d if x not in largest_3d]

# Compute the average cost for the 2D and 3D projection
print(sum(costs_2d) / len(costs_2d))
print(sum(costs_3d) / len(costs_3d))


# Plot the results
plt.plot(costs_2d, label="Costs 2D", marker='o')
plt.plot(costs_3d, label="Costs 3D", marker='s')

# Labels and title
plt.xlabel("Index")
plt.ylabel("Values")
plt.title("Plot of Costs in 2D and in 3D")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


print(" \n and removing the five smallest elements \n")

smallest_2d = heapq.nsmallest(5, costs_2d)
smallest_3d = heapq.nsmallest(5, costs_3d)

# Remove them from the lists
costs_2d = [x for x in costs_2d if x not in smallest_2d]
costs_3d = [x for x in costs_3d if x not in smallest_3d]

# Compute the average cost for the 2D and 3D projection
print(sum(costs_2d) / len(costs_2d))
print(sum(costs_3d) / len(costs_3d))


# Plot the results
plt.plot(costs_2d, label="Costs 2D", marker='o')
plt.plot(costs_3d, label="Costs 3D", marker='s')

# Labels and title
plt.xlabel("Index")
plt.ylabel("Values")
plt.title("Plot of Costs in 2D and in 3D")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


print(" \n and removing the five smallest elements \n")

smallest_2d = heapq.nsmallest(5, costs_2d)
smallest_3d = heapq.nsmallest(5, costs_3d)

# Remove them from the lists
costs_2d = [x for x in costs_2d if x not in smallest_2d]
costs_3d = [x for x in costs_3d if x not in smallest_3d]

# Compute the average cost for the 2D and 3D projection
print(sum(costs_2d) / len(costs_2d))
print(sum(costs_3d) / len(costs_3d))


# Plot the results
plt.plot(costs_2d, label="Costs 2D", marker='o')
plt.plot(costs_3d, label="Costs 3D", marker='s')

# Labels and title
plt.xlabel("Index")
plt.ylabel("Values")
plt.title("Plot of Costs in 2D and in 3D")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()