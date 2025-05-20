from projection_warp import _generate_velocities_kernel, \
                            _find_corners_heights, \
                            _bilinear_interpolation, \
                            _compute_normal, \
                            _compute_tangent_vector, \
                            _compute_next_waypoint, \
                            _generate_trajectories_kernel

from critics_warp import _evaluate_trajectories_kernel
import warp as wp
import numpy as np
import matplotlib.pyplot as plt


wp.init()

vec = wp.array([1.0, 2.0, 3.0, 4.0, 5.0],dtype=wp.float32)

print(wp.atomic_min(vec, 3, 3.0))


# x = 0.0
# y = 0.0
# goal = wp.vec2f(3.0, 3.5)
# goal_orientation = 1.57

# iterations = 3
# number_of_trajectories = 3
# radius = 0.01

# trajectories_data = [
#     wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(0.2, 0.2, 0.3), wp.vec3f(-0.1, 0.4, 0.2),
#     wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(0.4, 0.4, 0.3), wp.vec3f(0.7, 0.7, 0.2),
#     wp.vec3f(3.0, 3.0, 3.0), wp.vec3f(3.1, 3.1, 1.0), wp.vec3f(1.2, 0.9, 4.0)
# ]

# trajectories_wp = wp.array(trajectories_data, dtype=wp.vec3f)

# obstacles_data = [
#     # wp.vec3f(1.0, 1.0, 0.5)
# ]

# obstacles_wp = wp.array(obstacles_data, dtype=wp.vec3f)

# costs_wp = wp.zeros(number_of_trajectories, dtype=float)

# wp.launch(
#     kernel=_evaluate_trajectories_kernel,
#     dim= number_of_trajectories,
#     inputs=[x, y, goal, goal_orientation, trajectories_wp, iterations, obstacles_wp, radius, costs_wp],
# )

# scores = costs_wp.numpy()
# print("Computed Scores:", scores)

# scores = costs_wp.numpy()

# temperature = 1.0  
# normalized_scores = scores - np.min(scores)
# exponents = np.exp(-normalized_scores / temperature)
# weights = exponents / np.sum(exponents)


