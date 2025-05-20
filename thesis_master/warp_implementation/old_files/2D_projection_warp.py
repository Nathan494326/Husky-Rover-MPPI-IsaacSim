# import warp as wp

# @wp.func
# def _update_orientation(
#     heading_vector: wp.vec3f,
#     angular_velocity: float,
#     normal_vector: wp.vec3f,
#     dt: float,
# ) -> wp.vec3f:

#     heading_vector = heading_vector / wp.sqrt(wp.dot(heading_vector, heading_vector))

#     angle = angular_velocity * dt
#     cos_theta = wp.cos(angle)
#     sin_theta = wp.sin(angle)

#     # Rodrigues' rotation formula
#     rotated_heading_vector = (
#         heading_vector * cos_theta
#         + wp.cross(normal_vector, heading_vector) * sin_theta
#         + normal_vector * wp.dot(normal_vector, heading_vector) * (1.0 - cos_theta)
#     )

#     rotated_heading_vector = rotated_heading_vector / wp.sqrt(wp.dot(rotated_heading_vector, rotated_heading_vector))

#     return rotated_heading_vector

# # Define the kernel to call the function
# @wp.kernel
# def test_update_orientation(
#     heading_vector: wp.array(dtype=wp.vec3f),
#     angular_velocity: float,
#     normal_vector: wp.vec3f,
#     dt: float,
#     result: wp.array(dtype=wp.vec3f),
# ):
#     tid = wp.tid()
#     result[tid] = _update_orientation(heading_vector[tid], angular_velocity, normal_vector, dt)


# # Initialize Warp
# wp.init()

# # Define input values
# heading_vectors = [wp.vec3f(1.0, 1.0, 1.0)]  # Example heading vector along x-axis
# angular_velocity = wp.pi/4.0  # Example angular velocity (radians per second)
# normal_vector = wp.vec3f(0.0, 1.0, 0.0)  # Example normal vector along z-axis (rotation axis)
# dt = 1.0  # Time step

# # Allocate memory for input and output
# num_vectors = len(heading_vectors)
# d_heading_vector = wp.array(heading_vectors, dtype=wp.vec3f, device="cpu")
# d_result = wp.zeros(num_vectors, dtype=wp.vec3f, device="cpu")

# # Launch kernel
# wp.launch(
#     kernel=test_update_orientation,
#     dim=num_vectors,
#     inputs=[d_heading_vector, angular_velocity, normal_vector, dt, d_result],
#     device="cpu",
# )

# # Fetch results
# updated_heading = d_result.numpy()

# # Print the result
# print("Updated Heading Vector:", updated_heading[0])

import numpy as np

# Define the vector
v = np.array([-5.49386144e-02, 9.07762349e-01, -4.15871710e-01])

# Compute the Euclidean norm
norm = np.linalg.norm(v)

# Print the result
print("Norm of the vector:", norm)