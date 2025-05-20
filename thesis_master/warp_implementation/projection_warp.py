import warp as wp


"""
Computation of the corners around a particular point
"""

@wp.func
def _get_corners_heights(
    x: float,
    y: float,
    x_min: float,
    y_min: float,
    q: wp.mat22f,
    grid_size: int,
    resolution: float,
    Z: wp.array(dtype=float),
) -> wp.mat22f:
    """
    Find the four corners of the grid cell containing the point (x, y), and compute their corresponding heights.

    Args:
        x (wp.float32): X-coordinate of the point.
        y (wp.float32): Y-coordinate of the point.
        q (wp.mat22f): Matrix to store corner heights.
        resolution (wp.float32): Grid resolution.
        X (wp.array): 1D array representing the x-coordinates of the surface grid.
        Y (wp.array): 1D array representing the y-coordinates of the surface grid.
        Z (wp.array): 2D array representing the z-heights of the surface at each (X, Y) point.

    Returns:
        wp.mat22f: Updated matrix with the heights of the four corners.
    """

    # Find indices in the grid
    #i = wp.int((x - x_min) / resolution)
    #j = wp.int((y - y_min) / resolution)

    i = wp.int((x - x_min) / resolution)
    j = -wp.int((y + y_min) / resolution)

    # Fetch heights of the four corners
    q[0, 0] = Z[j*grid_size + i]       # Bottom-left corner
    q[0, 1] = Z[j*grid_size + i + 1]   # Bottom-right corner
    q[1, 0] = Z[(j + 1)*grid_size + i]   # Top-left corner
    q[1, 1] = Z[(j + 1)*grid_size + i + 1]  # Top-right corner

    return q

@wp.kernel
def _find_corners_heights(
    x: wp.array(dtype=float),
    y: wp.array(dtype=float),
    x_min: float,
    y_min: float,
    q: wp.array(dtype=wp.mat22f),
    grid_size: int,
    resolution: float,
    Z: wp.array(dtype=float),
):

    tid = wp.tid()
    q[tid] = _get_corners_heights(x[tid], y[tid], x_min, y_min, q[tid], grid_size, resolution, Z)


"""
Computation of the approximated height of a particular point
"""

@wp.func
def _bilinear_interpolator(
    x: float, 
    y: float, 
    q: wp.mat22f, 
    resolution: float
) -> float:
    """
    Perform bilinear interpolation for a point in Warp.

    Args:
        x (wp.float32): x-coordinate.
        y (wp.float32): y-coordinate.
        q (wp.mat22f): 2x2 array containing the z-values of the grid cell.
        resolution (wp.float32): Grid resolution.

    Returns:
        wp.float32: Interpolated height value.
    """
    # Normalize the coordinates
    x_normalized = x / resolution
    y_normalized = y / resolution

    # Get the fractional parts of the normalized coordinates
    x2 = x_normalized - wp.trunc(x_normalized)
    y2 = y_normalized - wp.trunc(y_normalized)

    # Perform bilinear interpolation
    result = (1.0 - x2) * (1.0 - y2) * q[0, 0] + x2 * (1.0 - y2) * q[1, 0] + (1.0 - x2) * y2 * q[0, 1] + x2 * y2 * q[1, 1]

    return result


@wp.kernel
def _bilinear_interpolation(
    x: wp.array(dtype=float),
    y: wp.array(dtype=float),
    q: wp.array(dtype=wp.mat22f),
    resolution: float,
    out: wp.array(dtype=float),
):
    """
    Bilinear interpolation of all the points in the array.

    Args:
        x (wp.array(dtype=float)): x coordinates.
        y (wp.array(dtype=float)): y coordinates.
        q (wp.array(dtype=wp.mat22f)): 2x2 matrices.
        out (wp.array(dtype=float)): output.
    """

    tid = wp.tid()
    out[tid] = _bilinear_interpolator(x[tid], y[tid], q[tid], resolution)


"""
Computation of the normal vector at the position of a particular point
"""

@wp.func
def _normal_on_grid(q: wp.mat22f, resolution: float) -> wp.vec3f:
    """
    Compute the normal of a quad on a regular grid.

    Args:
        q (wp.mat22f): 2x2 matrix containing the z-values of the grid cell.
        grid_size (wp.float32): The grid size.

    Returns:
        wp.vec3f: Normal vector (3D).
    """
    # Compute the components of the normal vector
    vec_x = -resolution / 2.0 * (q[0, 1] - q[0, 0] - q[1, 0] + q[1, 1])
    vec_y = -resolution / 2.0 * (q[1, 0] - q[0, 0] - q[0, 1] + q[1, 1])
    vec_z = resolution * resolution

    # Create the vector
    vec = wp.vec3f(vec_x, vec_y, vec_z)

    # Normalize the vector
    norm = wp.sqrt(vec_x * vec_x + vec_y * vec_y + vec_z * vec_z)
    return vec / norm

@wp.kernel
def _compute_normal(
    q: wp.array(dtype=wp.mat22f),
    resolution: float,
    normal: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()
    normal[tid] = _normal_on_grid(q[tid], resolution)


"""
Computation of the heading tangent vector
"""


@wp.func
def _get_heading_tangent_vector(normal: wp.vec3f, previous_heading_vector: wp.vec3f) -> wp.vec3f:
    """
    Project the previous heading vector onto the new plane defined by the given normal vector.

    Args:
        normal (wp.vec3f): The normal vector of the new plane (surface).
        previous_heading_vector (wp.vec3f): The previous heading vector of the robot.

    Returns:
        wp.vec3f: The new heading tangent vector after projection onto the plane.
    """
    # Compute the dot product of the heading vector and the normal
    dot_product = wp.dot(previous_heading_vector, normal)

    # Compute the projection of the heading vector onto the plane
    projection = previous_heading_vector - dot_product * normal

    # Normalize the projection to get the tangent vector
    norm = wp.sqrt(wp.dot(projection, projection))
    tangent_vector = projection / norm

    return tangent_vector

@wp.kernel
def _compute_tangent_vector(
    previous_heading_vectors: wp.array(dtype=wp.vec3f),
    normal: wp.array(dtype=wp.vec3f),
    heading_vectors: wp.array(dtype=wp.vec3f)
):
    tid = wp.tid()
    heading_vectors[tid] = _get_heading_tangent_vector(normal[tid], previous_heading_vectors[tid])



"""
Update the position and orientation of the robot
"""

@wp.func
def _update_position(
    x: float,
    y: float,
    heading_vector: wp.vec3f,
    linear_velocity: float,
    dt: float,
) -> wp.vec2f:

    heading_vector = heading_vector / wp.sqrt(wp.dot(heading_vector, heading_vector))

    displacement = heading_vector * linear_velocity * dt
    new_x = x + displacement[0]
    new_y = y + displacement[1]


    return wp.vec2f(new_x, new_y)

@wp.func
def _update_orientation(
    heading_vector: wp.vec3f,
    angular_velocity: float,
    normal_vector: wp.vec3f,
    dt: float,
) -> wp.vec3f:

    heading_vector = heading_vector / wp.sqrt(wp.dot(heading_vector, heading_vector))

    angle = angular_velocity * dt
    cos_theta = wp.cos(angle)
    sin_theta = wp.sin(angle)

    # Rodrigues' rotation formula
    rotated_heading_vector = (
        heading_vector * cos_theta
        + wp.cross(normal_vector, heading_vector) * sin_theta
        + normal_vector * wp.dot(normal_vector, heading_vector) * (1.0 - cos_theta)
    )

    rotated_heading_vector = rotated_heading_vector / wp.sqrt(wp.dot(rotated_heading_vector, rotated_heading_vector))

    return rotated_heading_vector


@wp.func
def _update_orientation_2D(
    heading_vector: wp.vec3f,
    angular_velocity: float,
    dt: float,
) -> wp.vec3f:

    # Compute rotation angle
    theta = angular_velocity * dt

    # Compute new heading using 2D rotation
    cos_theta = wp.cos(theta)
    sin_theta = wp.sin(theta)

    new_x = cos_theta * heading_vector[0] - sin_theta * heading_vector[1]
    new_y = sin_theta * heading_vector[0] + cos_theta * heading_vector[1]

    # Normalize the new heading vector
    norm = wp.sqrt(new_x * new_x + new_y * new_y)
    if norm > 0.0:
        new_x /= norm
        new_y /= norm

    # Return updated heading vector (keeping z = 0)
    return wp.vec3f(new_x, new_y, 0.0)




"""
Compute multiple trajectories
"""

@wp.kernel
def _generate_trajectories_kernel(
    position: wp.array(dtype=wp.vec2f),
    x_min: float,
    y_min:float,
    grid_size:int,
    q: wp.array(dtype=wp.mat22f),
    resolution: float,
    Z: wp.array(dtype=float),
    height: wp.array(dtype=float),
    normal: wp.array(dtype=wp.vec3f),
    heading_vectors: wp.array(dtype=wp.vec3f),
    previous_heading_vector: wp.vec3f,
    iterations: int,
    linear_velocities: wp.array(dtype=float),
    angular_velocities: wp.array(dtype=float),
    dt: float,
    trajectory: wp.array(dtype=wp.vec3f),
    lw: wp.array(dtype=wp.vec3f),
    rw: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()  
    q[tid] = _get_corners_heights(position[tid][0], position[tid][1], x_min, y_min, q[tid], grid_size, resolution, Z)
    height[tid] = _bilinear_interpolator(position[tid][0], position[tid][1], q[tid], resolution)
    normal[tid] = _normal_on_grid(q[tid], resolution)
    # heading_vectors[tid] = _get_heading_tangent_vector(normal[tid], previous_heading_vector)
    previous = _get_heading_tangent_vector(normal[tid], previous_heading_vector)
    
    for k in range(iterations):
        # Compute the next waypoint position of each trajectory
        position[tid] = _update_position(position[tid][0], position[tid][1], previous, linear_velocities[tid * iterations + k], dt)
        
        # Compute the height of the k-th waypoint of each trajectory
        q[tid] = _get_corners_heights(position[tid][0], position[tid][1], x_min, y_min, q[tid], grid_size, resolution, Z)
        height[tid] = _bilinear_interpolator(position[tid][0], position[tid][1], q[tid], resolution)
        
        # Compute the normal plan on which the k-th waypoint of each trajectory relies
        normal[tid] = _normal_on_grid(q[tid], resolution)
        
        previous = _get_heading_tangent_vector(normal[tid], previous)
        
        # Compute the next waypoint orientation of each trajectory
        current_hv = _update_orientation(previous, angular_velocities[tid * iterations + k], normal[tid], dt)
        heading_vectors[tid*iterations+k] = current_hv

        # Store the 3D position of the new waypoint of each trajectory
        trajectory[tid*iterations+k] = wp.vec3f(position[tid][0], position[tid][1], height[tid])

        # Get the heights of the left and right wheels
        offset = 0.2
        right = offset * wp.cross(normal[tid], current_hv)

        x_wheel = position[tid][0] + right[0]
        y_wheel = position[tid][1] + right[1]
        i_idx = wp.int((x_wheel - x_min) / resolution)
        j_idx = -wp.int((y_wheel + y_min) / resolution)
        lw_h = Z[j_idx*grid_size + i_idx] 
        lw[tid*iterations+k] = wp.vec3f(x_wheel, y_wheel, lw_h)

        x_wheel = position[tid][0] - right[0]
        y_wheel = position[tid][1] - right[1]
        i_idx = wp.int((x_wheel - x_min) / resolution)
        j_idx = -wp.int((y_wheel + y_min) / resolution)
        rw_h = Z[j_idx*grid_size + i_idx] 
        rw[tid*iterations+k] = wp.vec3f(x_wheel, y_wheel, rw_h)

        previous = heading_vectors[tid*iterations+k]


@wp.kernel
def _generate_trajectories_2D_kernel(
    position: wp.array(dtype=wp.vec2f),
    x_min: float,
    y_min:float,
    grid_size:int,
    q: wp.array(dtype=wp.mat22f),
    resolution: float,
    Z: wp.array(dtype=float),
    height: wp.array(dtype=float),
    heading_vectors: wp.array(dtype=wp.vec3f),
    previous_heading_vector: wp.vec3f,
    iterations: int,
    linear_velocities: wp.array(dtype=float),
    angular_velocities: wp.array(dtype=float),
    dt: float,
    trajectory: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()  
    previous = previous_heading_vector
    for k in range(iterations):
        position[tid] = _update_position(position[tid][0], position[tid][1], previous, linear_velocities[tid * iterations + k], dt)
        heading_vectors[tid*iterations+k] = _update_orientation_2D(previous, angular_velocities[tid * iterations + k], dt)

        q[tid] = _get_corners_heights(position[tid][0], position[tid][1], x_min, y_min, q[tid], grid_size, resolution, Z)
        height[tid] = _bilinear_interpolator(position[tid][0], position[tid][1], q[tid], resolution)
        
        # Store the 3D position of the new waypoint of each trajectory
        trajectory[tid*iterations+k] = wp.vec3f(position[tid][0], position[tid][1], height[tid])
        previous = heading_vectors[tid*iterations+k]

