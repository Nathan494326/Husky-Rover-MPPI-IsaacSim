import warp as wp


@wp.func
def _goal_angle_critic(
    x: float,
    y: float,
    goal: wp.vec2f,
    goal_orientation: float,
    trajectory: wp.array(dtype=wp.vec3f),
    start_idx: float,
    iterations: float,
) -> float:
    """
    This is for when the robot is near the goal. 
    It should increase the importance of trajectories getting in the right orientation.

    Args:
        x (float): Current x position.
        y (float): Current y position.
        goal (wp.vec2f): Target goal position.
        goal_orientation (float): Target goal orientation.
        trajectory (wp.array(dtype=wp.vec3f)): Trajectory of the robot.
        start_idx (float): Index of the first waypoint of the tid'th trajectory.
        iterations (float): Number of particules in one trajectory.

    Returns:
        float: Computed score based on difference between current orientation and goal orientation.
    """
    
    dist_to_goal = wp.sqrt((x - goal[0])*(x - goal[0]) + (y - goal[1])*(y - goal[1]))

    if dist_to_goal < 0.5:
        penultimate_point = trajectory[wp.int(start_idx + iterations - 2.0)] 
        last_point = trajectory[wp.int(start_idx + iterations - 1.0)]  
        
        orientation_diff = wp.abs(wp.atan((last_point[1] - penultimate_point[1])/(last_point[0] - penultimate_point[0])) - goal_orientation)
        return orientation_diff

    return 0.0  


@wp.func
def _path_orientation_critic(
    x: float,
    y: float,
    goal: wp.vec2f,
    trajectory: wp.array(dtype=wp.vec3f),
    start_idx: float,
    iterations: float,
) -> float:
    """
    This critic encourages making progress along the path.
    It drives the robot towards the goal.

    Args:
        x (float): Current x position.
        y (float): Current y position.
        goal (wp.vec2f): Target goal position.
        trajectory (wp.array(dtype=wp.vec3f)): Trajectory of the robot.
        start_idx (float): Index of the first waypoint of the tid'th trajectory.
        iterations (float): Number of particules in one trajectory.

    Returns:
        float: Computed score based on the progress made towards the goal.
    """

    x_diff = goal[0] - x
    y_diff = goal[1] - y

    penultimate_point = trajectory[wp.int(start_idx + iterations - 2.0)] 
    last_point = trajectory[wp.int(start_idx + iterations - 1.0)] 

    x_diff2 = last_point[0] - penultimate_point[0]
    y_diff2 = last_point[1] - penultimate_point[1]

    scalar_product = x_diff * x_diff2 + y_diff * y_diff2

    if x_diff * x_diff2 + y_diff * y_diff2 <= 0:
        return -scalar_product/(wp.abs(x_diff) + wp.abs(y_diff))

    return 0.0


@wp.func
def _path_follow_critic(
    x: float,
    y: float,
    goal: wp.vec2f,
    trajectory: wp.array(dtype=wp.vec3f),
    start_idx: float,
    iterations: float,
    horizon:float,
) -> float:
    """
    This critic encourages making progress along the path.
    It drives the robot towards the goal.

    Args:
        x (float): Current x position.
        y (float): Current y position.
        goal (wp.vec2f): Target goal position.
        trajectory (wp.array(dtype=wp.vec3f)): Trajectory of the robot.
        start_idx (float): Index of the first waypoint of the tid'th trajectory.
        iterations (float): Number of particules in one trajectory.

    Returns:
        float: Computed cost based on the progress made towards the goal.
    """

    epsilon = 1e-6  

    x_diff = goal[0] - x
    y_diff = goal[1] - y
    dist_to_goal = wp.sqrt((x_diff)*(x_diff) + (y_diff)*(y_diff))
    last_point = trajectory[wp.int(start_idx + iterations - 1.0)] 
    cost = float(0.0)

    if dist_to_goal > horizon:
        intermediate_goal_x = x + x_diff*horizon/(dist_to_goal+epsilon)
        intermediate_goal_y = y + y_diff*horizon/(dist_to_goal+epsilon)
        cost = wp.pow((last_point[0] - intermediate_goal_x)*(last_point[0] - intermediate_goal_x) + (last_point[1] - intermediate_goal_y)*(last_point[1] - intermediate_goal_y), 1.0)
        return cost * (1.0 + 2.0*horizon/dist_to_goal)
    
    for i in range(wp.int(start_idx), wp.int(start_idx + iterations - 1.0)):   
        cost += 10.0 * (wp.abs(trajectory[i][0] - goal[0]) + wp.abs(trajectory[i][1] - goal[1]))
    return cost


@wp.func
def _avoid_slope(
    trajectory: wp.array(dtype=wp.vec3f),
    start_idx: float,
    iterations: float,
) -> float:
    """
    This critic avoids going on slopes.
    The steeper the slope, the less importance is given to the corresponding trajectory.

    Args:
        trajectory (wp.array(dtype=wp.vec3f)): 1D array containing all trajectories.
        start_idx (int): Index of the first waypoint of the tid'th trajectory.
        iterations (int): Number of waypoints in one trajectory.
    Returns:
        float: Computed score based on the flatness of the trajectory.
    """
    
    total_slope = float(0.0)
    epsilon = 1e-6  
    
    for i in range(wp.int(start_idx), wp.int(start_idx + iterations - 1.0 - 2.0), 2):  
        # Compute the difference in z-coordinates between consecutive waypoints
        current_point = trajectory[i + 2]
        previous_point = trajectory[i]

        dz = current_point[2] - previous_point[2]
        d = wp.sqrt((current_point[0] - previous_point[0]) * (current_point[0] - previous_point[0]) + 
                                (current_point[1] - previous_point[1]) * (current_point[1] - previous_point[1]))

        ratio = wp.abs(dz / (d + epsilon))
        # angle = wp.atan(ratio)

        # Accumulate absolute slope value (to penalize both up and down slopes)
        total_slope += (1.0 + 5.0*ratio) * (1.0 + 5.0*ratio) # * (1.0 + 2.0*ratio)
    
    return total_slope

@wp.func
def _avoid_slope_wheels(
    lw_h: wp.array(dtype=wp.vec3f),
    rw_h: wp.array(dtype=wp.vec3f),
    start_idx: float,
    iterations: float,
) -> float:
    """
    This critic avoids going on slopes.
    The steeper the slope, the less importance is given to the corresponding trajectory.

    Args:
        trajectory (wp.array(dtype=wp.vec3f)): 1D array containing all trajectories.
        start_idx (int): Index of the first waypoint of the tid'th trajectory.
        iterations (int): Number of waypoints in one trajectory.
    Returns:
        float: Computed score based on the flatness of the trajectory.
    """
    
    total_slope = float(0.0)
    epsilon = 1e-6  
    
    for i in range(wp.int(start_idx), wp.int(start_idx + iterations - 1.0 - 2.0), 2):  
        # Compute the difference in z-coordinates between consecutive waypoints
        current_point_l = lw_h[i + 2]
        previous_point_l = lw_h[i]

        current_point_r = rw_h[i + 2]
        previous_point_r = rw_h[i]

        dz_l = current_point_l[2] - previous_point_l[2]
        d_l = wp.sqrt((current_point_l[0] - previous_point_l[0]) * (current_point_l[0] - previous_point_l[0]) + 
                                (current_point_l[1] - previous_point_l[1]) * (current_point_l[1] - previous_point_l[1]))

        dz_r = current_point_r[2] - previous_point_r[2]
        d_r = wp.sqrt((current_point_r[0] - previous_point_r[0]) * (current_point_r[0] - previous_point_r[0]) + 
                                (current_point_r[1] - previous_point_r[1]) * (current_point_r[1] - previous_point_r[1]))

        ratio_l = wp.abs(dz_l / (d_l + epsilon))
        ratio_r = wp.abs(dz_r / (d_r + epsilon))

        left_slope = (1.0 + 5.0*ratio_l) * (1.0 + 5.0*ratio_l)
        right_slope =  (1.0 + 5.0*ratio_r) * (1.0 + 5.0*ratio_r) 

        # Accumulate absolute slope value (to penalize both up and down slopes)
        if left_slope > right_slope:
            total_slope += left_slope
        else: 
            total_slope += right_slope
    
    return total_slope

@wp.func
def _avoid_obstacle(
    trajectory: wp.array(dtype=wp.vec3f),
    start_idx: float,
    iterations: float,
    half_width: float,
    resolution_costmap: float,
    costmap_size: int,
    costmap: wp.array(dtype=float),
) -> float:
    """
    This critic avoids collisions with obstacles

    Args:
        trajectory (wp.array(dtype=wp.vec3f)): 1D array containing all trajectories.
        start_idx (float): Index of the first waypoint of the tid'th trajectory.
        iterations (float): Number of waypoints in one trajectory.
        obstacles (wp.array(dtype=wp.vec3f)): 1D array containing vectors corresponding to the obstacles position and radius.
        radius (float): radius of the robot.
    Returns:
        float: Computed score based on the flatness of the trajectory.
    """

    obs_cost = wp.float32(0.0)
    for i in range(wp.int(start_idx), wp.int((start_idx + iterations))):  
        idx_x = (trajectory[i][0] + half_width) / resolution_costmap
        idx_y = (- trajectory[i][1] + half_width) / resolution_costmap
        
        costmap_cost = costmap[wp.int(idx_x) + costmap_size*wp.int(idx_y)]
        
        # normal run
        if costmap_cost > 0.99:
            obs_cost += 100000.0
        obs_cost += costmap_cost

        # for stats
        # if costmap_cost > 0.99:
        #     return 100000000.0 # * (iterations - (wp.float(i)-start_idx+1.0)) / (wp.float(i)-start_idx+1.0)
        # elif 0.65 < costmap_cost < 0.75:
        #     obs_cost += 1.0
        # elif 0.75 < costmap_cost < 0.85:
        #     obs_cost += 100.0
        # elif 0.85 < costmap_cost < 0.95:
        #     obs_cost += 10000.0
        # elif costmap_cost > 0.95:
        #     obs_cost += 1000000.0

    return obs_cost

@wp.func
def _maximise_speed(
    x: float,
    y: float,
    goal: wp.vec2f,
    linear_velocities: wp.array(dtype=float),
    target_speed: float,
    start_idx: float,
    iterations: float,
    horizon: float,
) -> float:
    
    x_diff = goal[0] - x
    y_diff = goal[1] - y
    dist_to_goal = wp.sqrt((x_diff)*(x_diff) + (y_diff)*(y_diff))

    if dist_to_goal < 2.0:
        return 0.0

    speed_diff = wp.float32(0.0)
    
    # for i in range(wp.int(start_idx), wp.int(start_idx + 10.0)):  
    #     speed_diff += 5.0 * (target_speed - linear_velocities[i])

    # for i in range(wp.int(start_idx + 10.0), wp.int(start_idx + iterations)):  
    #     speed_diff += (target_speed - linear_velocities[i])

    for i in range(wp.int(start_idx), wp.int(start_idx + iterations)):  
        speed_diff += (target_speed - linear_velocities[i]) / (linear_velocities[i] + 0.0001)

    
    return speed_diff

@wp.kernel
def _evaluate_trajectories_kernel(
    x: float,
    y: float,
    goal: wp.vec2f,
    goal_orientation: float,
    trajectories: wp.array(dtype=wp.vec3f),
    lw: wp.array(dtype=wp.vec3f),
    rw: wp.array(dtype=wp.vec3f),
    linear_velocities: wp.array(dtype=float),
    target_speed: float,
    number_of_trajectories: float,
    iterations: float,
    half_width: float,
    resolution_costmap: float,
    costmap_size: int,
    costmap: wp.array(dtype=float),
    horizon: float,
    costs: wp.array(dtype=float),
):
    tid = wp.tid()
    
    # costs[tid] += _path_orientation_critic(x, y, goal, trajectories, wp.float(tid)*iterations, iterations)
    costs[tid] += 100.5*_path_follow_critic(x, y, goal, trajectories, wp.float(tid)*iterations, iterations, horizon)
    # costs[tid] += 50.5*_avoid_slope(trajectories, wp.float(tid)*iterations, iterations) # 35.5
    costs[tid] += 50.5*_avoid_slope_wheels(lw, rw, wp.float(tid)*iterations, iterations) # 35.5
    costs[tid] += 0.5*_maximise_speed(x, y, goal, linear_velocities, target_speed, wp.float(tid)*iterations, iterations, horizon)
    costs[tid] += 25.0*_avoid_obstacle(trajectories, wp.float(tid)*iterations, iterations, half_width, resolution_costmap, costmap_size, costmap)

    # costs[tid] += 1.0*_avoid_slope(trajectories, number_of_trajectories, wp.float(tid)*iterations, iterations)
    # costs[tid] += 1.0*_maximise_speed(x, y, goal, linear_velocities, target_speed, wp.float(tid)*iterations, iterations, horizon)
    # costs[tid] += 1.0*_avoid_obstacle(trajectories, wp.float(tid)*iterations, iterations, half_width, resolution, grid_size, costmap)




@wp.kernel
def _compute_weights(costs: wp.array(dtype=wp.float32),
                     min_cost: wp.array(dtype=float),
                     weights: wp.array(dtype=wp.float32),
                     temperature: float):

    tid = wp.tid()
    wp.atomic_min(min_cost, 0, costs[tid])
    normalised_cost = costs[tid] - min_cost[0]
    weights[tid] = wp.exp(-normalised_cost/temperature)


@wp.kernel
def _compute_sum(weights: wp.array(dtype=wp.float32),
                 weights_sum: wp.array(dtype=wp.float32)):
    
    
    tid = wp.tid() 

    wp.atomic_add(weights_sum, 0, weights[tid])





@wp.kernel
def _compute_weighted_sum(weights: wp.array(dtype=wp.float32),
                         iterations: float,
                         array1: wp.array(dtype=wp.float32),
                         array2: wp.array(dtype=wp.float32),
                         weights_sum: wp.array(dtype=wp.float32),
                         out1: wp.array(dtype=wp.float32),
                         out2: wp.array(dtype=wp.float32),):

    tid = wp.tid()

    for k in range(wp.int(iterations)):
        out1[k] += weights[tid]*array1[tid*wp.int(iterations) + k]/weights_sum[0]
        out2[k] += weights[tid]*array2[tid*wp.int(iterations) + k]/weights_sum[0]







