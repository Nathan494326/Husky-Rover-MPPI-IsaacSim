import warp as wp



"""
Generation of random sequences of linear and angular velocities
"""


@wp.kernel
def _generate_velocities_kernel(
    iterations: int,
    seed: int,
    previous_linear_velocities: wp.array(dtype=float), 
    previous_angular_velocities: wp.array(dtype=float),
    std_dev_linear: float,
    min_linear_velocity: float,
    max_linear_velocity: float,
    std_dev_angular: float,
    min_angular_velocity: float,
    max_angular_velocity: float,
    linear_velocities: wp.array(dtype=float), 
    angular_velocities: wp.array(dtype=float)  
):
    tid = wp.tid() 
    if tid % iterations != (iterations-1):
        linear_velocities[tid] = wp.clamp(
                previous_linear_velocities[(tid%iterations) + 1] + std_dev_linear * wp.randn(wp.uint32(seed+tid)),
                min_linear_velocity,
                max_linear_velocity
            )
        angular_velocities[tid] = wp.clamp(
                previous_angular_velocities[(tid%iterations) + 1] + std_dev_angular * wp.randn(wp.uint32(seed+tid+iterations)),
                min_angular_velocity,
                max_angular_velocity
            )
        
    else:
        linear_velocities[tid] = wp.clamp(
                previous_linear_velocities[tid%iterations] + std_dev_linear * wp.randn(wp.uint32(seed+tid)),
                min_linear_velocity,
                max_linear_velocity
            )
        angular_velocities[tid] = wp.clamp(
                previous_angular_velocities[tid%iterations] + std_dev_angular * wp.randn(wp.uint32(seed+tid+iterations)),
                min_angular_velocity,
                max_angular_velocity
            )





@wp.kernel
def _generate_inputs_kernel(
    iterations: int,
    seed: int,
    previous_u1: wp.array(dtype=float), 
    previous_u2: wp.array(dtype=float),
    std_dev_u1: float,
    min_u1: float,
    max_u1: float,
    std_dev_u2: float,
    min_u2: float,
    max_u2: float,
    u1: wp.array(dtype=float), 
    u2: wp.array(dtype=float)  
):
    tid = wp.tid() 

    if tid % iterations != (iterations-1):
        u1[tid] = wp.clamp(
                previous_u1[(tid%iterations) + 1] + std_dev_u1 * wp.randn(wp.uint32(seed+tid+iterations)),
                min_u1,
                max_u1
            )
        u2[tid] = wp.clamp(
                previous_u2[(tid%iterations) + 1] + std_dev_u2 * wp.randn(wp.uint32(seed+tid+2*iterations)),
                min_u2,
                max_u2
            )
    else:
        u1[tid] = wp.clamp(
                previous_u1[tid%iterations] + std_dev_u1 * wp.randn(wp.uint32(seed+tid+3*iterations)),
                min_u1,
                max_u1
            )
        u2[tid] = wp.clamp(
                previous_u2[tid%iterations] + std_dev_u2 * wp.randn(wp.uint32(seed+tid+4*iterations)),
                min_u2,
                max_u2
            )



@wp.kernel
def _convert_inputs_to_velocities(
    iterations: int,
    r_wheels:float,

    current_left_velocity: float, 
    current_right_velocity: float, 

    v_min_linear: float,
    v_max_linear: float,
    v_min_angular: float,
    v_max_angular: float,

    u1: wp.array(dtype=float), 
    u2: wp.array(dtype=float), 

    linear_velocities: wp.array(dtype=float), 
    angular_velocities: wp.array(dtype=float),

    k: float,
    a: float
):
    tid = wp.tid() 

    left_wheel_velocity = current_left_velocity * a + u1[tid * iterations] * k * (1.0 - a)
    right_wheel_velocity = current_right_velocity * a + u2[tid * iterations] * k * (1.0 - a)

    linear_velocities[tid * iterations] = wp.clamp((left_wheel_velocity + right_wheel_velocity)/2.0, v_min_linear, v_max_linear)
    angular_velocities[tid * iterations] = wp.clamp((-left_wheel_velocity + right_wheel_velocity)/r_wheels, v_min_angular, v_max_angular)
    

    previous_left = left_wheel_velocity
    previous_right = right_wheel_velocity

    for i in range(iterations-1):
        left_wheel_velocity = previous_left * a + u1[tid * iterations + i + 1] * k * (1.0 - a)
        right_wheel_velocity = previous_right * a + u2[tid * iterations + i + 1] * k * (1.0 - a)
        
        linear_velocities[tid * iterations + i + 1] = wp.clamp((left_wheel_velocity + right_wheel_velocity)/2.0, v_min_linear, v_max_linear)
        angular_velocities[tid * iterations + i + 1] = wp.clamp((-left_wheel_velocity + right_wheel_velocity)/r_wheels, v_min_angular, v_max_angular)

        previous_left = left_wheel_velocity
        previous_right = right_wheel_velocity
        


