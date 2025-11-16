import omni
import copy
import math
import os

import numpy as np
from omni.isaac.kit import SimulationApp
from scipy.spatial.transform import Rotation as SSTR

seed_rock = 67
seed_terrain = 57


def EMAquat(q1, q2, alpha):
    dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]
    if dot < 0:
        alpha2 = -alpha
    else:
        alpha2 = copy.copy(alpha)
    x = q1[0] * (1 - alpha2) + q2[0] * alpha2
    y = q1[1] * (1 - alpha2) + q2[1] * alpha2
    z = q1[2] * (1 - alpha2) + q2[2] * alpha2
    w = q1[3] * (1 - alpha2) + q2[3] * alpha2
    s = math.sqrt(x * x + y * y + z * z + w * w)
    return x / s, y / s, z / s, w / s


cfg = {
    "renderer": "RayTracedLighting",
    "headless": False,
    "samples_per_pixel_per_frame": 32,
    "max_bounces": 6,
    "max_specular_transmission_bounces": 6,
    "max_volume_bounces": 4,
    "subdiv_refinement_level": 0,
}
# cfg = {
#    "headless": True,
# }


simulation_app = SimulationApp(cfg)


RSCfg_1_D = {
   "block_size": 25,
   "rock_dist_cfg": {
       "position_distribution": {
           "name": "thomas_point_process",
           "parent_density": 0.04,
           "child_density": 150,
           "sigma": 2.0,
       },
       "scale_distribution": {
           "name": "uniform",
           "min": 0.02,
           "max": 0.05,
       },
   },
}

RSCfg_2_D = {
   "block_size": 25,
   "rock_dist_cfg": {
       "position_distribution": {
           "name": "thomas_point_process",
           "parent_density": 0.01,
           "child_density": 10,
           "sigma": 4.5,
       },
       "scale_distribution": {
           "name": "uniform",
           "min": 0.05,
           "max": 0.2,
       },
   },
}

RSCfg_3_D = {
   "block_size": 25,
   "rock_dist_cfg": {
       "position_distribution": {
           "name": "thomas_point_process",
           "parent_density": 0.003, # 0.0025,
           "child_density": 4, # 5
           "sigma": 4.5, # 5.0
       },
       "scale_distribution": {
           "name": "uniform",
           "min": 0.5,
           "max": 1.0,
       },
   },
}

RSCfg_4_D = {
   "block_size": 25,
   "rock_dist_cfg": {
       "position_distribution": {
           "name": "thomas_point_process",
           "parent_density": 0.004, # 0.001,
           "child_density": 3, # 3
           "sigma": 8.5, # 4.0
       },
       "scale_distribution": {
           "name": "uniform",
           "min": 1.0,
           "max": 2.0,
       },
   },
}


#RGCfg_1_D = {
#    "rock_sampler_cfg": RSCfg_1_D,
#    "rock_assets_folder": "assets/USD_Assets/rocks/lunar_rocks/rocks_s2_r128",
#    "instancer_name": "very_small_rock_instancer",
#    "seed": seed_rock,
#    "block_span": 1,
#    "add_colliders": False,
#}
#RGCfg_2_D = {
#    "rock_sampler_cfg": RSCfg_2_D,
#    "rock_assets_folder": "assets/USD_Assets/rocks/lunar_rocks/rocks_s3_r512",
#    "instancer_name": "small_rock_instancer",
#    "seed": seed_rock,
#    "block_span": 2,
#    "add_colliders": False,
#    "collider_mode": "none",
#}
RGCfg_3_D = {
   "rock_sampler_cfg": RSCfg_3_D,
   "rock_assets_folder": "assets/USD_Assets/rocks/lunar_rocks/rocks_s4_r1024",
   "instancer_name": "small_rock_instancer",
   "seed": seed_rock,
   "block_span": 2,
   "add_colliders": False,
   "collider_mode": "none",
}
RGCfg_4_D = {
   "rock_sampler_cfg": RSCfg_4_D,
   "rock_assets_folder": "assets/USD_Assets/rocks/lunar_rocks/rocks_s5_r2048",
   "instancer_name": "small_rock_instancer",
   "seed": seed_rock,
   "block_span": 2,
   "add_colliders": False,
   "collider_mode": "none",
}


LSTCfg_D = {
    "seed": 42,
    "crater_gen_seed": seed_terrain,
    "crater_gen_distribution_seed": seed_terrain,
    "crater_gen_metadata_seed": seed_terrain,
    "rock_gen_main_seed": None,
    "profiling": True,
    "update_every_n_meters": 2.0,
    "z_scale": 1.0,
    "block_size": 25,
    "dbs_max_elements": 10000000,
    "dbs_save_to_disk": False,
    "dbs_write_interval": 1000,
    "hr_dem_resolution": 0.025,
    "hr_dem_generate_craters": True,
    "hr_dem_num_blocks": 2,
    "crater_gen_densities": [0.025, 0.05, 0.5], # [0.025, 0.05, 0.5], # [0.03, 0.025, 0.03],
    "crater_gen_radius": [[1.5, 2.5], [0.75, 1.5], [0.25, 0.5]],
    "crater_gen_profiles_path": "assets/Terrains/crater_spline_profiles.pkl",
    "crater_gen_padding": 10.0,
    "crater_gen_min_xy_ratio": 0.85,
    "crater_gen_max_xy_ratio": 1.0,
    "crater_gen_random_rotation": True,
    "crater_gen_num_unique_profiles": 10000,
    "num_workers_craters": 8,
    "num_workers_interpolation": 1,
    "input_queue_size": 400,
    "output_queue_size": 30,
    "hrdem_interpolation_method": "bicubic",
    "hrdem_interpolator_name": "PIL",
    "hrdem_interpolator_padding": 2,
    "lr_dem_folder_path": "assets/Terrains/SouthPole",
    "lr_dem_name": "Site20_final_adj_5mpp_surf",
    "starting_position": (0, 0),
    "geo_cm_num_texels_per_level": 64,
    "geo_cm_target_res": 0.02,
    "geo_cm_fine_interpolation_method": "bilinear",
    "geo_cm_coarse_interpolation_method": "bilinear",
    "geo_cm_fine_acceleration_mode": "gpu",
    "geo_cm_coarse_acceleration_mode": "gpu",
    "geo_cm_semantic_label": "terrain",
    "geo_cm_texture_name": "LunarRegolith8k",
    "geo_cm_texture_path": "assets/Textures/LunarRegolith8k.mdl",
    "geo_cm_apply_smooth_shading": False,
    "terrain_collider_enabled": False,
    "terrain_collider_resolution": 0.05,
    "terrain_collider_cache_size": 10,
    "terrain_collider_building_threshold": 4.0,
    "rock_gen_cfgs": [
        RGCfg_3_D,
        RGCfg_4_D,
    ],
}


# Function to multiply two quaternions
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    # Quaternion multiplication formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])

# Function to rotate a vector by a quaternion
def rotate_vector_by_quaternion(v, q):
    # Convert the vector into a quaternion (pure quaternion, scalar part is 0)
    v_q = np.array([0, v[0], v[1], v[2]])
    
    # Compute the conjugate of the quaternion q (for unit quaternion, conjugate is inverse)
    q_conjugate = np.array([q[0], -q[1], -q[2], -q[3]])
    
    # Apply the rotation: v' = q * v * q^-1 (q_conjugate)
    rotated_q = quaternion_multiply(quaternion_multiply(q, v_q), q_conjugate)
    
    # The rotated vector is the vector part of the resulting quaternion
    rotated_vector = rotated_q[1:]  # Discard the scalar (w) part
    
    return rotated_vector


def project_vector(a, b):
    # Ensure the vectors are numpy arrays for vector operations
    a = np.array(a)
    b = np.array(b)
    
    # Calculate the dot product of a and b
    dot_product_ab = np.dot(a, b)
    
    # Calculate the dot product of b with itself (magnitude squared of b)
    dot_product_bb = np.dot(b, b)
    
    # Calculate the projection of a onto b
    projection = (dot_product_ab / dot_product_bb) * b
    
    return projection

def transform_trajs(wp_traj, block_x_current, block_y_current, half_block, dem_height):
    np_trajs = wp_traj.numpy()
    np_trajs = np_trajs.reshape(-1, 100, 3)[::50]
    np_trajs = np_trajs.reshape(-1, 3)[::10]

    trajs_to_plot = np_trajs.copy()
    trajs_to_plot[:, 0] = - np_trajs[:, 1] + block_x_current + half_block
    trajs_to_plot[:, 1] = np_trajs[:, 0] + block_y_current + half_block
    # trajs_to_plot[:, 2] += dem_height
    return trajs_to_plot



def extract_rocks_data(rocks_list):
    obstacles = []
    for i in range(0, len(rocks_list), 2):
        positions = rocks_list[i]
        scales = rocks_list[i + 1]

        for pos, scale in zip(positions, scales):
            x, y = pos[0], pos[1]
            radius = max((scale[0], scale[1]))  
            obstacles.append([x, y, radius])

    return obstacles 


def update_costmap_with_obstacles_vectorized(costmap, X, Y, obstacles, robot_radius, map_origin_global, safety_margin=0.15):

    x0, y0 = map_origin_global

    for x_global, y_global, r_obs in obstacles:
        x_local = x_global - x0
        y_local = y_global - y0

        total_radius = r_obs + robot_radius + safety_margin

        mask = (X - x_local)**2 + (Y - y_local)**2 <= total_radius**2
        costmap[mask] = 0.0


if __name__ == "__main__":

    from omni.isaac.core import World
    from omni.usd import get_context
    from pxr import UsdLux, UsdGeom, Gf, UsdShade, Vt, Sdf, Usd
    from omni.isaac.core.utils.stage import open_stage, add_reference_to_stage


    from src.terrain_management.large_scale_terrain.pxr_utils import set_xform_ops
    from src.terrain_management.large_scale_terrain.utils import ScopedTimer
    from src.configurations.environments import LargeScaleTerrainConf
    from src.terrain_management.large_scale_terrain_manager import (
        LargeScaleTerrainManager,
    )
    from assets import get_assets_path
    
    from thesis_master.warp_implementation.MPPI_isaac import Surface, Robot, MPPI_Controller, plot_2d_surface_with_trajectory, plot_costmap_with_frames
    import time


    class HuskyController:
        
        def __init__(self, stage, robot_path = "/husky_robot"):
            self._robot_path = robot_path
            self._stage = stage
            self._robot_base_path = "base_link"
            self._front_left_wheel_path = "base_link/front_left_wheel"
            self._front_right_wheel_path = "base_link/front_right_wheel"
            self._rear_left_wheel_path = "base_link/rear_left_wheel"
            self._rear_right_wheel_path = "base_link/rear_right_wheel"
        
        def get_joints(self):
            self._robot_base = self._stage.GetPrimAtPath(os.path.join(self._robot_path, self._robot_base_path))
            self._joint_flw = self._stage.GetPrimAtPath(os.path.join(self._robot_path, self._front_left_wheel_path))
            self._joint_frw = self._stage.GetPrimAtPath(os.path.join(self._robot_path, self._front_right_wheel_path))
            self._joint_rlw = self._stage.GetPrimAtPath(os.path.join(self._robot_path, self._rear_left_wheel_path))
            self._joint_rrw = self._stage.GetPrimAtPath(os.path.join(self._robot_path, self._rear_right_wheel_path))

        def set_velocity_target(self, cmd_left, cmd_right):
            self._joint_flw.GetAttribute("drive:angular:physics:targetVelocity").Set(cmd_left)
            self._joint_frw.GetAttribute("drive:angular:physics:targetVelocity").Set(cmd_right)
            self._joint_rlw.GetAttribute("drive:angular:physics:targetVelocity").Set(cmd_left)
            self._joint_rrw.GetAttribute("drive:angular:physics:targetVelocity").Set(cmd_right)
            
        def get_wheel_velocities(self):
            left_speed = (self._joint_flw.GetAttribute("drive:angular:physics:targetVelocity").Get() + 
                        self._joint_rlw.GetAttribute("drive:angular:physics:targetVelocity").Get())/2
            right_speed = (self._joint_frw.GetAttribute("drive:angular:physics:targetVelocity").Get() + 
                        self._joint_rrw.GetAttribute("drive:angular:physics:targetVelocity").Get())/2
            return left_speed, right_speed
        
        def get_robot_velocity(self):
            linear_velocity = self._robot_base.GetAttribute("physics:velocity").Get()
            angular_velocity = self._robot_base.GetAttribute("physics:angularVelocity").Get()
            linear_velocity_list = [linear_velocity[0], linear_velocity[1],linear_velocity[2]]
            angular_velocity_list = [angular_velocity[0], angular_velocity[1],angular_velocity[2]]
            return linear_velocity_list, angular_velocity_list

        def get_robot_pose(self):
            position = self._robot_base.GetAttribute("xformOp:translate").Get()
            quaternion = self._robot_base.GetAttribute("xformOp:orient").Get()
            position_list = [position[0], position[1], position[2]]
            quaternion_wxyz = [quaternion.GetReal(), quaternion.GetImaginary()[0], quaternion.GetImaginary()[1], quaternion.GetImaginary()[2]]
            return position_list, quaternion_wxyz
        
        def set_robot_pose(self, position, quat_wxyz):
            pos = Gf.Vec3d(position[0], position[1], position[2])
            quat = Gf.Quatd(quat_wxyz[0], Gf.Vec3d(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]))
            root_prim = self._stage.GetPrimAtPath(self._robot_path)
            root_prim.GetAttribute("xformOp:translate").Set(pos)
            root_prim.GetAttribute("xformOp:orient").Set(quat)


    world = World(stage_units_in_meters=1.0)
    stage = get_context().get_stage()
    asset_path = os.path.join(os.getcwd(), "assets")
    robot_asset_path =  get_assets_path() + "/USD_Assets/robots/ros2_husky_PhysX_vlp16.usd"

    # Let there be light
    light = UsdLux.DistantLight.Define(stage, "/World/sun")
    light.CreateIntensityAttr(1000.0)
    # set_xform_ops(light.GetPrim(), Gf.Vec3d(0, 0, 0), Gf.Quatd(0.76, (0.65, 0, 0)), Gf.Vec3d(1, 1, 1))
    set_xform_ops(light.GetPrim(), Gf.Vec3d(0, 0, 0), Gf.Quatd(np.cos(np.pi/6), Gf.Vec3d(np.sin(np.pi/6), 0, 0)), Gf.Vec3d(1, 1, 1))

    x = 0.0
    y = 0.0
    initial_position = (x,y)

    LSTCfg_D["starting_position"] = (initial_position)

    from src.terrain_management.large_scale_terrain.mppi_instancer import VisualizeMPPI
    visualizer = VisualizeMPPI(stage, "/World/MPPI_Visualizer")
    visualizer.build_visualizer()

    # Prime the world
    for _ in range(100):
        world.step(render=True)

    lstm_settings = LargeScaleTerrainConf(**LSTCfg_D)
    LSTM = LargeScaleTerrainManager(lstm_settings)

    # import warp
    # print(warp.__file__)

    print("starting to build...")
    LSTM.build()
    dem_height = LSTM.get_height_global(initial_position)
    # print(dem_height)
    # Add Ze robot.
    robot_root_path = "/robot_husky"
    add_reference_to_stage(robot_asset_path, robot_root_path)
    robot_root = stage.GetPrimAtPath(robot_root_path)
    HC = HuskyController(stage, robot_root_path)
    HC.set_robot_pose((x,y,dem_height + 0.5), [1.0, 0, 0, 0])

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    for _ in range(100):
        world.step(render=True)

    HC.get_joints()
    
    import warp as wp
    import cv2

    # Parameters
    goal_x = 65.0
    goal_y = 65.0
    initial_hv = ([1.0,0.0,0.0])
    DEM_warp = LSTM.nested_clipmap_manager.fine_clipmap_manager._geo_clipmap.DEM_sampler.dem_wp
    grid_size = 7000
    resolution = 0.025
    half_width = grid_size*resolution/2
    block_size = 25
    half_block = block_size / 2

    # Current hr_dem and block we are in
    DEM_np = DEM_warp.numpy().reshape(grid_size,grid_size)
    block_x_current, block_y_current = LSTM.map_manager.get_hr_map_current_block_coordinates()


    # Convert data into the right frame
    local_x = x - block_x_current - half_block
    local_y = y - block_y_current - half_block
    x_goal_local = goal_x - block_x_current - half_block
    y_goal_local = goal_y - block_y_current - half_block
    traj_points = np.array([[local_y,-local_x]])
    trajectories = [traj_points]

    coords = (x, y)
    origin = (block_x_current + half_block, block_y_current + half_block)


    # Set up surface, robot and controller instances
    rocks_data = extract_rocks_data(LSTM.rocks_data)
    surface = Surface("", "", "manual", "", grid_size=grid_size, half_width=half_width, origin=origin, bumps=[], radius_robot=0.3, obstacles=rocks_data)
    surface.Z = DEM_np
    
    robot = Robot(x=local_y, y=-local_x, heading_vector=initial_hv, config_file="thesis_master/warp_implementation/config.yaml")
    wheel_radius = 0.2
    
    controller_3d = MPPI_Controller(surface, robot, "thesis_master/warp_implementation/config.yaml", goal_x=y_goal_local, goal_y=-x_goal_local, goal_orientation=2.2)
    controller_3d.warp_setup()
    err = 0
    total_err = 0 
    
    print("Starting simulation")

    i = 0
    
    plot_costmap_with_frames(surface.costmap, (-87.5, 87.5, -87.5, 87.5), show_live=False, frame=i, frame_folder='costmap_frame_folder')

    while True:        
        # print("=============================")
        controller_3d.reset("controller")
        controller_3d.MPPI_step(proj="3d")
        # time.sleep(0.001)
        lin_cmd = controller_3d.optimal_lin_vel_wp.numpy()[0]
        ang_cmd = controller_3d.optimal_ang_vel_wp.numpy()[0]

        left_stored = lin_cmd - ang_cmd * controller_3d.robot.radius/2
        right_stored = lin_cmd + ang_cmd * controller_3d.robot.radius/2

        if i % 3 == 0:
            left_cmd = left_stored
            right_cmd = right_stored
            err = 0
            total_err = 0

        i += 1

        HC.set_velocity_target(left_cmd*180/np.pi/wheel_radius, right_cmd*180/np.pi/wheel_radius)
        # HC.set_velocity_target(0, 0)

        position, quat = HC.get_robot_pose()
        coords = (position[0], position[1])

        vector = np.array([1.0, 0.0, 0]) 
        heading_vector = rotate_vector_by_quaternion(vector, quat)
        heading_vector = [heading_vector[1], -heading_vector[0], heading_vector[2]]
        controller_3d.robot.update_position(position[1] - block_y_current - half_block, 
                                            -(position[0] - block_x_current - half_block), 
                                            position[2], 
                                            heading_vector)
        

        lin_vel_vector, ang_vel_vector = HC.get_robot_velocity()

        # lin_vel = np.linalg.norm(lin_vel_vector)
        ang_vel = ang_vel_vector[2]*np.pi/180
        err = ang_vel - ang_cmd
        total_err += err 

        left_cmd += (err / 20 + total_err / 39)
        right_cmd -= (err / 20 + total_err / 39)

        controller_3d.std_dev_u1 = np.maximum(0.25, 0.25 - ang_vel*ang_vel/3)
        controller_3d.std_dev_u2 = np.maximum(0.25, 0.25 + ang_vel*ang_vel/3)

        left_wheel_speed, right_wheel_speed = HC.get_wheel_velocities()
        controller_3d.robot.left_wheel_speed = left_stored # left_wheel_speed*wheel_radius*np.pi/180
        controller_3d.robot.right_wheel_speed = right_stored # right_wheel_speed*wheel_radius*np.pi/180


        # Visualisation tool for the MPPI trajectories
        
        # if i % 50 == 0:
        #     points =  transform_trajs(controller_3d.trajectories, block_x_current, block_y_current, half_block, dem_height)
        #     costs = controller_3d.costs_wp.numpy()
        #     costs = costs[::50]

        #     costs = (costs - np.min(costs))/ (np.max(costs))

        #     costs = costs.repeat(10)
        #     visualizer.update_visualizer(points, costs)


        # if i % 50 == 0:
        #     print("height of the robot:", ( controller_3d.robot.z[-1]))
        #     print("height of the left wheel:", (controller_3d.left_wheel_heights.numpy()[6:9]))
        #     print("height of the right wheel:", (controller_3d.right_wheel_heights.numpy()[6:9]))


        with ScopedTimer("update_visual_mesh", active=False):
            update, coords = LSTM.update_visual_mesh(coords)

        with ScopedTimer("env_step", active=False):
            world.step(render=True)


        coords = (position[0], position[1])

        block_x, block_y = LSTM.map_manager.get_hr_map_current_block_coordinates()
        if block_x_current != block_x or block_y_current != block_y:
            # visualizer.update_visualizer(np.array([[0, 0, 0]]), np.array([1]))

            shift_x = block_x - block_x_current
            shift_y = block_y - block_y_current

            block_x_current = block_x
            block_y_current = block_y

            origin = (block_x_current + half_block, block_y_current + half_block)

            DEM_warp = LSTM.nested_clipmap_manager.fine_clipmap_manager._geo_clipmap.DEM_sampler.dem_wp
            rocks_data = extract_rocks_data(LSTM.rocks_data)

            controller_3d.surface.costmap = controller_3d.surface.create_obstacles_costmap(rocks_data, origin)
            # controller_3d.costmap_wp = wp.array(controller_3d.surface.costmap.flatten(), dtype=wp.float32, device="cuda") 
            controller_3d.costmap_wp.assign(controller_3d.surface.costmap.flatten())
            # DEM_np = DEM_warp.numpy().reshape(grid_size,grid_size)
            plot_costmap_with_frames(controller_3d.surface.costmap, (-87.5, 87.5, -87.5, 87.5), show_live=False, frame=i, frame_folder='costmap_frame_folder')

            controller_3d.Z_wp = DEM_warp
            
            origin = (block_x_current + half_block, block_y_current + half_block)
            for j in range(len(controller_3d.robot.x)):
                controller_3d.robot.x[j] -= shift_y
                controller_3d.robot.y[j] += shift_x

            controller_3d.goal_x -= shift_y
            controller_3d.goal_y += shift_x
            controller_3d.goal = wp.vec2f(controller_3d.goal_x, controller_3d.goal_y)
    


    timeline.stop()
    LSTM.map_manager.hr_dem_gen.shutdown()
    simulation_app.close()
