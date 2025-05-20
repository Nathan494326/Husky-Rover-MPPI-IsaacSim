import csv
import numpy as np

def compute_averages(csv_filename):
    with open(csv_filename, mode="r") as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip header row
        data = [list(map(float, row)) for row in reader]  # Convert rows to float
    
    averages = []
    for i in range(0, len(data), 8):  # Process in chunks of 8
        chunk = data[i:i+8]
        if len(chunk) < 8:
            continue  # Ignore incomplete chunks
        avg_values = np.mean(chunk, axis=0)
        averages.append(avg_values)
    
    return headers, averages

# Process both files
csv_filename_2D = "warp_implementation/trajectory_metrics_2D.csv"
csv_filename_3D = "warp_implementation/trajectory_metrics_3D.csv"

headers_2D, averages_2D = compute_averages(csv_filename_2D)
headers_3D, averages_3D = compute_averages(csv_filename_3D)

# Print results
print("Averages for 2D file:")
print(headers_2D)
for avg in averages_2D:
    print(avg)

print("\nAverages for 3D file:")
print(headers_3D)
for avg in averages_3D:
    print(avg)














######## STATS #########



# average_ratio_of_loops_gained = 0
# average_ratio_of_distance_gained = 0
# average_ratio_of_climbed_degrees_gained = 0
# average_ratio_of_up_slope_gained = 0

# for i in range(1, 100):
#     x_start = np.random.uniform(-25.0, -15.0)
#     y_start = np.random.uniform(-10.0, 5.0)
#     x_goal = np.random.uniform(10.0, 22.0)
#     y_goal = np.random.uniform(-22.0, 22.0)
#     x_hr = np.random.uniform(-1.0, 1.0)
#     y_hr = np.random.uniform(-1.0, 1.0)

#     robot_2d = Robot(x=x_start, y=y_start, heading_vector=[x_hr, y_hr, 0.0], radius=0.3)
#     robot_3d = Robot(x=x_start, y=y_start, heading_vector=[x_hr, y_hr, 0.0], radius=0.3)

#     controller_2d = MPPI_Controller(surface, robot_2d, "warp_implementation/config.yaml", goal_x=x_goal, goal_y=y_goal, goal_orientation=2.2)
#     controller_3d = MPPI_Controller(surface, robot_3d, "warp_implementation/config.yaml", goal_x=x_goal, goal_y=y_goal, goal_orientation=2.2)

#     # 2D
#     controller_2d.run("2d")
#     trajectory = np.array(list(zip(robot_2d.x, robot_2d.y, robot_2d.z)))
#     plot_surface_with_trajectory(surface.X, surface.Y, surface.Z, [trajectory], surface.half_width*2, surface.half_width*2, surface.half_width*2) 
#     length_2d, angle_up_2d, angle_down_2d, distance_up_2d = compute_path_metrics(trajectory)
#     print(f"Total Path Length: {length_2d:.2f} meters")
#     print(f"Total Angle Climbed Up: {angle_up_2d:.2f} degrees")
#     print(f"Total Distance Climbed Up: {distance_up_2d:.2f} m")

#     # 3D
#     controller_3d.run("3d")
#     trajectory = np.array(list(zip(robot_3d.x, robot_3d.y, robot_3d.z)))
#     plot_surface_with_trajectory(surface.X, surface.Y, surface.Z, [trajectory], surface.half_width*2, surface.half_width*2, surface.half_width*2) 
#     length_3d, angle_up_3d, angle_down_3d, distance_up_3d = compute_path_metrics(trajectory)
#     print(f"Total Path Length: {length_3d:.2f} meters")
#     print(f"Total Angle Climbed Up: {angle_up_3d:.2f} degrees")
#     print(f"Total Distance Climbed Up: {distance_up_3d:.2f} m")

#     if controller_3d.loop == 2000 and controller_2d == 2000:
#         print(x_goal, y_goal)

#     current_loop_gained_ratio = (controller_2d.loop - controller_3d.loop)/controller_2d.loop
#     average_ratio_of_loops_gained = average_ratio_of_loops_gained * (i-1)/i + current_loop_gained_ratio * 1/i

#     current_ratio_of_distance_gained = (length_2d - length_3d)/length_2d
#     average_ratio_of_distance_gained = average_ratio_of_distance_gained * (i-1)/i + current_ratio_of_distance_gained * 1/i

#     current_ratio_of_climbed_degrees_gained = (angle_up_2d - angle_up_3d)/angle_up_2d
#     average_ratio_of_climbed_degrees_gained = average_ratio_of_climbed_degrees_gained * (i-1)/i + current_ratio_of_climbed_degrees_gained * 1/i

#     current_ratio_of_up_slope_gained = (distance_up_2d - distance_up_3d)/distance_up_2d
#     average_ratio_of_up_slope_gained = average_ratio_of_up_slope_gained * (i-1)/i + current_ratio_of_up_slope_gained * 1/i

#     if i % 10 == 0:
#         print(f"Current Loop Gained Ratio: {average_ratio_of_loops_gained:.6f}")
#         print(f"Current Ratio of Distance Gained: {average_ratio_of_distance_gained:.6f}")
#         print(f"Current Ratio of Climbed Degrees Gained: {average_ratio_of_climbed_degrees_gained:.6f}")
#         print(f"Current Ratio of Up Slope Gained: {average_ratio_of_up_slope_gained:.6f}")

#     print(i)







########## FOR THE STATISTICS #############
# number_of_runs = 8
# start_points = []
# regions = [(-18, -18), (-18, 0), (-18, 18),  
#             (0, 18), (18, 18), (0, 18),        
#             (18, -18), (0, -18)]    

# for cx, cy in regions:
#     for _ in range(number_of_runs):
#         x = np.random.uniform(cx - 2, cx + 2)  # Small variation around center
#         y = np.random.uniform(cy - 2, cy + 2)
#         start_points.append((x, y))


# end_points = []
# regions = [(18, 18), (0, 18), (18, -18),       
#             (0, -18), (-18, -18), (-18, 0), 
#             (-18, 18), (0, 18), ]    

# for cx, cy in regions:
#     for _ in range(number_of_runs):
#         x = np.random.uniform(cx - 2, cx + 2) 
#         y = np.random.uniform(cy - 2, cy + 2)
#         end_points.append((x, y))


# heading_vectors = []
# for i in range(number_of_runs*8):
#     heading_vectors.append([np.random.uniform(-1.0, 1.0) , np.random.uniform(-1.0, 1.0) , 0.0])


# raw_data_2D = []
# raw_data_3D = []

# csv_filename_2D = "warp_implementation/trajectory_metrics_2D.csv"
# csv_filename_3D = "warp_implementation/trajectory_metrics_3D.csv"

# user_input = input("Do you want to clear the CSV files? (yes/no): ")
# if user_input.lower() == 'yes':
#     open(csv_filename_2D, 'w').close()
#     open(csv_filename_3D, 'w').close()
#     print("The CSV files have been cleared.")
# else:
#     print("The CSV files were not cleared.")


# print("Computing the data with 2D projections...")
# for i in range(len(start_points)):
#     robot = Robot(x=start_points[i][0], y=start_points[i][1], heading_vector=heading_vectors[i], radius=0.3)
#     controller = MPPI_Controller(surface, robot, "warp_implementation/config.yaml", goal_x=end_points[i][0], goal_y=end_points[i][1], goal_orientation=2.2)
#     controller.run("2d")
#     trajectory = np.array(list(zip(robot.x, robot.y, robot.z)))

#     length, angle_up, angle_down, distance_up = compute_path_metrics(trajectory)
#     raw_data_2D.append([controller.loop, length, angle_up, angle_down, distance_up])
#     print(i)
    

# print("Computing the data with 3D projections...")
# for i in range(len(start_points)):
#     robot = Robot(x=start_points[i][0], y=start_points[i][1], heading_vector=heading_vectors[i], radius=0.3)
#     controller = MPPI_Controller(surface, robot, "warp_implementation/config.yaml", goal_x=end_points[i][0], goal_y=end_points[i][1], goal_orientation=2.2)
#     controller.run("3d")
#     trajectory = np.array(list(zip(robot.x, robot.y, robot.z)))

#     length, angle_up, angle_down, distance_up = compute_path_metrics(trajectory)
#     raw_data_3D.append([controller.loop, length, angle_up, angle_down, distance_up])
#     print(i)


# with open(csv_filename_2D, mode="w", newline="") as file:
#     writer = csv.writer(file)
    
#     writer.writerow(["Loops", "Length", "Angle Up", "Angle Down", "Distance Up"])
    
#     for i in range(len(raw_data_2D)):        
#         writer.writerow(raw_data_2D[i])

# with open(csv_filename_3D, mode="w", newline="") as file:
#     writer = csv.writer(file)
    
#     writer.writerow(["Loops", "Length", "Angle Up", "Angle Down", "Distance Up"])
    
#     for i in range(len(raw_data_3D)):        
#         writer.writerow(raw_data_3D[i])

# print(f"Metrics saved correctly!")

