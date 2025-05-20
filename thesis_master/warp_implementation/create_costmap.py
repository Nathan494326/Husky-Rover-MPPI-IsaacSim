import numpy as np
import cv2
import matplotlib.pyplot as plt


# costmap = np.load('test_nathan_costmap_transformed.npy')
# plt.figure(figsize=(10, 10))
# plt.imshow(costmap, cmap='jet', origin='lower')
# plt.colorbar(label="Cost Value")
# plt.title("Generated Costmap from Binary Grid Map")
# plt.show()


# # Load binary grid map (1 = obstacle, 0 = free space)
# binary_map = np.load("costmap_750_obs.npy").astype(np.uint8)

# # Invert the map: OpenCV expects 0 for obstacles and nonzero for free space
# inverted_map = 1 - binary_map  

# # Compute the distance transform
# distance_map = cv2.distanceTransform(inverted_map, cv2.DIST_L2, 5)

# # Normalize the costmap (optional, to scale between 0 and 1)
# costmap = cv2.normalize(distance_map, None, 0, 1.0, cv2.NORM_MINMAX)

# costmap = (1 - costmap)**10
# # Save the new costmap
# np.save("costmap_750_transformed.npy", costmap)

costmap = np.load('costmap_750_transformed.npy')

# Display the costmap
plt.figure(figsize=(10, 10))
plt.imshow(costmap, cmap='jet', origin='lower')
plt.colorbar(label="Cost Value")
plt.title("Generated Costmap from Binary Grid Map")
plt.show()

