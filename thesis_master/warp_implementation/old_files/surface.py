import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Load the data from the .npy file
data = np.load("test_nathan.npy")

# Take the first 500 elements in both directions
data = data[:, :]  # Slice the first 500 rows and columns

# Display the image
plt.imshow(data, cmap="jet", origin="lower", aspect="equal")  
plt.colorbar(label="Intensity")  # Add a color bar
plt.title("2D Image from test_nathan.npy")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()


# Load the data from the .npy file
data = np.load("test_nathan.npy")

# Take the first 500 elements in both directions
data = data[1000:2500, 1000:2500]  # Slice the first 500 rows and columns
x = np.arange(1500)  # X-axis values (first 500 elements)
y = np.arange(1500)  # Y-axis values (first 500 elements)
X, Y = np.meshgrid(x, y)  # Create coordinate grids

# Create a 3D figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, data, cmap="viridis")

# Labels and title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis (Surface Value)")
ax.set_title("3D Surface Plot (First 500x500 Elements)")

ax.set_xlim(0, 1500)
ax.set_ylim(0, 1500)
ax.set_zlim(-2300, -2150)
# Show the plot
plt.show()
