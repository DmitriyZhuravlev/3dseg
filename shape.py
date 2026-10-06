import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load the grayscale image
image_path = "/home/dzhura/ComputerVision/data/img/test/out/removed_pig.JPG"  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize image for faster processing (retain aspect ratio)
target_size = 150  # Adjust this as needed
height, width = image.shape
scale = target_size / max(height, width)
new_size = (int(width * scale), int(height * scale))
image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

# Normalize the image to range [0, 1]
image = image.astype(np.float32) / 255.0

# Smooth the image to reduce noise
smoothed_image = gaussian_filter(image, sigma=2)

# Compute gradients (approximating surface normals)
grad_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=5)
grad_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=5)

# Compute depth from gradients using Poisson reconstruction
def poisson_reconstruction(grad_x, grad_y):
    h, w = grad_x.shape
    grad_z = np.zeros((h, w), dtype=np.float32)
    grad_z[1:, :] += np.cumsum(grad_y[1:, :], axis=0)
    grad_z[:, 1:] += np.cumsum(grad_x[:, 1:], axis=1)
    return grad_z

depth_map = poisson_reconstruction(grad_x, grad_y)

# Normalize depth map
depth_map -= depth_map.min()
depth_map /= depth_map.max()

# Estimate the ground plane using RANSAC
def estimate_ground_plane_old(x, y, z):
    """Estimate the ground plane from x, y, z points using RANSAC."""
    points = np.stack((x, y), axis=1)  # Combine x and y coordinates
    model = make_pipeline(PolynomialFeatures(degree=1), RANSACRegressor())
    model.fit(points, z)
    z_pred = model.predict(points)
    return z_pred.reshape(x.shape), model
    
# Estimate the ground plane using RANSAC
def estimate_ground_plane(x, y, z):
    """Estimate the ground plane from x, y, z points using RANSAC."""
    # Combine x and y into a single feature array
    points = np.stack((x, y), axis=1)
    
    # Create a pipeline with PolynomialFeatures and RANSACRegressor
    model = make_pipeline(PolynomialFeatures(degree=1), RANSACRegressor())
    model.fit(points, z)  # Fit the model to the points
    
    # Predict the z values of the plane
    z_pred = model.predict(points)
    
    # Extract the coefficients of the plane equation z = ax + by + c
    poly_features = model.named_steps['polynomialfeatures']
    ransac = model.named_steps['ransacregressor']
    coefficients = ransac.estimator_.coef_
    intercept = ransac.estimator_.intercept_
    
    # Compute the normal vector of the plane (a, b, -1)
    a, b = coefficients[1:3]  # Coefficients correspond to x and y
    normal = np.array([a, b, -1])
    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
    
    return z_pred.reshape(x.shape), model, normal

# Generate 3D coordinates
height, width = depth_map.shape
x = np.linspace(0, width - 1, width)
y = np.linspace(0, height - 1, height)
x, y = np.meshgrid(x, y)
z = depth_map

# Flatten for ground plane fitting
x_flat, y_flat, z_flat = x.flatten(), y.flatten(), z.flatten()

# Fit ground plane and compute predictions
z_ground_flat, ground_model, _ = estimate_ground_plane(x_flat, y_flat, z_flat)

# Reshape the ground plane back to the depth map shape
z_ground = z_ground_flat.reshape(depth_map.shape)

# Align ground plane to z = 0
ground_plane_offset = np.min(z_ground)
z_ground -= ground_plane_offset
z -= ground_plane_offset

# Subtract ground plane from depth map for object detection
z_above_ground = z - z_ground
z_above_ground[z_above_ground < 0] = 0  # Clamp values below ground plane

# Debug: Visualize the ground plane and depth above it
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(depth_map, cmap='viridis')
plt.title("Original Depth Map")
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(z_ground, cmap='viridis')
plt.title("Estimated Ground Plane (Aligned to z=0)")
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(z_above_ground, cmap='viridis')
plt.title("Depth Above Ground")
plt.colorbar()
plt.show()

# 3D Visualization: Plot ground plane and depth map
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot original depth map
ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(z), alpha=0.7, rstride=1, cstride=1, antialiased=True, shade=False)

# Plot ground plane
#ax.plot_surface(x, y, z_ground, color='red', alpha=0.5, rstride=1, cstride=1, label="Ground Plane")

# # Light and camera position
# camera_light_position = np.array([width // 2, -height, 1])

# # Plot light rays
# rays_end = np.array([x.flatten(), y.flatten(), z.flatten()]).T
# for end in rays_end[::1000]:  # Sample rays to avoid clutter
    # ax.plot(
        # [camera_light_position[0], end[0]],  # x-coordinates
        # [camera_light_position[1], end[1]],  # y-coordinates
        # [camera_light_position[2], end[2]],  # z-coordinates
        # color="yellow", alpha=0.3, lw=0.5
    # )

# # Light source point
# ax.scatter(*camera_light_position, color="red", label="Light/Camera Source")

ax.set_title("Reconstructed 3D Shape with Ground Plane and Light Rays")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()
