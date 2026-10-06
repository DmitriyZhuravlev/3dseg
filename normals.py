import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("/home/dzhura/ComputerVision/data/img/test/out/removed_pig.JPG")
color_image_for_slic = image.copy()

# Apply Gaussian Blur to smooth the image (optional but recommended for better gradient calculation)
smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

# Parameters for SLIC segmentation
region_size = 30  # Region size for segmentation
ruler = 10.0*3  # Ruler parameter for segmentation
slic_iterations = 10  # Number of iterations for SLIC

# Apply SLIC segmentation
slic = cv2.ximgproc.createSuperpixelSLIC(
    color_image_for_slic,
    algorithm=cv2.ximgproc.MSLIC,
    region_size=region_size,
    ruler=ruler
)
slic.iterate(slic_iterations)

# Retrieve SLIC labels and contour mask
labels = slic.getLabels()
contour_mask = slic.getLabelContourMask(False)

# Compute gradients (approximating surface normals)
grad_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=5)
grad_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=5)

# Normalize the gradient direction based on lightness for surface normal
normals = np.dstack((grad_x, grad_y))  # Only x and y gradients

# Get unique segment labels
unique_labels = np.unique(labels)

# Initialize an image to draw the result
output_image = color_image_for_slic.copy()

# Iterate through all unique segments to compute the average normal and draw arrows
for label in unique_labels:
    # Get all pixels belonging to the current segment
    segment_pixels = np.argwhere(labels == label)

    # Compute the center of the segment (mean of pixel coordinates)
    center_y = np.mean(segment_pixels[:, 0])
    center_x = np.mean(segment_pixels[:, 1])

    # Compute the average normal for this segment
    segment_normals = normals[segment_pixels[:, 0], segment_pixels[:, 1]]
    avg_normal_x = np.mean(segment_normals[:, 0])
    avg_normal_y = np.mean(segment_normals[:, 1])

    # Normalize the average normal
    norm = np.linalg.norm([avg_normal_x, avg_normal_y])
    if norm != 0:
        avg_normal_x /= norm
        avg_normal_y /= norm

    # Scale the normal for visualization (adjust scale for larger/smaller arrows)
    scale = 10
    avg_normal_x *= scale
    avg_normal_y *= scale

    # Draw the normal vector as an arrow in 2D
    cv2.arrowedLine(output_image, (int(center_x), int(center_y)), 
                    (int(center_x + avg_normal_x), int(center_y + avg_normal_y)), 
                    (0, 255, 0), 2, tipLength = 0.5)

# Display the result
cv2.imshow('SLIC Segments with Normals (2D)', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the result
cv2.imwrite('output_image_with_normals_2d.jpg', output_image)
