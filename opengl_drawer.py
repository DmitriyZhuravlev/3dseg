from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np


def generate_random_color():
    """Generates a random color and returns it in both BGR (for cv2) and RGB (for plt) formats."""
    color_rgb = np.random.randint(
        0, 256, 3).tolist()  # Random RGB color in 0-255 range
    color_bgr = color_rgb[::-1]  # Reverse for BGR format for cv2
    color_rgb_normalized = [c / 255.0 for c in color_rgb]  # Normalize for plt
    return color_bgr, color_rgb_normalized
# def draw_3d_bounding_box(ax, lower_face, h, color="orange"):
    # lower_face = np.array(lower_face)

    # # Check if lower_face is in 2D and convert to 3D by adding z=0 for the lower face
    # if lower_face.shape[1] == 2:  # If 2D, add z=0 for the lower face
        # lower_face = np.hstack([lower_face, np.zeros((4, 1))])

    # # Create upper_face by adding height h to the z-coordinates of lower_face
    # upper_face = lower_face.copy()
    # upper_face[:, 2] += h  # Add height to the z-coordinate

    # # Define the vertices of the bounding box
    # verts = [
        # [lower_face[0], lower_face[1], upper_face[1], upper_face[0]],  # Front face
        # [lower_face[1], lower_face[2], upper_face[2], upper_face[1]],  # Right face
        # [lower_face[2], lower_face[3], upper_face[3], upper_face[2]],  # Back face
        # [lower_face[3], lower_face[0], upper_face[0], upper_face[3]],  # Left face
        # [lower_face[0], lower_face[1], lower_face[2], lower_face[3]],  # Bottom face
        # [upper_face[0], upper_face[1], upper_face[2], upper_face[3]]   # Top face
    # ]

    # # Draw the 3D bounding box
    # ax.add_collection3d(Poly3DCollection(verts, facecolors=color, linewidths=1, edgecolors='r', alpha=.25))

def draw_3d_bounding_box(ax, lower_face, h, color=None):
    lower_face = np.array(lower_face)

    if color is None:
        _, color = generate_random_color()

    # Convert to 3D by adding z=0 for the lower face if in 2D
    if lower_face.shape[1] == 2:
        lower_face = np.hstack([lower_face, np.zeros((4, 1))])

    # Create the upper face by adding height h
    upper_face = lower_face.copy()
    upper_face[:, 2] += h

    verts = [
        [lower_face[0], lower_face[1], upper_face[1], upper_face[0]],
        [lower_face[1], lower_face[2], upper_face[2], upper_face[1]],
        [lower_face[2], lower_face[3], upper_face[3], upper_face[2]],
        [lower_face[3], lower_face[0], upper_face[0], upper_face[3]],
        [lower_face[0], lower_face[1], lower_face[2], lower_face[3]],
        [upper_face[0], upper_face[1], upper_face[2], upper_face[3]]
    ]

    ax.add_collection3d(Poly3DCollection(verts, facecolors=color, linewidths=1, edgecolors=color, alpha=.25))

def set_equal_axes(ax):
    """Sets equal scaling for 3D axes to ensure a consistent aspect ratio."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    
    max_range = limits.max() - limits.min()
    center = limits.mean(axis=1)
    
    ax.set_xlim3d([center[0] - max_range / 2, center[0] + max_range / 2])
    ax.set_ylim3d([center[1] - max_range / 2, center[1] + max_range / 2])
    ax.set_zlim3d([center[2] - max_range / 2, center[2] + max_range / 2])
    
def set_axes_limits(ax, lower_faces, heights):
    """Set axis limits based on the range of the bounding boxes."""
    lower_faces = np.array(lower_faces)  # Ensure lower_faces is a NumPy array
    heights = np.array(heights)  # Ensure heights is also a NumPy array

    min_x, max_x = np.min(lower_faces[:, :, 0]), np.max(lower_faces[:, :, 0])
    min_y, max_y = np.min(lower_faces[:, :, 1]), np.max(lower_faces[:, :, 1])
    
    # Add heights to each z-coordinate of the lower faces to find upper limits
    min_z = np.min(lower_faces[:, :, 2])
    max_z = np.max(lower_faces[:, :, 2] + heights[:, np.newaxis])  # Reshape heights to (190, 1)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)

# Main function to draw multiple cubes
def draw_cubes_in_3d(lower_faces, heights, colors):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    while True:
        assert len(lower_faces) == len(heights), "Mismatched number of lower faces and heights"
        ax.cla()  # Clear axes for fresh rendering

        for lower_face, height, color in zip(lower_faces, heights, colors):
            draw_3d_bounding_box(ax, lower_face, height, color=color)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_aspect('equal', adjustable='box')
        
        # Set the axis limits based on the data ranges
        #set_axes_limits(ax, lower_faces, heights)

        plt.draw()
        plt.pause(0.001)
        
        print("Press any key to render new boxes...")
        if plt.waitforbuttonpress():
            break  # Exit loop if a key is pressed
    
    plt.ioff()
    plt.close(fig)
