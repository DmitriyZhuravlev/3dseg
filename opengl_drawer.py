from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
import cv2
import trimesh
from scipy.spatial import ConvexHull

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
    
def draw_cubes_with_bounding_image(bounding_box_image, lower_faces, heights, colors):
    """
    Display the bounding_box_image on the left and draw 3D bounding boxes on the right.
    """
    #plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(12, 6))  # Create a wide figure for side-by-side display

    # Create subplots
    ax_img = fig.add_subplot(121)  # Left subplot for the bounding_box_image
    ax_3d = fig.add_subplot(122, projection='3d')  # Right subplot for the 3D bounding boxes
    #ax_3d.view_init(elev=90, azim=-90)  # Align axes like in a mathematical system

    if True: #while True:
        # Left: Display the bounding_box_image
        ax_img.cla()  # Clear the image plot
        ax_img.imshow(cv2.cvtColor(bounding_box_image, cv2.COLOR_BGR2RGB))
        ax_img.axis('off')  # Turn off axis for better visualization
        ax_img.set_title("Bounding Box Image")

        # Right: Display the 3D bounding boxes
        ax_3d.cla()  # Clear the 3D plot
        for lower_face, height, color in zip(lower_faces, heights, colors):
            draw_3d_bounding_box(ax_3d, lower_face, height, color=color)

        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title("3D Bounding Boxes")
        #set_equal_axes(ax_3d)  # Ensure equal scaling for the 3D plot

        # plt.draw()
        # plt.pause(0.001)  # Small delay for interactive updates

        # # Break the loop if a key is pressed
        # # print("Press any key to render new boxes...")
        # # if plt.waitforbuttonpress():
            # # break

    # plt.ioff()  # Turn off interactive mode
    # plt.close(fig)  # Close the figure when done
    plt.show()


def draw_3d_points_with_hull(ext_3d):
    # Validate and filter points
    points = np.array([p for p in ext_3d if isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3], dtype=np.float64)

    if len(points) < 4:  # Convex Hull needs at least 4 non-coplanar points
        raise ValueError(f"Insufficient valid 3D points. Found {len(points)} valid points.")
    else:
        print(f"3D points number: {len(points)}")

    # Compute Convex Hull
    hull = ConvexHull(points)

    # Plot 3D Points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="blue", marker="o")


    # Draw Convex Hull
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], "r-")

    # Fill Convex Hull Faces
    ax.add_collection3d(Poly3DCollection(points[hull.simplices], alpha=0.2, edgecolor="r"))

    ax.set_aspect('equal', adjustable='box')
    plt.show()



# def draw_3d_bounding_box(ax, lower_face, height, color='blue'):
    # """
    # Draw a 3D bounding box on the given Matplotlib 3D axis.
    # """
    # # Define the 8 vertices of the cube
    # x, y, z = np.array(lower_face).T
    # h = height
    # upper_face = np.array(lower_face) + [0, 0, h]

    # # Combine bottom and top faces
    # vertices = np.vstack([lower_face, upper_face])

    # # Define faces
    # faces = [
        # [vertices[i] for i in [0, 1, 3, 2]],  # Bottom face
        # [vertices[i] for i in [4, 5, 7, 6]],  # Top face
        # [vertices[i] for i in [0, 1, 5, 4]],  # Side face
        # [vertices[i] for i in [2, 3, 7, 6]],  # Side face
        # [vertices[i] for i in [0, 2, 6, 4]],  # Side face
        # [vertices[i] for i in [1, 3, 7, 5]],  # Side face
    # ]

    # # Draw the cube
    # poly3d = Poly3DCollection(faces, alpha=0.3, edgecolor="k")
    # poly3d.set_facecolor(color)
    # ax.add_collection3d(poly3d)

# def compute_convex_hull_boxes(lower_faces, heights):
    # """
    # Compute additional boxes needed to connect all given bounding boxes into a convex shape.
    # """
    # all_points = np.vstack([np.array(lower_faces), np.array(lower_faces) + [0, 0, heights]])
    
    # hull = ConvexHull(all_points)
    # additional_boxes = []

    # for simplex in hull.simplices:
        # # Compute the centroid of each hull face and generate a connecting box
        # centroid = np.mean(all_points[simplex], axis=0)
        # new_lower_face = centroid[:2]
        # new_height = centroid[2]
        
        # # Ensure it doesn't already exist
        # if not any(np.allclose(new_lower_face, existing[:2]) for existing in lower_faces):
            # additional_boxes.append((new_lower_face, new_height))
    
    # return additional_boxes

# def draw_cubes_with_bounding_image(bounding_box_image, lower_faces, heights, colors):
    # """
    # Display the bounding_box_image on the left and draw 3D bounding boxes on the right,
    # ensuring convex connectivity by adding extra boxes.
    # """
    # fig = plt.figure(figsize=(12, 6))  

    # # Create subplots
    # ax_img = fig.add_subplot(121)  
    # ax_3d = fig.add_subplot(122, projection='3d')  

    # # Left: Display the bounding_box_image
    # ax_img.imshow(cv2.cvtColor(bounding_box_image, cv2.COLOR_BGR2RGB))
    # ax_img.axis('off')  
    # ax_img.set_title("Bounding Box Image")

    # # Compute additional bounding boxes for convex connection
    # additional_boxes = compute_convex_hull_boxes(lower_faces, heights)

    # # Right: Display the 3D bounding boxes
    # for lower_face, height, color in zip(lower_faces, heights, colors):
        # draw_3d_bounding_box(ax_3d, lower_face, height, color=color)

    # # Draw the additional convex hull boxes
    # for lower_face, height in additional_boxes:
        # draw_3d_bounding_box(ax_3d, [lower_face], height, color='gray')

    # ax_3d.set_xlabel('X')
    # ax_3d.set_ylabel('Y')
    # ax_3d.set_zlabel('Z')
    # ax_3d.set_title("Convex 3D Bounding Boxes")

    # plt.show()
