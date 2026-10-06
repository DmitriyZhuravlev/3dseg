from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from collections import defaultdict
from scipy.spatial import distance
#import trimesh
#from scipy.spatial import ConvexHull
from enum import Enum
class cv_colors(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    ORANGE = (44, 162, 247)
    PURPLE = (247, 44, 200)
    MINT = (239, 255, 66)
    YELLOW = (2, 255, 250)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    GRAY = (128, 128, 128)
    LIGHT_BLUE = (173, 216, 230)
    DARK_GREEN = (0, 100, 0)
    BROWN = (42, 42, 165)
    PINK = (203, 192, 255)
    GOLD = (0, 215, 255)
    SILVER = (192, 192, 192)
    TEAL = (128, 128, 0)
    NAVY = (128, 0, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    
def draw_interpolated_surface(centers, resolution=100, method='linear'):
    """
    Draws an interpolated smooth surface using the 3D cube centers.

    :param centers: List of 3D center points (N, 3) -> [(x, y, z)].
    :param resolution: Grid resolution for interpolation.
    :param method: Interpolation method ('linear', 'cubic', 'nearest').
    """

    # Extract X, Y, and Z from centers
    X = centers[:, 0]
    Y = centers[:, 1]
    Z = centers[:, 2]  # These represent the surface heights

    # Create a grid for interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(X.min(), X.max(), resolution),
        np.linspace(Y.min(), Y.max(), resolution)
    )

    # Interpolate Z values
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method=method)
    
    # Clip Z values to ensure Z >= 0
    grid_z = np.clip(grid_z, 0, None)  # Clipping only lower bound

    # Plot the interpolated surface
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')

    # Scatter the original centers for reference
    ax.scatter(X, Y, Z, color='red', label='Cube Centers')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (Height)")
    ax.set_title("Interpolated Surface from 3D Cube Centers")
    ax.set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show(block=False)
    
def draw_cuboids(ax, bottoms, heights, color='blue', alpha=0.3):
    """Draw 3D cuboids using bottom face points and heights."""
    for bottom, height in zip(bottoms, heights):
        # Define top face by adding height to the Z-coordinates
        top = bottom + np.array([0, 0, height])

        # Define vertices of the cuboid
        vertices = np.array([
            bottom[0], bottom[1], bottom[2], bottom[3],  # Lower face
            top[0], top[1], top[2], top[3]               # Upper face
        ])
        
        # Define faces (quads)
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Side 1
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Side 2
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Side 3
            [vertices[3], vertices[0], vertices[4], vertices[7]],  # Side 4
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]]   # Top
        ]

        # Add cuboid to plot
        poly3d = Poly3DCollection(faces, color=color, alpha=alpha, edgecolor='black')
        ax.add_collection3d(poly3d)

def draw_triangulated_surface(ax, tops):
    """Create and plot a Delaunay triangulated surface based on cuboid tops."""
    points = np.array([top.mean(axis=0) for top in tops])  # Get center points of top faces
    tri = Delaunay(points[:, :2])  # Triangulate over X-Y plane

    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=tri.simplices, cmap='viridis', alpha=0.7)


def draw_interpolated_surface(tops, resolution=1000, method='linear'):
    """
    Draws an interpolated smooth surface using the 3D upper face points of cubes.

    :param tops: List of 4 corner points for each cube's upper face [[(x1, y1, z1), (x2, y2, z2), ...]].
    :param xmax: Maximum X value for clipping.
    :param ymax: Maximum Y value for clipping.
    :param zmax: Maximum Z value for clipping.
    :param resolution: Grid resolution for interpolation.
    :param method: Interpolation method ('linear', 'cubic', 'nearest').
    """

    # Convert list of 4-point faces into a flat numpy array
    upper_face_3d = np.array(tops, dtype=np.float64).reshape(-1, 3)  # Flatten to (N, 3)

    X, Y, Z = upper_face_3d[:, 0], upper_face_3d[:, 1], upper_face_3d[:, 2]

    # Create a grid for interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, X.max(), resolution),
        np.linspace(0, Y.max(), resolution)
    )

    # Interpolate Z values
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method=method)

    # Clip interpolated values to the valid range [0, zmax]
    # grid_z = np.clip(grid_z, 0, zmax)

    # Plot the interpolated surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')

    # Scatter the original upper face points
    ax.scatter(X, Y, Z, color='red', s=10, label='Upper Face Points')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (Height)")
    ax.set_title("Interpolated Surface from Cube Upper Faces")
    plt.legend()
    plt.show(block=False)

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

def darken_color(color, factor=0.7):
    rgb = mcolors.to_rgb(color)  # Convert to (R, G, B)
    return tuple([max(0, c * factor) for c in rgb])  # Darken by factor

def draw_3d_bounding_box(ax, lower_face, h, color="blue", alpha = 0.25):
    lower_face = np.array(lower_face)

    if color is None:
        _, color = generate_random_color()

    edge_color = darken_color(color)  # Compute darker edge color

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

    ax.add_collection3d(Poly3DCollection(verts, facecolors=color, linewidths=1, edgecolors=edge_color, alpha=alpha))

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
# def draw_cubes_in_3d(lower_faces, heights, colors):
    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # while True:
        # assert len(lower_faces) == len(heights), "Mismatched number of lower faces and heights"
        # ax.cla()  # Clear axes for fresh rendering

        # for lower_face, height, color in zip(lower_faces, heights, colors):
            # draw_3d_bounding_box(ax, lower_face, height, color=color)

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # ax.set_aspect('equal', adjustable='box')
        
        # # Set the axis limits based on the data ranges
        # #set_axes_limits(ax, lower_faces, heights)

        # plt.draw()
        # plt.pause(0.001)
        
        # print("Press any key to render new boxes...")
        # if plt.waitforbuttonpress():
            # break  # Exit loop if a key is pressed
    
    # plt.ioff()
    #plt.close(fig)
    
def draw_cubes_in_3d(lower_faces, heights, colors, closest_points_3d=None):
    """
    Draws multiple cubes in 3D and optionally plots closest points.
    
    Args:
        lower_faces (list): List of lower face coordinates for each cube.
        heights (list): List of heights for each cube.
        colors (list): List of colors for each cube.
        closest_points_3d (list, optional): List of closest 3D points to plot.
    """
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    while plt.fignum_exists(fig.number):  # Keeps running unless the window is closed
        assert len(lower_faces) == len(heights), "Mismatched number of lower faces and heights"
        ax.cla()  # Clear axes for fresh rendering

        # Draw 3D bounding boxes
        for lower_face, height, color in zip(lower_faces, heights, colors):
            if color is not None:
                draw_3d_bounding_box(ax, lower_face, height, color=color)

        # Draw closest 3D points (if provided)
        if closest_points_3d:
            #closest_points_3d = np.array(closest_points_3d)
            for i, point in enumerate(closest_points_3d):
                cv = list(cv_colors)[(i) % len(cv_colors)].value  # Ensure unique colors
                color = [c / 255.0 for c in cv[::-1]]  # Normalize for plt
                ax.scatter(point[0], point[1], point[2], color=color, s=100, label=f"Closest Point {i}")
                    # ax.scatter(closest_points_3d[:, 0], closest_points_3d[:, 1], closest_points_3d[:, 2],
                               # color="red", s=50, label="Closest Points")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_aspect("auto")
        ax.legend()

        plt.draw()
        plt.pause(0.001)

        print("Press any key to update, close the window to exit...")
        if plt.waitforbuttonpress():  # Now it waits but doesn't break the loop
            break

    plt.ioff()  # Turn off interactive mode when the window is closed
    
def draw_cubes_with_bounding_image(bounding_box_image, lower_faces, heights, colors, alpha = 0.25):
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
        #colors = [(128, 128, 128) if c is None else c for c in (colors or [])]
        for lower_face, height, color in zip(lower_faces, heights, colors):
            # if color is None:
            #color = [c / 255.0 for c in cv_colors.BLUE.value[::-1]] 
            draw_3d_bounding_box(ax_3d, lower_face, height, color, alpha) #, color=color)


        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title("3D Bounding Boxes")
        #set_equal_axes(ax_3d)  # Ensure equal scaling for the 3D plot
        ax_3d.set_aspect('equal', adjustable='box')

        # plt.draw()
        # plt.pause(0.001)  # Small delay for interactive updates

        # # Break the loop if a key is pressed
        # # print("Press any key to render new boxes...")
        # # if plt.waitforbuttonpress():
            # # break

    # plt.ioff()  # Turn off interactive mode
    # plt.close(fig)  # Close the figure when done
    plt.show(block=False)


# def draw_3d_points_with_hull(ext_3d):
    # # Validate and filter points
    # points = np.array([p for p in ext_3d if isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3], dtype=np.float64)

    # if len(points) < 4:  # Convex Hull needs at least 4 non-coplanar points
        # raise ValueError(f"Insufficient valid 3D points. Found {len(points)} valid points.")
    # else:
        # print(f"3D points number: {len(points)}")

    # # Compute Convex Hull
    # hull = ConvexHull(points)

    # # Plot 3D Points
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="blue", marker="o")


    # # Draw Convex Hull
    # for simplex in hull.simplices:
        # ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], "r-")

    # # Fill Convex Hull Faces
    # ax.add_collection3d(Poly3DCollection(points[hull.simplices], alpha=0.2, edgecolor="r"))

    # ax.set_aspect('equal', adjustable='box')
    # plt.show()



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



def triangulate_face(face_indices):
    """Triangulate a quad face (split into two triangles)."""
    return [
        (face_indices[0], face_indices[1], face_indices[2]),
        (face_indices[0], face_indices[2], face_indices[3])
    ]

def build_mesh_from_cuboids(bottoms, tops):
    vertices = []
    vertex_map = {}  # map from tuple to vertex index
    faces_counter = defaultdict(int)

    def add_vertex(v):
        v_key = tuple(np.round(v, decimals=5))  # rounding helps with float precision
        if v_key not in vertex_map:
            vertex_map[v_key] = len(vertices)
            vertices.append(v_key)
        return vertex_map[v_key]

    def add_face(face_pts):
        indices = [add_vertex(p) for p in face_pts]
        key = tuple(sorted(indices))  # unordered key to detect internal faces
        faces_counter[key] += 1
        return indices

    for bottom, top in zip(bottoms, tops):
        # bottom and top are arrays of shape (4, 3), assumed ordered
        # Build vertical sides by connecting edges between bottom and top
        for i in range(4):
            j = (i + 1) % 4
            side = [bottom[i], bottom[j], top[j], top[i]]
            indices = add_face(side)

        # Add bottom face
        bottom_face = add_face(bottom)
        # Add top face
        top_face = add_face(top)

    # Keep only faces that appear exactly once (external faces)
    final_faces = []
    for face_key, count in faces_counter.items():
        if count == 1:
            # Triangulate the quad (face_key is sorted, so restore order from original)
            ordered_face = [i for i in face_key]
            final_faces.extend(triangulate_face(ordered_face))

    return np.array(vertices), final_faces


# def draw_mesh_matplotlib(vertices, triangles, color='skyblue', edge_color='k', alpha=0.8):
    # """
    # Draws a 3D triangular mesh using matplotlib.
    
    # Args:
        # vertices (np.ndarray): Nx3 array of 3D vertex positions.
        # triangles (list of tuples): List of triangular face indices (3-tuples).
        # color (str): Face color.
        # edge_color (str): Edge line color.
        # alpha (float): Transparency.
    # """
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # mesh_faces = []
    # for tri in triangles:
        # pts = [vertices[i] for i in tri]
        # mesh_faces.append(pts)

    # mesh_collection = Poly3DCollection(mesh_faces, facecolors=color, edgecolors=edge_color, linewidths=0.5, alpha=alpha)
    # ax.add_collection3d(mesh_collection)

    # # Auto scale to the mesh size
    # verts_np = np.array(vertices)
    # max_range = (verts_np.max(axis=0) - verts_np.min(axis=0)).max() / 2.0
    # mid = verts_np.mean(axis=0)

    # ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    # ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    # ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # plt.tight_layout()
    # plt.show()


def build_triangulation_from_upper_face_centers(tops):
    """
    Build triangle mesh based on centers of upper faces, each connected to its two nearest neighbors.

    Args:
        tops (list of np.ndarray): Each item is a 4x3 array (vertices of the top face of a cuboid).

    Returns:
        centers (np.ndarray): N x 3 array of 3D center points.
        triangles (list of tuple): Each triangle is a tuple of 3 indices.
    """
    centers = []
    for i, top in enumerate(tops):
        if top.shape != (4, 3):
            print(f"[WARNING] Invalid shape at index {i}: {top.shape}, expected (4, 3)")
            continue
        center = np.mean(top, axis=0)
        centers.append(center)

    centers = np.array(centers)
    triangles = []

    if len(centers) < 3:
        print("[INFO] Not enough centers for triangulation.")
        return centers, triangles

    for i, ci in enumerate(centers):
        dists = distance.cdist([ci], centers)[0]
        nearest_indices = np.argsort(dists)

        # Skip itself
        neighbors = nearest_indices[1:3]
        if len(neighbors) >= 2:
            triangles.append((i, neighbors[0], neighbors[1]))

    return centers, triangles

def draw_mesh_matplotlib(vertices, triangles, color='skyblue'):
    """
    Draws 3D triangular mesh using matplotlib.

    Args:
        vertices (np.ndarray): N x 3 array of 3D points.
        triangles (list of tuples): List of 3 index tuples forming triangles.
    """
    
    print("Vertices shape:", vertices.shape)
    #print("Vertices:", vertices)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Draw triangle faces
    mesh = []
    for tri in triangles:
        pts = [vertices[i] for i in tri]
        mesh.append(pts)

    tri_collection = Poly3DCollection(mesh, facecolors=color, alpha=0.5, edgecolor='k')
    ax.add_collection3d(tri_collection)

    # Draw points
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='r', s=20)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Mesh Based on Upper Face Centers")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Real-time OpenGL viewer (viewer3d/) integration
# ---------------------------------------------------------------------------
def _cuboids_payload(lower_faces, heights, colors=None):
    cuboids = []
    colors = list(colors) if colors is not None else [None] * len(lower_faces)
    for lower_face, h, color in zip(lower_faces, heights, colors):
        lf = np.asarray(lower_face, dtype=np.float64)
        if lf.shape[1] == 2:  # same convention as draw_3d_bounding_box: z = 0
            lf = np.hstack([lf, np.zeros((len(lf), 1))])
        entry = {"bottom": lf.tolist(), "height": float(h)}
        if color is not None:
            entry["color"] = [float(c) for c in mcolors.to_rgb(color)] if isinstance(color, str) \
                else [float(c) for c in color]
        cuboids.append(entry)
    return {"cuboids": cuboids, "z_up": True}


def export_cuboids_json(path, lower_faces, heights, colors=None):
    """Save cuboids (same arguments as draw_cubes_with_bounding_image) for `python -m viewer3d --scene`."""
    import json
    with open(path, "w") as f:
        json.dump(_cuboids_payload(lower_faces, heights, colors), f)
    return path


def show_cuboids_3d(lower_faces, heights, colors=None, ground_texture=None):
    """Open the interactive OpenGL viewer on the given cuboids (blocks until closed)."""
    from viewer3d.app import run_window
    from viewer3d.scenes import DEFAULT_GROUND_TEXTURE, build_cuboid_scene
    scene = build_cuboid_scene(_cuboids_payload(lower_faces, heights, colors),
                               ground_texture=ground_texture or DEFAULT_GROUND_TEXTURE)
    run_window(scene)
