import cv2
import numpy as np
import os
from opengl_drawer import *
#from shapely.geometry import Polygon
from shapely.geometry import LineString, Point, Polygon
from scipy.spatial import ConvexHull
import flowiz as fz

from lifting import *
from enum import Enum
from collections import deque

debug = True

# vertical vp
line1 = [[710, 952], [646, 234]]
line2 = [[2326, 1162], [2466, 478]]

# left horizontal vp 
line3 = [[1072, 1572], [720, 982]]
line4 = [[1082, 810], [668, 240]]
#line4 = [[2452, 442], [1900, 24]]

# right horizontal vp 
line5 = [[1128, 1574], [2320, 1156]]
line6 = [[1098, 874], [2458, 488]]

# Define input lines as NumPy arrays
lines = {
    "vertical_lines": np.array([line1, line2]),
    "left_hor_lines": np.array([line3, line4]),
    "right_hor_lines": np.array([line5, line6]),
}


def compute_intersection(line1, line2):
    """
    Compute the intersection of two lines.
    Args:
        line1: [[x1, y1], [x2, y2]] (two points defining the first line).
        line2: [[x1, y1], [x2, y2]] (two points defining the second line).
    Returns:
        Intersection point [x, y] or None if lines are parallel.
    """
    l1 = np.cross([line1[0][0], line1[0][1], 1], [line1[1][0], line1[1][1], 1])
    l2 = np.cross([line2[0][0], line2[0][1], 1], [line2[1][0], line2[1][1], 1])
    intersection = np.cross(l1, l2)
    if intersection[2] == 0:
        return None  # Parallel lines
    return [intersection[0] / intersection[2], intersection[1] / intersection[2]]

# Compute vanishing points
vert_vp = compute_intersection(lines["vertical_lines"][0], lines["vertical_lines"][1])
hor_left_vp = compute_intersection(lines["left_hor_lines"][0], lines["left_hor_lines"][1])
hor_right_vp = compute_intersection(lines["right_hor_lines"][0], lines["right_hor_lines"][1])




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

class HeatMapColors(Enum):
    DARK_BLUE = (128, 0, 0)    # Coolest
    BLUE = (255, 0, 0)
    LIGHT_BLUE = (255, 128, 0)
    GREEN = (255, 255, 0)
    YELLOW = (128, 255, 128)
    ORANGE = (0, 255, 255)
    RED = (0, 165, 255)        # Hottest
    # WHITE = (255, 255, 255)    # High intensity overflow
    # BLACK = (0, 0, 0)          # Low intensity base


def interpolate_heatmap_colors(levels: int):
    """
    Interpolate between the heatmap colors for smooth transitions.
    
    :param levels: Total number of levels in the heatmap.
    :return: A list of interpolated colors for each level.
    """
    # Define the key colors from the HeatMapColors enum
    key_colors = [color.value for color in HeatMapColors]
    key_points = np.linspace(0, 1, len(key_colors))  # Normalize key points between 0 and 1
    
    # Create interpolation functions for R, G, B channels
    red_interp = np.interp(np.linspace(0, 1, levels), key_points, [c[2] for c in key_colors])
    green_interp = np.interp(np.linspace(0, 1, levels), key_points, [c[1] for c in key_colors])
    blue_interp = np.interp(np.linspace(0, 1, levels), key_points, [c[0] for c in key_colors])
    
    # Combine interpolated channels into a list of RGB tuples
    interpolated_colors = [(int(b), int(g), int(r)) for r, g, b in zip(red_interp, green_interp, blue_interp)]
    return interpolated_colors

# Function to convert labels to a color image
def labels_to_color(labels):
    # Create a color map: assign each unique label a random color
    unique_labels = np.unique(labels)
    color_map = {label: np.random.randint(0, 255, 3) for label in unique_labels}
    
    # Create an RGB image where each pixel is colored based on its label
    color_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        color_image[labels == label] = color
    
    color_image[labels == 0] = cv_colors.WHITE.value
    return color_image
    
def find_boundary_with_diff(moving_mask):
    # Compute row and column differences
    diff_x = np.diff(moving_mask, axis=1, append=0)
    diff_y = np.diff(moving_mask, axis=0, append=0)

    # Combine differences to form a boundary mask
    boundary_mask = ((diff_x != 0) | (diff_y != 0)).astype(np.uint8)

    return boundary_mask
    
def find_tangent_points(mask_center, hull_points, vp):

   # Step 3: Calculate the reference angle (line from VP to mask center)
    ref_angle = np.arctan2(mask_center[1] - vp[1], mask_center[0] - vp[0])

    # Step 4: Compute angles for all boundary points
    tangent_candidates = []
    for vertex in hull_points:
        angle = np.arctan2(vertex[1] - vp[1], vertex[0] - vp[0])  # Angle from VP to vertex
        angle_diff = angle - ref_angle

        # Normalize angle difference to be in range [-π, π]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        tangent_candidates.append((angle_diff, vertex))

    # Step 5: Find the maximum positive and negative angle differences
    positive_angle_candidate = max(tangent_candidates, key=lambda x: x[0])
    negative_angle_candidate = min(tangent_candidates, key=lambda x: x[0])

    pos_point = positive_angle_candidate[1]
    neg_point = negative_angle_candidate[1]

    # Step 6: Compute the tangent line equations in homogeneous coordinates
    pos_line = np.cross([vp[0], vp[1], 1], [pos_point[0], pos_point[1], 1])
    neg_line = np.cross([vp[0], vp[1], 1], [neg_point[0], neg_point[1], 1])

    return pos_point, pos_line, neg_point, neg_line


    
def extract_segment_pixels(boundary_mask, pos_point, neg_point):
    # Step 1: Extract all boundary points
    boundary_points = np.argwhere(boundary_mask > 0)  # Format: [(y, x), ...]

    # Convert to (x, y) format for easier processing
    boundary_points_xy = [(x, y) for y, x in boundary_points]

    # Step 2: Sort boundary points based on their angle from the pos_point
    def angle_from_pos(point):
        # Compute angle of point relative to pos_point using arctan2
        return np.arctan2(point[1] - pos_point[1], point[0] - pos_point[0])

    # Sort points by angle from pos_point
    boundary_points_xy.sort(key=angle_from_pos)

    # Step 3: Find the indices of pos_point and neg_point in sorted points
    pos_idx = next(i for i, point in enumerate(boundary_points_xy) if np.array_equal(point, tuple(pos_point)))
    neg_idx = next(i for i, point in enumerate(boundary_points_xy) if np.array_equal(point, tuple(neg_point)))

    # Step 4: Extract the segment by traversing the sorted boundary
    if pos_idx <= neg_idx:
        segment = boundary_points_xy[pos_idx:neg_idx + 1]
    else:
        # Wrap around the boundary
        segment = boundary_points_xy[pos_idx:] + boundary_points_xy[:neg_idx + 1]

    # Convert back to (y, x) format for further processing
    segment_pixels = [(x, y) for x, y in segment]
    return segment_pixels
    
def get_projected_box(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False):
    #mask = (labels == label)
    lower_face, upper_face = None, None
    ys, xs = np.where(mask)
    
    if xs.size > 0 and ys.size > 0:
        # Compute boundary points
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        top_point = (xs[np.argmin(ys)], y_min)
        bottom_point = (xs[np.argmax(ys)], y_max)
        left_point = (x_min, ys[np.argmin(xs)])
        right_point = (x_max, ys[np.argmax(xs)])
        
        # Collect all points in the mask
        points = np.argwhere(mask > 0)[:, [1, 0]]  # Switch to (x, y) format
        
        if len(points) < 3:
            # Skip segments with fewer than 3 points
            return None, None
        
        # Calculate mask center
        mask_center = np.mean(points, axis=0)
        
        # Compute 3D box from plain mask
        try:
            lower_face, upper_face = compute_3d_box_from_plain_mask_new(
                mask, vert_vp, hor_left_vp, hor_right_vp, debug)
        except Exception as e:
            print(f"Error processing segment with label")# {label}: {e}")
            return None, None
        
    return lower_face, upper_face
    
def draw_cube(image,
              lower_face,
              upper_face,
              color=cv_colors.RED.value,
              thickness=1):
    """
    Draws a cube using the provided lower and upper faces, handling index mismatches.
    Draws the lower and upper faces and the vertical lines connecting corresponding vertices.
    
    Args:
    - image: The image on which to draw.
    - lower_face: np.array of shape (4, 2), lower face of the cube.
    - upper_face: np.array of shape (4, 2), upper face of the cube.
    - color: The color to use for drawing (default: red).
    - thickness: The line thickness (default: 1).
    """
    # Ensure both faces are numpy arrays with 4 points each
    lower_face = np.array(lower_face, dtype=np.int32).reshape((-1, 1, 2))
    upper_face = np.array(upper_face, dtype=np.int32).reshape((-1, 1, 2))

    # Draw the lower and upper faces as quadrilaterals
    cv2.polylines(image, [lower_face],
                  isClosed=True,
                  color=color,
                  thickness=thickness)
    cv2.polylines(image, [upper_face],
                  isClosed=True,
                  color=color,
                  thickness=thickness)

    # Define the index mapping between lower and upper face
    index_mapping = [(0, 0), (1, 1), (2, 2), (3, 3)]

    # Draw vertical lines connecting corresponding points based on index mapping
    for lower_idx, upper_idx in index_mapping:
        cv2.line(image,
                 tuple(lower_face[lower_idx][0]),
                 tuple(upper_face[upper_idx][0]),
                 color=color,
                 thickness=thickness)

    return image

def compute_3d_box_from_plain_mask_new(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False):

    # Step 1: Extract non-zero points (boundary of the mask)
    points = np.argwhere(mask > 0)  # Extract all non-zero pixel coordinates (row, col)
    points = points[:, [1, 0]]  # Switch to (x, y) format for consistency

    if len(points) < 3:
        raise ValueError(f"Not enough points in mask for a valid polygon. Found {len(points)} points.")

    # Step 2: Compute Convex Hull for a clean boundary
    #hull = ConvexHull(points)
    hull_points = points #points[hull.vertices]
    #polygon = Polygon(hull_points)  # Create a polygon using the convex hull

    # Step 3: Calculate the reference angle (line from VP to mask center)
    mask_center = np.mean(hull_points, axis=0)  # Centroid of the convex hull
    #ref_angle = np.arctan2(mask_center[1] - vp[1], mask_center[0] - vp[0])
    # TODO left, right ?
    point_b, line2, point_a, line1 = find_tangent_points(mask_center, hull_points, hor_right_vp)
    point_d, line4, point_c, line3  = find_tangent_points(mask_center, hull_points, vert_vp)
    # TODO check orientation
    point_f, line6, point_e, line5  = find_tangent_points(mask_center, hull_points, hor_left_vp)

    # Step 2: Compute intersections for corners
    def compute_intersection(line1, line2):
        """Compute the intersection of two lines in homogeneous coordinates."""
        inter = np.cross(line1, line2)
        if inter[2] != 0:
            return inter[:2] / inter[2]
        return None  # Parallel lines

    corner_a = compute_intersection(line1, line6)
    corner_c = compute_intersection(line6, line3)
    corner_b = compute_intersection(line4, line1)
    
    
    
    
    # Step 3: Calculate corner H
    if corner_c is not None and corner_b is not None:
        line_c_h = LineString([corner_c, hor_right_vp])
        line_b_h = LineString([corner_b, hor_left_vp])
        if line_c_h.intersects(line_b_h):
            intersection_point = line_c_h.intersection(line_b_h)
            if intersection_point.geom_type == "Point":
                corner_h = (intersection_point.x, intersection_point.y)
            else:
                corner_h = None  # Handle unexpected intersection types (e.g., LineString)
        else:
            corner_h = None
    else:
        corner_h = None

    epsilon = 1e-1  # Small shift value
    if corner_h is None:
        # Slightly shift corner_c and corner_b if no intersection exists
        shifted_corner_c = (corner_c[0] + epsilon, corner_c[1] + epsilon)
        shifted_corner_b = (corner_b[0] - epsilon, corner_b[1] - epsilon)
        
        line_c_h = LineString([shifted_corner_c, hor_right_vp])
        line_b_h = LineString([shifted_corner_b, hor_left_vp])
        
        if line_c_h.intersects(line_b_h):
            intersection_point = line_c_h.intersection(line_b_h)
            if intersection_point.geom_type == "Point":
                corner_h = (intersection_point.x, intersection_point.y)
                print("Found with epsilon")
            else:
                corner_h = None  # Handle unexpected intersection types
        else:
            corner_h = None

    # if corner_h is None:
        # # Slightly shift corner_c and corner_b if no intersection exists
        # shifted_corner_c = (corner_c[0] + epsilon, corner_c[1] + epsilon)
        # shifted_corner_b = (corner_b[0] - epsilon, corner_b[1] - epsilon)
        
        # line_c_h = LineString([shifted_corner_c, hor_right_vp])
        # line_b_h = LineString([shifted_corner_b, hor_left_vp])
        
        # if line_c_h.intersects(line_b_h):
            # intersection_point = line_c_h.intersection(line_b_h)
            # if intersection_point.geom_type == "Point":
                # corner_h = (intersection_point.x, intersection_point.y)
            # else:
                # corner_h = None  # Handle unexpected intersection types
        # else:
            # corner_h = None
    
    
    corner_c1 = compute_intersection(line2, line3)
    corner_b1 = compute_intersection(line5, line4)

    corner_a11 = get_intersect(hor_left_vp, corner_c1, vert_vp, corner_a)
    corner_a12 = get_intersect(hor_right_vp, corner_b1, vert_vp, corner_a)
    
    corner_a1 = corner_a11 if corner_a11[1] < corner_a12[1] else corner_a12
    corner_c1 = get_intersect(hor_left_vp, corner_a1, vert_vp, corner_c)
    corner_b1 = get_intersect(hor_right_vp, corner_a1, vert_vp, corner_b)

    center = get_intersect(corner_c, corner_b, corner_a, corner_h)
    center1 = get_intersect(corner_c1, corner_b1, vert_vp, center)
    corner_h1 = get_intersect(corner_a1, center1, vert_vp, corner_h)

    # Compile the list of corners (including `corner_a` again to close the loop)
    corners = [corner_a, corner_b, corner_h, corner_c, corner_a]
    corners1 = [corner_a1, corner_b1, corner_h1, corner_c1, corner_a1]
    
    # Debugging and visualization
    if debug:
        plt.imshow(mask, cmap='gray')  # Show the mask in the background
    
        # Function to safely plot points
        def plot_point(corner, color, label, marker='o'):
            if corner is not None and len(corner) == 2:
                plt.scatter(*corner, color=color, marker=marker, label=label)
   
        def draw_line_segment(vp, point, color):
            if vp is not None and point is not None:
                plt.plot([vp[0], point[0]], [vp[1], point[1]], color=color, linestyle="--") 
  
      
        plot_point(corner_a, 'red', "corner a")
        plot_point(corner_b, 'blue', "corner b")
        plot_point(corner_c, 'green', "corner c")
        plot_point(corner_h, 'yellow', "corner h")
        
        plot_point(corner_a1, 'red', "corner a1")
        plot_point(corner_b1, 'blue', "corner b1")
        plot_point(corner_c1, 'green', "corner c1")
        plot_point(corner_h1, 'yellow', "corner h1")
    
        # Plot tangent lines
        draw_line_segment(hor_right_vp, point_a, 'red')
        draw_line_segment(hor_right_vp, point_b, 'red')
        draw_line_segment(vert_vp, point_c, 'blue')
        draw_line_segment(vert_vp, point_d, 'blue')
        draw_line_segment(hor_left_vp, point_e, 'green')
        draw_line_segment(hor_left_vp, point_f, 'green')
        # plot_point(corner_a, 'teal', "corner_a")
        # plot_point(corner_c, 'green', "corner_c")
        # plot_point(corner_b, 'yellow', "corner_b")


    
        # Filter valid corners for visualization
        valid_corners = [corner for corner in corners if corner is not None]
        valid_corners1 = [corner for corner in corners1 if corner is not None]
    
        # Zoom into the mask region
        rows, cols = np.where(mask)
        if rows.size > 0 and cols.size > 0:
            plt.xlim([cols.min() - 10, cols.max() + 10])
            plt.ylim([rows.max() + 10, rows.min() - 10])  # Invert y-axis for correct orientation
    
        # Show legend and final plot
        plt.legend()
        plt.show()
        
    # print([corner_a, corner_b, corner_h, corner_c])
    # print([corner_a1, corner_b1, corner_h1, corner_c1])

    return np.array([corner_a, corner_b, corner_h, corner_c]), np.array([corner_a1, corner_b1, corner_h1, corner_c1])


def find_neighbors_within_mask(labels, e=10):
    # Initialize the dictionary with each unique label
    neighbors_dict = {label: {} for label in range(1, np.max(labels) + 1)}

    # Iterate through each pixel in the labels array
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            current_label = labels[y, x]

            # Only proceed if the current pixel is within the moving mask
            if current_label == 0:
                continue

            # Define neighbor pixel offsets within epsilon distance
            neighbors = {
                'top': (y - e, x),
                'bottom': (y + e, x),
                'left': (y, x - e),
                'right': (y, x + e),
                'top-left': (y - e, x - e),
                'top-right': (y - e, x + e),
                'bottom-left': (y + e, x - e),
                'bottom-right': (y + e, x + e),
            }

            # Iterate through the neighboring directions
            for direction, (ny, nx) in neighbors.items():
                # Skip neighbors that are out of bounds
                if ny < 0 or ny >= labels.shape[
                        0] or nx < 0 or nx >= labels.shape[1]:
                    continue

                # Skip if neighbor is outside the moving mask
                # if moving_mask[ny, nx] == 0:
                # continue

                neighbor_label = labels[ny, nx]
                if neighbor_label == 0:
                    continue

                # Only add if neighbor label is different from current label
                if neighbor_label != current_label:
                    if neighbor_label not in neighbors_dict[current_label]:
                        neighbors_dict[current_label][neighbor_label] = set()

                    # Record the direction of this neighbor relative to current label
                    neighbors_dict[current_label][neighbor_label].add(
                        direction)

    return neighbors_dict

def calc_height(lower_face, upper_face, bottom_face, inv_ipm_matrix, top_inv_ipm_matrix, object_height):
    proj_height = np.linalg.norm(lower_face[0] - upper_face[0])  # height
    proj_bottom = map_points_to_BEV([bottom_face[0]], inv_ipm_matrix)
    proj_top = map_points_to_BEV([proj_bottom[0]], top_inv_ipm_matrix)

    #dist = np.linalg.norm(camera_position_bev - bottom_face[0])
    full_height = np.linalg.norm(proj_bottom[0] - proj_top[0]) # object_height

    height = object_height * proj_height / full_height

    return height

# Utility function to check face overlap in BEV with epsilon parameter
def faces_overlap(face1, face2, region_size, k=0.1):

    # Compute epsilon based on region_size
    epsilon = k * region_size

    if face1 is None or face2 is None: return False
    # Define polygons for each face based on the corner points
    p1 = Polygon([face1[0], face1[1], face1[2], face1[3]])
    p2 = Polygon([face2[0], face2[1], face2[2], face2[3]])

    # Expand the polygons slightly by epsilon
    p1_buffered = p1.buffer(epsilon)
    p2_buffered = p2.buffer(epsilon)

    # Check if the buffered polygons intersect (overlap)
    return p1_buffered.intersects(p2_buffered)

def validate_bottom_face_points(bottom_face, object_length, object_width, epsilon=0.0):
    if bottom_face is None:
        return False

    epsilon=object_length/100
    # Check if any x-coordinate is within the allowed bounds
    x_inside = np.any((bottom_face[:, 0] >= -epsilon) & (bottom_face[:, 0] <= object_length + epsilon))

    # Check if any y-coordinate is within the allowed bounds
    y_inside = np.any((bottom_face[:, 1] >= -epsilon) & (bottom_face[:, 1] <= object_width + epsilon))

    # Return True if any point is inside the valid boundary
    return x_inside and y_inside

def map_points_to_BEV(points, ipm_matrix):
    """
    Maps an array of 2D points to the Bird's Eye View using the IPM matrix.

    Args:
    - points: Array of shape (N, 2) where N is the number of points, each represented as (x, y).
    - ipm_matrix: The 3x3 Inverse Perspective Mapping matrix.

    Returns:
    - bev_points: Transformed points in BEV space.
    """
    # Convert points to NumPy array if it is a list
    if isinstance(points, list):
        points = np.array(points)
    # Convert points to homogeneous coordinates (N, 3) where the third coordinate is 1
    num_points = points.shape[0]
    homogeneous_points = np.hstack([points, np.ones(
        (num_points, 1))])  # Shape: (N, 3)

    # Perform matrix multiplication with IPM matrix (3x3)
    transformed_points = np.dot(homogeneous_points,
                                ipm_matrix.T)  # Shape: (N, 3)

    # Convert back from homogeneous coordinates to 2D by dividing by the third coordinate
    bev_points = transformed_points[:, :2] / transformed_points[:,
                                                                2][:,
                                                                   np.newaxis]

    return bev_points


def process_segments_bfs(
    start_label,
    lower_face,
    upper_face,
    labels,
    z,
    mat,
    object_length,
    object_width,
    object_height,
    inv_ipm_matrix,
    top_inv_ipm_matrix,
    neighbors_dict,
    region_size
):
    # Queue for BFS: stores tuples (current_label, mode, level, z, mat)
    queue = deque()
    queue.append((start_label, "down_to_top", 0, z, mat, lower_face, upper_face))

    segments = {}
    max_depth = 0

    while queue:
        current_label, mode, level, current_z, current_mat, current_lower_face, current_upper_face = queue.popleft()
        max_depth = max(max_depth, level)

        # Skip if already processed
        if current_label in segments and segments[current_label]['used']:
            continue

        # Initialize the current segment if not already done
        if current_label not in segments:
            segment = {
                'level': level,
                'label': current_label,
                'lower_face': current_lower_face,
                'upper_face': current_upper_face,
                'height': None,
                'z': current_z,
                'mat': current_mat,
                'used': False,
            }

            if current_lower_face is None: continue
            # Map lower face (bottom) to BEV and validate
            bottom_face = map_points_to_BEV(current_lower_face, current_mat)
            if not validate_bottom_face_points(bottom_face, object_length, object_width):
                continue  # Skip invalid bottom face

            # Compute height and finalize segment data
            height = calc_height(
                current_lower_face, current_upper_face, bottom_face,
                inv_ipm_matrix, top_inv_ipm_matrix, object_height
            )
            segment.update({'bottom': bottom_face, 'height': height, 'used': True})
            segments[current_label] = segment

        # Process neighbors based on the current mode
        for neighbor_label in neighbors_dict.get(current_label, []):
            if neighbor_label in segments and segments[neighbor_label]['used']:
                continue

            relative_positions = neighbors_dict[current_label][neighbor_label]
            mask = (labels == neighbor_label)
            neighbor_lower_face, neighbor_upper_face = get_projected_box(
                mask, vert_vp, hor_left_vp, hor_right_vp, debug=False
            )

            # Down-to-top processing
            if mode == "down_to_top" and ({'top', 'top-left', 'top-right'} & set(relative_positions))  and faces_overlap(current_upper_face, neighbor_lower_face, region_size):
                pts1 = np.float32(current_upper_face)
                pts2 = np.float32(current_lower_face)
                local_persp = cv2.getPerspectiveTransform(pts1, pts2)
                neighbor_mat = np.dot(current_mat, local_persp)

                # Add neighbor to the queue for BFS
                queue.append((neighbor_label, "down_to_top", level + 1, current_z + height, neighbor_mat, neighbor_lower_face, neighbor_upper_face))

            # Top-to-down processing
            elif mode == "top_to_down" and ({'bottom', 'bottom-left', 'bottom-right'} & set(relative_positions)) and faces_overlap(neighbor_upper_face, current_lower_face, region_size):
                pts1 = np.float32(neighbor_upper_face)
                pts2 = np.float32(neighbor_lower_face)
                local_persp = cv2.getPerspectiveTransform(pts2, pts1)
                neighbor_mat = np.dot(current_mat, local_persp)
                neighbor_bottom_face = map_points_to_BEV(neighbor_upper_face, current_mat)
                neighbor_height = calc_height(
                    neighbor_lower_face, neighbor_upper_face, neighbor_bottom_face,
                    inv_ipm_matrix, top_inv_ipm_matrix, object_height
                )

                # Add neighbor to the queue for BFS
                queue.append((neighbor_label, "top_to_down", level - 1, current_z - neighbor_height, neighbor_mat, neighbor_lower_face, neighbor_upper_face))

            # Down-to-top processing
            elif mode == "same" and ({'left', 'right', 'bottom-left', 'bottom-right', 'top-left', 'top-right'} & set(relative_positions)) and faces_overlap(current_lower_face, neighbor_lower_face, region_size):
                # Add neighbor to the queue for BFS
                queue.append((neighbor_label, "same", level, current_z, current_mat, neighbor_lower_face, neighbor_upper_face))

        # After processing all neighbors in down-to-top mode, switch to top-to-down
        if mode == "down_to_top":
            queue.append((current_label, "top_to_down", level, current_z, current_mat, current_lower_face, current_upper_face))
            
        if mode == "top_to_down":
            queue.append((current_label, "same", level, current_z, current_mat, current_lower_face, current_upper_face))

    return max_depth, segments

def reflect_segment(lower_face, object_width):
    """
    Reflect the lower face of a segment across the object_width/2 axis.
    
    Parameters:
        lower_face (np.array): Array of points representing the lower face of the segment.
        object_width (float): The width of the object to determine the axis of symmetry.

    Returns:
        np.array: Reflected lower face.
    """
    reflected_face = lower_face.copy()
    reflected_face[:, 1] = object_width - lower_face[:, 1]  # Reflect y-coordinates
    return reflected_face

def process_images(reference_image_path, folder_path, output_path, method="otsu", 
                   threshold_value=50, region_size=40, ruler=30, slic_iterations=10):
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Read the reference image (background image)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    if reference_image is None:
        print(f"Error: Could not read the reference image at {reference_image_path}")
        return
        
    print(f"vert_vp : {vert_vp}")
    print(f"hor_left_vp : {hor_left_vp}")
    print(f"hor_right_vp : {hor_right_vp}")

    # Process each image in the folder
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        current_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        color_image = cv2.imread(image_path)  # Read the color image for visualization

        if current_image is None:
            print(f"Warning: Could not read image at {image_path}, skipping.")
            continue

        # Compute absolute difference between reference and current image
        abs_diff = cv2.absdiff(reference_image, current_image)
        abs_diff_blur = cv2.GaussianBlur(abs_diff, (15, 15), 0)

        # Apply thresholding to create a binary object mask
        if method == "adaptive":
            object_mask = cv2.adaptiveThreshold(
                abs_diff_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        elif method == "otsu":
            _, object_mask = cv2.threshold(abs_diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:  # Fixed threshold
            _, object_mask = cv2.threshold(abs_diff_blur, threshold_value, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean the binary mask
        kernel = np.ones((13, 13), np.uint8)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)

        # Fill internal holes using contour hierarchy
        contours, hierarchy = cv2.findContours(object_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            if hierarchy[0][i][3] != -1:  # If contour is a child (hole)
                cv2.drawContours(object_mask, contours, i, 255, thickness=cv2.FILLED)

        # Convert grayscale image to color for SLIC
        color_image_for_slic = color_image.copy() #cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR)

        # Apply SLIC segmentation
        slic = cv2.ximgproc.createSuperpixelSLIC(
            color_image_for_slic,
            algorithm=cv2.ximgproc.MSLIC,
            region_size=region_size,
            ruler=ruler
        )
        slic.iterate(slic_iterations)

        # Retrieve SLIC labels
        labels = slic.getLabels()
        contour_mask = slic.getLabelContourMask(False)

        # Filter SLIC segments based on the object mask
        mask_height, mask_width = object_mask.shape
        for segment_label in np.unique(labels):
            # Get all pixel coordinates belonging to the current segment
            segment_pixels = np.argwhere(labels == segment_label)
            total_pixels_in_segment = len(segment_pixels)
            
            # Count the number of pixels in the segment that intersect with the object mask
            intersecting_pixels = np.sum(
                (0 <= segment_pixels[:, 0]) & (segment_pixels[:, 0] < mask_height) &  # Within height bounds
                (0 <= segment_pixels[:, 1]) & (segment_pixels[:, 1] < mask_width) &  # Within width bounds
                (object_mask[segment_pixels[:, 0], segment_pixels[:, 1]] > 0)         # Intersects with mask
            )
            
            # Check if at least 50% of the segment's area belongs to the mask
            if intersecting_pixels / total_pixels_in_segment < 0.5:
                labels[labels == segment_label] = 0  # Mark as background
                
        labels_combined_color = labels_to_color(labels)
        #cv2.imshow("Segmentation", labels_combined_color)
        
                # Initialize variables
        levels = np.copy(labels)

        height, width = current_image.shape[:2]
        camera_position = np.float32([width/3, height])
        level = 0  # Start from level 0
        levels_image = color_image_for_slic.copy()
        
        while np.any(levels > 0):# and level < 1:  # Continue until all segments are marked
            # Assign a color for the current level
            cv2_color = list(cv_colors)[level % len(cv_colors)].value
        
            # Find the boundary mask
            boundary_mask = find_boundary_with_diff(levels > level)
            boundary_coords = np.argwhere(boundary_mask > 0)  # Extract (y, x) coordinates of the boundary
        
            # Step 3: Calculate the centroid of the boundary points
            if len(boundary_coords) == 0:
                break  # Exit if no more boundaries exist
            mask_center = np.mean(boundary_coords, axis=0)  # Centroid as [y, x]
        
            # Step 4: Find tangent points based on the centroid and camera position
            pos_point, pos_line, neg_point, neg_line = find_tangent_points(
                mask_center[::-1],  # Convert [y, x] to [x, y] for the tangent function
                boundary_coords[:, ::-1],  # Convert all points to [x, y]
                camera_position
            )
        
            # Extract border pixels between the tangent points
            border_pixels = extract_segment_pixels(boundary_mask, pos_point, neg_point)
        
            # Define 8-connectivity offsets
            offsets = [(-1, -1), (-1, 0), (-1, 1),
                       (0, -1), (0, 0), (0, 1),
                       (1, -1), (1, 0), (1, 1)]
        
            # Identify border segments
            border_segments = set()
            for x, y in border_pixels:
                if contour_mask[y, x] > 0:  # Check if the pixel is on the contour
                    # Check neighbors of the current border pixel
                    for dy, dx in offsets:
                        ny, nx = y + dy, x + dx
                        # Skip out-of-bound pixels
                        if ny < 0 or ny >= levels .shape[0] or nx < 0 or nx >= levels .shape[1]:
                            continue

                        segment_label = levels [ny, nx]
                        if segment_label > level:  # Ignore masked-out levels 
                            border_segments.add(segment_label)
        
            # Remove border segments from segmentation and color them
            for label in border_segments:
                levels_image[levels  == label] = cv2_color
                levels [levels  == label] = level  # Remove the labeled segments
        
            # Increment the level for the next color
            level += 1
        
        # Final visualization
        levels_image[0 < contour_mask] = cv_colors.BLACK.value


        # Initialize dictionaries and lists
        segments = {}
        lower_faces = []
        heights = []
        colors = []

        box_image = color_image.copy()
        mask = (levels > 0)
        object_length = 21
        object_width = 7
        object_height = 7
        
        lower_face, upper_face = get_projected_box(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)

        # Validate computed faces
        if lower_face is not None and upper_face is not None:

            pts1 = np.float32(lower_face)
            pts2 = np.float32([(0, 0), (object_length, 0), (object_length, object_width), (0, object_width)])
            pts3 =np.float32(upper_face)

            # Generate the perspective transformation matrices
            ipm_matrix = cv2.getPerspectiveTransform(pts1, pts2)
            inv_ipm_matrix = cv2.getPerspectiveTransform(pts2, pts1)
            top_inv_ipm_matrix = cv2.getPerspectiveTransform(pts1, pts3)



            print("Drawing Cube")
            # Draw 3D cube
            draw_cube(
                box_image,
                lower_face.astype("int"),
                upper_face.astype("int"),
                color=list(cv_colors)[len(cv_colors) - 1].value,
                thickness=14
            )
            # Draw circles on the lower face points
            for i, point in enumerate(lower_face):
                # Define the color for the circle
                color=list(cv_colors)[i % len(cv_colors)].value
        
                # Draw the circle
                cv2.circle(
                    box_image,
                    center=(int(point[0]), int(point[1])),  # Convert to integer coordinates
                    radius=15,  # Circle radius
                    color=color,
                    thickness=-1  # Filled circle
                )
            
            
            neighbors_dict = find_neighbors_within_mask(labels)


            
            max_level = level - 1
    
           
    
            avg_bottom_3d = [
                [x, y, 0] for x, y in [(object_length, 0), (object_length, object_width), (0, object_width), (0, 0)]
            ]
    
            lower_faces.append(np.array(avg_bottom_3d))
            heights.append(object_height)
            cv2_color = list(cv_colors)[len(cv_colors) - 1].value
            plt_color = [c / 255.0 for c in cv2_color[::-1]]  # Normalize for plt
            colors.append(plt_color)
          
            
            for label in np.unique(labels):
                mask = (labels == label)
                # Extract the lower and upper faces for the segment
                lower_face, upper_face = get_projected_box(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)
                    
                # Validate computed faces
                if lower_face is not None and upper_face is not None:
                    # Draw 3D cube
                    draw_cube(
                        box_image,
                        lower_face.astype("int"),
                        upper_face.astype("int"),
                        color=cv_colors.BLACK.value,
                        thickness=2
                    )
    

            ground_level = 0
            segments_in_level = np.unique(labels[(levels == ground_level)])
    
            max_depth = ground_level
    
            # Process all segments using BFS
            for label in segments_in_level:
                # Create a mask for the segment
                mask = (labels == label)
            
                # Extract the lower and upper faces for the segment
                lower_face, upper_face = get_projected_box(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)
            
                # Initialize the BFS process for the segment
                depth, segment_data = process_segments_bfs(
                    start_label=label,               # Label of the starting segment
                    lower_face=lower_face,           # Lower face points of the segment
                    upper_face=upper_face,           # Upper face points of the segment
                    labels=labels,                   # Label matrix for all segments
                    z=0,                             # Starting height
                    mat=ipm_matrix,                  # Initial transformation matrix
                    object_length=object_length,     # Object length
                    object_width=object_width,       # Object width
                    object_height=object_height,     # Object height
                    inv_ipm_matrix=inv_ipm_matrix,   # Inverse Perspective Mapping matrix
                    top_inv_ipm_matrix=top_inv_ipm_matrix,  # Top-view matrix for height calculation
                    neighbors_dict=neighbors_dict,   # Neighbor relationships
                    region_size=region_size          # Overlap sensitivity
                )
            
                # Update the maximum depth of the process
                if max_depth < depth:
                    max_depth = depth
            
                # Merge the processed segment data into the global `segments`
                segments.update(segment_data)
    
            print(f"Max depth: {max_depth}")
            heatmap_colors = interpolate_heatmap_colors(max_depth)
            
            # Collect data for 3D rendering
            for segment in segments.values():
                avg_lower_face = segment['lower_face']
                avg_upper_face = segment['upper_face']
                bottom = segment['bottom']
                
    
                
                #segment['height'] = height
                height = segment['height']
                level = segment['level']
                
                if avg_lower_face is not None and avg_upper_face is not None and bottom is not None:# and level < 3:
                
                    avg_bottom_3d = [
                        [x, y, segment['z']] for x, y in segment['bottom']
                    ]
                    
                    lower_faces.append(np.array(avg_bottom_3d))
                    heights.append(height)
    
                    cv2_color = heatmap_colors[level % max_depth]
    
                    plt_color = [c / 255.0 for c in cv2_color[::-1]]  # Normalize for plt
                    colors.append(plt_color)
                    
                    if False and reflect:
                        # Reflect the lower face and add the reflected data
                        reflected_bottom = reflect_segment(np.array(avg_bottom_3d), object_width)
                        lower_faces.append(reflected_bottom)
                        heights.append(height)  # Symmetric object, height remains the same
                        colors.append(plt_color)  # Use the same color for symmetry
                    
                    # Draw 3D cube
                    draw_cube(
                        box_image,
                        avg_lower_face.astype("int"),
                        avg_upper_face.astype("int"),
                        color=cv2_color,
                        thickness=2
                    )
                    

        # Draw SLIC contours on the original image
        slic_contour_mask = slic.getLabelContourMask(thick_line=False)
        segmented_output = current_image.copy()
        segmented_output[slic_contour_mask > 0] = 255  # Mark contours in white

        # Save the output images
        cv2.imwrite(os.path.join(output_path, f"abs_diff_{filename}"), abs_diff)
        cv2.imwrite(os.path.join(output_path, f"object_mask_{filename}"), object_mask)
        cv2.imwrite(os.path.join(output_path, f"slic_segments_{filename}"), labels_combined_color)
        cv2.imwrite(os.path.join(output_path, f"levels_{filename}"), levels_image)
        cv2.imwrite(os.path.join(output_path, f"box_{filename}"), box_image)

        print(f"Processed {filename}: Results saved to {output_path}")
        
        if len(lower_faces) > 0:# and frame_count == 80:
            draw_cubes_in_3d(lower_faces, heights, colors)
            #draw_cubes_with_bounding_image(box_image, lower_faces, heights, colors)


# Example usage
reference_image_path = "/home/dmytrozhuravlov/cv/data/img/out/reference.JPG"  # Replace with your reference image path
# folder_path = "/home/dmytrozhuravlov/cv/data/img/out/"  # Replace with your folder path
# output_path = "/home/dmytrozhuravlov/cv/data/img/out/out"  # Replace with your output folder path

# reference_image_path = "/home/dmytrozhuravlov/cv/data/img/out/reference.JPG"  # Replace with your reference image path
folder_path = "/home/dmytrozhuravlov/cv/data/img/test/"  # Replace with your folder path
output_path = "/home/dmytrozhuravlov/cv/data/img/test/out"  # Replace with your output folder path

threshold_value = 25  # Adjust threshold value as needed

process_images(reference_image_path, folder_path, output_path, region_size=40//2, ruler=30//2, method="otsu")
