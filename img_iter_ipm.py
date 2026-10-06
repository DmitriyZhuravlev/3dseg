import cv2
import numpy as np
import os
from opengl_drawer import *
#from shapely.geometry import Polygon
#from shapely.geometry import LineString, Point, Polygon
from scipy.spatial import ConvexHull
#import flowiz as fz

from graph import *
from cube import *



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
    EMPTY = None

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
    



# def find_neighbors_within_mask(labels, e=10):
    # # Initialize the dictionary with each unique label
    # neighbors_dict = {label: {} for label in range(1, np.max(labels) + 1)}

    # # Iterate through each pixel in the labels array
    # for y in range(labels.shape[0]):
        # for x in range(labels.shape[1]):
            # current_label = labels[y, x]

            # # Only proceed if the current pixel is within the moving mask
            # if current_label == 0:
                # continue

            # # Define neighbor pixel offsets within epsilon distance
            # neighbors = {
                # 'top': (y - e, x),
                # 'bottom': (y + e, x),
                # 'left': (y, x - e),
                # 'right': (y, x + e),
                # 'top-left': (y - e, x - e),
                # 'top-right': (y - e, x + e),
                # 'bottom-left': (y + e, x - e),
                # 'bottom-right': (y + e, x + e),
            # }

            # # Iterate through the neighboring directions
            # for direction, (ny, nx) in neighbors.items():
                # # Skip neighbors that are out of bounds
                # if ny < 0 or ny >= labels.shape[
                        # 0] or nx < 0 or nx >= labels.shape[1]:
                    # continue

                # # Skip if neighbor is outside the moving mask
                # # if moving_mask[ny, nx] == 0:
                # # continue

                # neighbor_label = labels[ny, nx]
                # if neighbor_label == 0:
                    # continue

                # # Only add if neighbor label is different from current label
                # if neighbor_label != current_label:
                    # if neighbor_label not in neighbors_dict[current_label]:
                        # neighbors_dict[current_label][neighbor_label] = set()

                    # # Record the direction of this neighbor relative to current label
                    # neighbors_dict[current_label][neighbor_label].add(
                        # direction)

    # return neighbors_dict

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
    
# # Utility function to check face overlap in BEV and return intersection size
# def faces_overlap_area(face1, face2, region_size, k=0.1):
    # """
    # Compute the overlap between two faces in BEV and return the intersection size.
    
    # Args:
        # face1 (list): Coordinates of the first face corners [(x1, y1), (x2, y2), ...].
        # face2 (list): Coordinates of the second face corners [(x1, y1), (x2, y2), ...].
        # region_size (float): Approximate size of the region, used for scaling epsilon.
        # k (float): Scaling factor for epsilon (default: 0.1).
        
    # Returns:
        # float: Intersection size (area of overlap) between the two faces.
    # """
    # # Compute epsilon based on region_size
    # epsilon = k * region_size

    # # If either face is None, return no intersection
    # if face1 is None or face2 is None:
        # return 0.0

    # # Define polygons for each face based on the corner points
    # p1 = Polygon([face1[0], face1[1], face1[2], face1[3]])
    # p2 = Polygon([face2[0], face2[1], face2[2], face2[3]])

    # # Expand the polygons slightly by epsilon
    # p1_buffered = p1.buffer(epsilon)
    # p2_buffered = p2.buffer(epsilon)

    # # Check if the buffered polygons intersect
    # if p1_buffered.intersects(p2_buffered):
        # # Calculate the intersection polygon
        # intersection = p1_buffered.intersection(p2_buffered)
        # # Return the area of the intersection
        # return intersection.area

    # # If no intersection, return 0.0
    # return 0.0
    
# Utility function to check face overlap in BEV and return IoU
def faces_overlap_iou(face1, face2, region_size, k=0.1):
    """
    Compute the Intersection over Union (IoU) between two faces in BEV.

    Args:
        face1 (list): Coordinates of the first face corners [(x1, y1), (x2, y2), ...].
        face2 (list): Coordinates of the second face corners [(x1, y1), (x2, y2), ...].
        region_size (float): Approximate size of the region, used for scaling epsilon.
        k (float): Scaling factor for epsilon (default: 0.1).

    Returns:
        float: IoU between the two faces (0.0 if no overlap or invalid inputs).
    """
    # Compute epsilon based on region_size
    epsilon = k * region_size

    # If either face is None, return IoU of 0.0
    if face1 is None or face2 is None:
        return 0.0

    # Define polygons for each face based on the corner points
    p1 = Polygon([face1[0], face1[1], face1[2], face1[3]])
    p2 = Polygon([face2[0], face2[1], face2[2], face2[3]])

    # Expand the polygons slightly by epsilon
    p1_buffered = p1.buffer(epsilon)
    p2_buffered = p2.buffer(epsilon)

    # Check if the buffered polygons intersect
    if p1_buffered.intersects(p2_buffered):
        # Calculate the intersection polygon
        intersection = p1_buffered.intersection(p2_buffered)
        intersection_area = intersection.area

        # Calculate the union polygon
        union = p1_buffered.union(p2_buffered)
        union_area = union.area

        # Return the IoU (intersection area / union area)
        return intersection_area / union_area

    # If no intersection, IoU is 0.0
    return 0.0

def validate_bottom_face_points(bottom_face, object_length, object_width, epsilon=0.0):
    if bottom_face is None:
        return False

    epsilon=object_length/1000
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


def process_segments_bfs_static(
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
            epsilon=object_height/1000
            if -epsilon > current_z + height or object_height + epsilon < current_z + height:
                continue

            segment.update({'bottom': bottom_face, 'height': height, 'used': True})
            segments[current_label] = segment

        # Process neighbors based on the current mode
        for neighbor_label in neighbors_dict.get(current_label, []):
            if neighbor_label in segments and segments[neighbor_label]['used']:
                continue

            relative_positions = neighbors_dict[current_label][neighbor_label]
            mask = (labels == neighbor_label)
            neighbor_lower_face, neighbor_upper_face, _ = get_projected_box(
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
        if 0 != level and current_label in segments and segments[current_label]['used']:
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

            if current_lower_face is None:
                continue

            # Map lower face (bottom) to BEV and validate
            bottom_face = map_points_to_BEV(current_lower_face, current_mat)
            if not validate_bottom_face_points(bottom_face, object_length, object_width):
                continue  # Skip invalid bottom face

            # Compute height and finalize segment data
            height = calc_height(
                current_lower_face, current_upper_face, bottom_face,
                inv_ipm_matrix, top_inv_ipm_matrix, object_height
            )
            epsilon = object_height / 1000
            if -epsilon > current_z + height or object_height + epsilon < current_z + height:
                continue

            segment.update({'bottom': bottom_face, 'height': height, 'used': True})
            segments[current_label] = segment

        # Collect and sort neighbors by overlap area
        neighbors = []
        for neighbor_label in neighbors_dict.get(current_label, []):
            if neighbor_label in segments and segments[neighbor_label]['used']:
                continue

            relative_positions = neighbors_dict[current_label][neighbor_label]
            mask = (labels == neighbor_label)
            neighbor_lower_face, neighbor_upper_face, _ = get_projected_box(
                mask, vert_vp, hor_left_vp, hor_right_vp, debug=False
            )

            if ({'top', 'top-left', 'top-right'} & set(relative_positions)):
                overlap_area = faces_overlap_iou(current_upper_face, neighbor_lower_face, region_size)
                if overlap_area > 0:
                    neighbors.append((neighbor_label, "down_to_top", neighbor_lower_face, neighbor_upper_face, overlap_area))

            elif  ({'bottom', 'bottom-left', 'bottom-right'} & set(relative_positions)):
                overlap_area = faces_overlap_iou(neighbor_upper_face, current_lower_face, region_size)
                if overlap_area > 0:
                    neighbors.append((neighbor_label, "top_to_down", neighbor_lower_face, neighbor_upper_face, overlap_area))

            elif ({'left', 'right', 'bottom-left', 'bottom-right', 'top-left', 'top-right'} & set(relative_positions)):
                overlap_area = faces_overlap_iou(current_lower_face, neighbor_lower_face, region_size)
                if overlap_area > 0:
                    neighbors.append((neighbor_label, "same", neighbor_lower_face, neighbor_upper_face, overlap_area))

        # Sort neighbors by overlap area (descending)
        neighbors.sort(key=lambda x: x[4], reverse=True)

        # Add sorted neighbors to the queue
        for neighbor_label, next_mode, neighbor_lower_face, neighbor_upper_face, _ in neighbors:
            pts1 = np.float32(current_upper_face if next_mode == "down_to_top" else neighbor_upper_face)
            pts2 = np.float32(current_lower_face if next_mode == "down_to_top" else neighbor_lower_face)
            local_persp = cv2.getPerspectiveTransform(pts1, pts2)
            neighbor_mat = np.dot(current_mat, local_persp)

            height = calc_height(
                neighbor_lower_face, neighbor_upper_face, map_points_to_BEV(neighbor_lower_face, current_mat),
                inv_ipm_matrix, top_inv_ipm_matrix, object_height
            )

            queue.append((neighbor_label, next_mode, level + (1 if next_mode == "down_to_top" else -1), current_z + height, neighbor_mat, neighbor_lower_face, neighbor_upper_face))

        # Switch modes if applicable
        # if mode == "down_to_top":
            # queue.append((current_label, "top_to_down", level, current_z, current_mat, current_lower_face, current_upper_face))

        # if mode == "top_to_down":
            # queue.append((current_label, "same", level, current_z, current_mat, current_lower_face, current_upper_face))

    return max_depth, segments


def process_segments_bfs_min(
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
            # Update z if a smaller value is found
            segments[current_label]['z'] = min(segments[current_label]['z'], current_z)
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

            if current_lower_face is None:
                continue

            # Map lower face (bottom) to BEV and validate
            bottom_face = map_points_to_BEV(current_lower_face, current_mat)
            if not validate_bottom_face_points(bottom_face, object_length, object_width):
                continue  # Skip invalid bottom face

            # Compute height and finalize segment data
            height = calc_height(
                current_lower_face, current_upper_face, bottom_face,
                inv_ipm_matrix, top_inv_ipm_matrix, object_height
            )
            epsilon = object_height / 1000
            if -epsilon > current_z + height or object_height + epsilon < current_z + height:
                continue

            segment.update({'bottom': bottom_face, 'height': height, 'used': True})
            segments[current_label] = segment

        # Collect and sort neighbors by overlap area
        neighbors = []
        for neighbor_label in neighbors_dict.get(current_label, []):
            if neighbor_label in segments and segments[neighbor_label]['used']:
                continue

            relative_positions = neighbors_dict[current_label][neighbor_label]
            mask = (labels == neighbor_label)
            neighbor_lower_face, neighbor_upper_face, _ = get_projected_box(
                mask, vert_vp, hor_left_vp, hor_right_vp, debug=False
            )

            if ({'top', 'top-left', 'top-right'} & set(relative_positions)):
                overlap_area = faces_overlap_iou(current_upper_face, neighbor_lower_face, region_size)
                if overlap_area > 0:
                    neighbors.append((neighbor_label, "down_to_top", neighbor_lower_face, neighbor_upper_face, overlap_area))

            elif  ({'bottom', 'bottom-left', 'bottom-right'} & set(relative_positions)):
                overlap_area = faces_overlap_iou(neighbor_upper_face, current_lower_face, region_size)
                if overlap_area > 0:
                    neighbors.append((neighbor_label, "top_to_down", neighbor_lower_face, neighbor_upper_face, overlap_area))

            elif ({'left', 'right', 'bottom-left', 'bottom-right', 'top-left', 'top-right'} & set(relative_positions)):
                overlap_area = faces_overlap_iou(current_lower_face, neighbor_lower_face, region_size)
                if overlap_area > 0:
                    neighbors.append((neighbor_label, "same", neighbor_lower_face, neighbor_upper_face, overlap_area))

        # Sort neighbors by overlap area (descending)
        neighbors.sort(key=lambda x: x[4], reverse=True)

        # Add sorted neighbors to the queue
        for neighbor_label, next_mode, neighbor_lower_face, neighbor_upper_face, _ in neighbors:
            pts1 = np.float32(current_upper_face if next_mode == "down_to_top" else neighbor_upper_face)
            pts2 = np.float32(current_lower_face if next_mode == "down_to_top" else neighbor_lower_face)
            local_persp = cv2.getPerspectiveTransform(pts1, pts2)
            neighbor_mat = np.dot(current_mat, local_persp)

            height = calc_height(
                neighbor_lower_face, neighbor_upper_face, map_points_to_BEV(neighbor_lower_face, current_mat),
                inv_ipm_matrix, top_inv_ipm_matrix, object_height
            )

            # Ensure the z-coordinate is minimal
            minimal_z = min(current_z, current_z + height)
            queue.append((neighbor_label, next_mode, level + (1 if next_mode == "down_to_top" else -1), minimal_z, neighbor_mat, neighbor_lower_face, neighbor_upper_face))

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
    
def find_segment_for_point(labels, contour_mask, point):
    """
    Find the segment label for a given point.
    If the point is on a boundary, find the nearest non-boundary, non-background segment.
    
    Args:
        labels (ndarray): SLIC segmentation labels.
        contour_mask (ndarray): Contour mask from SLIC.
        point (tuple): (x, y) coordinates of the point.
    
    Returns:
        int: The segment label of the point (or nearest segment if on a boundary), ignoring label 0.
    """
    x, y = point

    # Ensure the point is within bounds
    if x < 0 or y < 0 or x >= labels.shape[1] or y >= labels.shape[0]:
        return None  # Out of bounds

    # If not on a boundary and not background, return the segment label
    if not contour_mask[y, x] and labels[y, x] != 0:
        return labels[y, x]

    # If on a boundary or background, find the closest non-boundary, non-background segment
    for radius in range(1, 10):  # Search up to 10 pixels around
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < labels.shape[1] and 0 <= ny < labels.shape[0]:
                    if not contour_mask[ny, nx] and labels[ny, nx] != 0:  # Found a valid segment
                        return labels[ny, nx]

    return None  # No valid segment found


def mark_segment(image, labels, segment_label, color=(0, 0, 255)):
    """
    Mark a specific segment in the given color.
    
    Args:
        image (ndarray): Original image where the segment will be highlighted.
        labels (ndarray): SLIC segmentation labels.
        segment_label (int): The label of the segment to mark.
        color (tuple): RGB color for marking (default: red).
    
    Returns:
        ndarray: Image with the marked segment.
    """
    marked_image = image.copy()
    mask = (labels == segment_label)  # Get all pixels belonging to the segment
    marked_image[mask] = color  # Apply color to the segment

    return marked_image

def resize_to_height(image, target_height):
    height, width = image.shape[:2]
    scaling_factor = target_height / height
    new_width = int(width * scaling_factor)
    resized_image = cv2.resize(image, (new_width, target_height))
    return resized_image


def segment_mask_with_colors_origin(mask, extreme_points):
    """
    Segments the mask into 5 color-coded regions based on the extreme points.
    
    Args:
        mask (numpy.ndarray): Binary mask (0 for background, 255 for foreground).
        extreme_points (list): List of 5 (x, y) extreme points 
                               [left, down-left, down-right, right, up].
    
    Returns:
        numpy.ndarray: Segmented mask with different colors for each region.
    """
    h, w = mask.shape[:2]
    
    # Convert extreme points to numpy array
    extreme_points = np.array(extreme_points, dtype=np.int32)

    # Create an empty color image
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Generate random but distinct colors for each region
    #np.random.seed(42)  # For consistent colors
    colors = np.random.randint(0, 255, (6, 3), dtype=np.uint8)

    # Create a distance map where each pixel is assigned to the nearest extreme point
    y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # Find the closest extreme point for each pixel
    distances = np.zeros((h, w, len(extreme_points)), dtype=np.float32)
    for i, (x, y) in enumerate(extreme_points):
        distances[:, :, i] = np.sqrt((x_indices - x) ** 2 + (y_indices - y) ** 2)

    # Get the index of the closest extreme point for each pixel
    closest_region = np.argmin(distances, axis=2)

    # Assign colors based on the closest extreme point
    for i in range(6):
        colored_mask[closest_region == i] = colors[i]

    # Ensure segmentation follows the mask shape
    colored_mask[mask == 0] = [0, 0, 0]  # Keep background black

    return colored_mask

def compute_distance(x1, y1, x2, y2, metric="chebyshev"):
    """
    Computes the distance between two points using different distance metrics.
    
    Args:
        x1, y1 (int): Coordinates of first point.
        x2, y2 (int): Coordinates of second point.
        metric (str): Distance metric ('euclidean', 'manhattan', 'chebyshev').
        
    Returns:
        float: Computed distance.
    """
    if metric == "euclidean":
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    elif metric == "manhattan":
        return np.abs(x2 - x1) + np.abs(y2 - y1)
    elif metric == "chebyshev":
        return max(np.abs(x2 - x1), np.abs(y2 - y1))
    else:
        raise ValueError("Unsupported distance metric. Choose 'euclidean', 'manhattan', or 'chebyshev'.")

def segment_mask_with_colors(mask, extreme_points, metric="euclidean"):
    """
    Segments the mask into color-coded regions based on extreme points and a chosen distance metric.
    
    Args:
        mask (numpy.ndarray): Binary mask (0 for background, 255 for foreground).
        extreme_points (list): List of (x, y) extreme points 
                               [left, down-left, down-right, right, up].
        metric (str): Distance metric ('euclidean', 'manhattan', 'chebyshev').
    
    Returns:
        numpy.ndarray: Segmented mask with different colors for each region.
    """
    h, w = mask.shape[:2]
    
    # Ensure mask is uint8 for OpenCV operations
    mask = (mask * 255).astype(np.uint8) if mask.dtype == bool else mask

    # Convert extreme points to numpy array
    extreme_points = np.array(extreme_points, dtype=np.int32)

    # Create an empty color image
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Generate distinct colors for each region
    colors = np.random.randint(0, 255, (len(extreme_points), 3), dtype=np.uint8)

    # Compute distance map based on the chosen metric
    y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    distances = np.zeros((h, w, len(extreme_points)), dtype=np.float32)

    for i, (x, y) in enumerate(extreme_points):
        distances[:, :, i] = np.vectorize(compute_distance)(x_indices, y_indices, x, y, metric=metric)

    # Get the index of the closest extreme point for each pixel
    closest_region = np.argmin(distances, axis=2)

    # Assign colors based on the closest extreme point
    for i in range(len(extreme_points)):
        colored_mask[closest_region == i] = colors[i]

    # Ensure segmentation follows the mask shape
    colored_mask[mask == 0] = [0, 0, 0]  # Keep background black

    return colored_mask


def segment_mask_with_colors_5(mask, extrem, metric="euclidean"):
    """
    Segments the mask into color-coded regions based on convex hull defects and center.

    Args:
        mask (numpy.ndarray): Binary mask (0 for background, 255 for foreground).
        metric (str): Distance metric ('euclidean', 'manhattan', 'chebyshev').

    Returns:
        numpy.ndarray: Segmented mask with different colors for each region.
    """
    h, w = mask.shape[:2]

    # Ensure mask is uint8 for OpenCV operations
    mask = (mask * 255).astype(np.uint8) if mask.dtype == bool else mask

    # Find contours and convex hull
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w, 3), dtype=np.uint8)  # Return empty mask if no contour

    contour = max(contours, key=cv2.contourArea)  # Use the largest contour
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    if defects is None or len(defects) < 2:
        return np.zeros((h, w, 3), dtype=np.uint8)  # Return empty mask if no defects

    # Compute center of mass of the mask
    M = cv2.moments(contour)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    center = (center_x, center_y)

    # Get defect points
    defect_points = []
    for i in range(defects.shape[0]):
        _, _, f, _ = defects[i, 0]
        defect_points.append(tuple(contour[f][0]))

    defect_points = sorted(defect_points, key=lambda p: np.arctan2(p[1] - center_y, p[0] - center_x))

    # Create an empty color image
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Generate distinct colors for each segment
    colors = np.random.randint(0, 255, (len(defect_points), 3), dtype=np.uint8)

    # Compute distance map
    y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    distances = np.zeros((h, w, len(defect_points)), dtype=np.float32)

    for i, (x, y) in enumerate(defect_points):
        distances[:, :, i] = np.vectorize(compute_distance)(x_indices, y_indices, x, y, metric=metric)

    # Find the closest defect pair
    for i in range(len(defect_points) - 1):
        d1, d2 = defect_points[i], defect_points[i + 1]
        
        # Compute distance to center
        dist_center = np.vectorize(compute_distance)(x_indices, y_indices, center_x, center_y, metric=metric)
        
        # Compute distance to both defects
        dist1 = np.vectorize(compute_distance)(x_indices, y_indices, d1[0], d1[1], metric=metric)
        dist2 = np.vectorize(compute_distance)(x_indices, y_indices, d2[0], d2[1], metric=metric)
        
        # Define region by comparing distances
        region_mask = (dist1 < dist2) & (dist1 < dist_center)
        colored_mask[region_mask] = colors[i]

    # Ensure segmentation follows the mask shape
    colored_mask[mask == 0] = [0, 0, 0]  # Keep background black

    return colored_mask

def segment_mask_with_defects3(mask, extreme_points):
    """
    Segments the mask into regions based on the closest convex hull defect points.

    Args:
        mask (numpy.ndarray): Binary mask (0 for background, 255 for foreground).
        extreme_points (list): List of 5 (x, y) extreme points 
                               [left, down-left, down-right, right, up].
    
    Returns:
        numpy.ndarray: Segmented mask with different colors for each region.
    """
    h, w = mask.shape[:2]
    
    # Convert extreme points to numpy array
    extreme_points = np.array(extreme_points, dtype=np.int32)

    # Create an empty color image
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Generate distinct colors for each region
    colors = np.random.randint(0, 255, (len(extreme_points), 3), dtype=np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask.copy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return colored_mask  # Return empty mask if no contours found

    # Process the largest contour (assuming it's the main object)
    contour = max(contours, key=cv2.contourArea)

    # Compute convex hull and convexity defects
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    if defects is None:
        return colored_mask  # No convexity defects, return empty mask

    defect_points = []
    for i in range(defects.shape[0]):
        _, _, f, d = defects[i, 0]
        if d > 1000:  # Threshold to filter out small defects
            defect_points.append(tuple(contour[f][0]))

    defect_points = np.array(defect_points, dtype=np.int32)

    # Create a distance map to the nearest extreme or defect point
    y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    all_points = np.vstack((extreme_points, defect_points))  # Combine extreme & defect points

    distances = np.zeros((h, w, len(all_points)), dtype=np.float32)
    for i, (x, y) in enumerate(all_points):
        distances[:, :, i] = np.sqrt((x_indices - x) ** 2 + (y_indices - y) ** 2)

    # Assign each pixel to the nearest extreme/defect point
    closest_region = np.argmin(distances, axis=2)

    # Assign colors based on the closest region
    for i in range(len(all_points)):
        colored_mask[closest_region == i] = colors[i % len(extreme_points)]  # Cycle colors

    # Ensure segmentation follows the mask shape
    colored_mask[mask == 0] = [0, 0, 0]  # Keep background black

    return colored_mask



def segment_mask_with_defects(mask, extreme_points):
    """
    Segments the mask into regions using the closest convexity defect points.

    Args:
        mask (numpy.ndarray): Binary mask (0 for background, 255 for foreground).
        extreme_points (list): List of 5 (x, y) extreme points 
                               [left, down-left, down-right, right, up].

    Returns:
        numpy.ndarray: Segmented mask with different colors for each region.
    """
    h, w = mask.shape[:2]

    # Convert extreme points to numpy array
    extreme_points = np.array(extreme_points, dtype=np.int32)

    # Create an empty color image
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Generate distinct colors
    colors = np.random.randint(0, 255, (len(extreme_points), 3), dtype=np.uint8)

    # Find contour from mask
    mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return colored_mask  # Return empty mask if no contours found

    # Get convex hull and convexity defects
    hull = cv2.convexHull(contours[0], returnPoints=False)
    defects = cv2.convexityDefects(contours[0], hull)

    # Extract defect points between extreme points
    defect_points = []
    if defects is not None:
        for i in range(defects.shape[0]):
            _, _, farthest, _ = defects[i, 0]
            defect_points.append(tuple(contours[0][farthest][0]))

    # Convert defect points to numpy array
    defect_points = np.array(defect_points, dtype=np.int32)

    # Create a distance map where each pixel is assigned to the nearest defect point
    y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # Find the closest defect point for each pixel
    distances = np.zeros((h, w, len(defect_points)), dtype=np.float32)
    for i, (x, y) in enumerate(defect_points):
        distances[:, :, i] = np.sqrt((x_indices - x) ** 2 + (y_indices - y) ** 2)

    # Get the index of the closest defect point for each pixel
    closest_region = np.argmin(distances, axis=2)

    # Assign colors based on the closest defect point
    for i in range(len(defect_points)):
        colored_mask[closest_region == i] = colors[i % len(colors)]

    # Ensure segmentation follows the mask shape
    colored_mask[mask == 0] = [0, 0, 0]  # Keep background black

    return colored_mask


    
def segment_mask(mask, extreme_points):
    """
    Segments the mask into 5 regions based on the nearest extreme point.

    Args:
        mask (numpy.ndarray): Binary mask (0 for background, 255 for foreground).
        extreme_points (list): List of (x, y) extreme points.

    Returns:
        labels (numpy.ndarray): Label array where each pixel is assigned to a region (0-4).
    """
    h, w = mask.shape[:2]

    # Compute distance to each extreme point
    y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    distances = np.zeros((h, w, len(extreme_points)), dtype=np.float32)

    for i, (x, y) in enumerate(extreme_points):
        distances[:, :, i] = np.sqrt((x_indices - x) ** 2 + (y_indices - y) ** 2)

    # Assign each pixel to the nearest extreme point
    labels = np.argmin(distances, axis=2)

    # Ensure segmentation follows the original mask
    labels[mask == 0] = -1  # Background pixels get a special label (-1)

    return labels

def compute_distance_2d(p1, p2):
    """Computes Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def project_point_on_line(p, a, b):
    """
    Projects point `p` onto the line segment `ab` and returns the closest point.

    Args:
        p (array-like): 2D point (x, y).
        a (array-like): 2D endpoint of line segment (x, y).
        b (array-like): 2D endpoint of line segment (x, y).

    Returns:
        projected_point (array): Closest 2D point on segment `ab`.
        t (float): Ratio along the segment (0 = a, 1 = b).
    """
    a, b, p = np.array(a), np.array(b), np.array(p)
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)  # Clamp t between 0 and 1
    projected_point = a + t * ab
    return projected_point, t

def find_closest_3d_point(extrem, pr_boxes, bottoms, heights, excluded = []):
    """
    Finds the closest 2D projection on cube edges and maps it to 3D space.

    Args:
        extrem (list): List of 2D extreme points [(x, y), ...].
        pr_boxes (list): List of projected boxes [(lower_face, upper_face), ...].
        bottoms (list): List of 3D bottom coordinates of the boxes.
        heights (list): Corresponding heights of the 3D boxes.

    Returns:
        list: List of closest 3D points.
    """
    closest_points = []
    closest_points_pr = []
    

    for ext_point in extrem:
        min_dist = float("inf")
        closest_2d = None
        closest_3d = None

        for i, (lower_face, upper_face) in enumerate(pr_boxes):
            bottom_3d = bottoms[i]
            height = heights[i]
            if i in excluded: continue

            # Check closest point on bottom and top face edges
            for j in range(len(lower_face)):
                a, b = lower_face[j], lower_face[(j + 1) % len(lower_face)]  # Bottom edge
                projected_2d, t = project_point_on_line(ext_point, a, b)
                dist = compute_distance_2d(ext_point, projected_2d)

                if dist < min_dist:
                    min_dist = dist
                    closest_2d = projected_2d
                    a_3d, b_3d = bottom_3d[j], bottom_3d[(j + 1) % len(lower_face)]
                    closest_3d = a_3d * (1 - t) + b_3d * t  # Correct interpolation

                a, b = upper_face[j], upper_face[(j + 1) % len(upper_face)]  # Top edge
                projected_2d, t = project_point_on_line(ext_point, a, b)
                dist = compute_distance_2d(ext_point, projected_2d)

                if dist < min_dist:
                    min_dist = dist
                    closest_2d = projected_2d
                    a_3d, b_3d = bottom_3d[j] + np.array([0, 0, height]), bottom_3d[(j + 1) % len(lower_face)] + np.array([0, 0, height])
                    closest_3d = a_3d * (1 - t) + b_3d * t  # Correct interpolation

            # Check closest point on vertical edges
            for j, (lf, uf) in enumerate(zip(lower_face, upper_face)): 
                projected_2d, t = project_point_on_line(ext_point, lf, uf)
                dist = compute_distance_2d(ext_point, projected_2d)

                if dist < min_dist:
                    min_dist = dist
                    closest_2d = projected_2d
                    a_3d, b_3d = bottom_3d[j], bottom_3d[j] + np.array([0, 0, height])
                    closest_3d = a_3d * (1 - t) + b_3d * t  # Correct interpolation

        closest_points.append(closest_3d)
        closest_points_pr.append(closest_2d)

    return closest_points, closest_points_pr

def process_segment_orig(f_2d, f_3d, marked_image, segment_label, ext_labels, nbs_dict, processed_labels, bottoms, heights, colors, level = 0, mark = True):
    """ Process segment and recursively process nbs """
    
    # Check if segment was already processed
    if segment_label in processed_labels:
        return
    #if level > 2: return
    
    processed_labels.add(segment_label)  # Mark as processed
    
    # Get mask for the segment
    mask = (ext_labels == segment_label)
    
    # Extract projected box
    lower_face, upper_face = f_2d
    bottom, top = f_3d
    
    if lower_face is None or upper_face is None:
        return  # Skip if projection failed

    cube_2d = get_projected_cube_faces(lower_face, upper_face)  # Get all 6 faces of the cube
    cube_3d = get_projected_cube_faces(bottom, top)

    print("Original cube 3D faces:", cube_3d)

    # Select relevant faces based on direction
    face_pairs = {
        "left": ("right", "left"),   # Check right of current vs left of nb
        "right": ("left", "right"),  # Check left of current vs right of nb
        "top": ("lower", "upper"),  # Check back of current vs front of nb
        "bottom": ("upper", "lower"),   # Check front of current vs back of nb
    }
    
    # Process nbing segments recursively
    nbs = nbs_dict.get(segment_label, {})
    
    for nb_label, directions in nbs.items():  # Get direction info
        for direction in directions:  # Iterate over each possible direction
            if nb_label not in processed_labels:
                # Check overlap before processing nb
                nb_mask = (ext_labels == nb_label)
                nb_lower, nb_upper, _ = get_projected_box(nb_mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)
                nb_lower_3d, nb_upper_3d = None, None
                if nb_lower is None or nb_upper is None:
                    continue
        
                nb_cube_2d = get_projected_cube_faces(nb_lower, nb_upper)

                if direction in face_pairs:
                    print(f"direction: {direction}")
                    face1, face2 = face_pairs[direction]
        
                    overlap_area = faces_overlap_area(cube_2d[face1], nb_cube_2d[face2])
                    if overlap_area > 0:
                        print(f"Overlap found on {face1} ↔ {face2}, Area = {overlap_area}")
                        if direction == "left":
                            # Convert face points to float32 for transformation
                            pts1 = np.float32(cube_2d[face1])

                            arr = cube_3d[face1].copy()
                            # Step 1: Remove the first column (x) and keep only the second and third columns
                            arr = arr[:, 1:]  # This slices the array from the second column onwards (index 1)
                            # Step 2: Swap the second and third columns (y and z)
                            arr[:, [0, 1]] = arr[:, [1, 0]]

                            pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)

                            # Compute Inverse Perspective Mapping (IPM) matrix
                            ipm = cv2.getPerspectiveTransform(pts1, pts2)
                        
                            # Compute distances
                            dist_2d = dist(cube_2d[face1], cube_2d[face2])
                            dist_3d = dist(cube_3d[face1], cube_3d[face2])
                            
                            print(f"2D distance: {dist_2d}")
                            print(f"3D distance: {dist_3d}")

                            nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                        
                            # Compute neighbor's 3D distance using ratio
                            nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
                        
                            # Transform the 2D points of the neighbor's face into BEV (Bird's Eye View)
                            left_3d = map_points_to_BEV(nb_cube_2d[face2], ipm)  # Output is [z, y]
                            
                            # Swap columns to reorder from [z, y] → [y, z]
                            left_3d = left_3d[:, [1, 0]]
                            
                            # Extend to [x, y, z] by adding the correct x-coordinate
                            left_3d = np.column_stack((np.full(left_3d.shape[0], np.array(cube_3d[face1])[0, 0]), left_3d))
                            # Step 2: Compute right_3d correctly (ensure 2D shape is maintained)
                            right_3d = left_3d.copy()  # Copy left_3d structure
                            right_3d[:, 0] += nb_dist_3d  # Add nb_dist_3d to x-coordinates
                            
                            nb_upper_3d = np.array([left_3d[1], right_3d[1], right_3d[2], left_3d[2]])
                            nb_lower_3d = np.array([left_3d[0], right_3d[0], right_3d[3], left_3d[3]])
                            
                            
                            # print("Right face :", cube_3d[face1])
                            # print("NB Left face: ", left_3d)

                        if direction == "bottom":
                            # Convert face points to float32 for transformation
                            pts1 = np.float32(cube_2d[face1])

                            arr = cube_3d[face1].copy()
                            # Step 1: Remove the first column (x) and keep only the second and third columns
                            arr = arr[:, :2]

                            pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)

                            # Compute Inverse Perspective Mapping (IPM) matrix
                            ipm = cv2.getPerspectiveTransform(pts1, pts2)
                        
                            # Compute distances
                            dist_2d = dist(cube_2d[face1], cube_2d[face2])
                            dist_3d = dist(cube_3d[face1], cube_3d[face2])
                            
                            print(f"2D distance: {dist_2d}")
                            print(f"3D distance: {dist_3d}")

                            nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                        
                            # Compute neighbor's 3D distance using ratio
                            nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
                        
                            # Transform the 2D points of the neighbor's face into BEV (Bird's Eye View)
                            bottom_3d = map_points_to_BEV(nb_cube_2d[face2], ipm)  # Output is [x, y]
                            
                            # Extend to [x, y, z] by adding the correct x-coordinate
                            nb_lower_3d = np.column_stack(( bottom_3d, np.full(bottom_3d.shape[0], np.array(cube_3d[face1])[0, 2])))
                            # Step 2: Compute right_3d correctly (ensure 2D shape is maintained)
                            nb_upper_3d = nb_lower_3d.copy()  # Copy left_3d structure
                            nb_upper_3d[:, 2] += nb_dist_3d  # Add nb_dist_3d to z-coordinates
                            
                    # Mark segment on the image
                    color = list(cv_colors)[(level+1) % (len(cv_colors) - 1)].value
                    if mark:
                        marked_image[nb_mask] = color
                    if nb_lower is not None and nb_upper is not None:
                        # Draw 3D cube
                        draw_cube(
                            marked_image,
                            nb_lower.astype("int"),
                            nb_upper.astype("int"),
                            color=color,
                            thickness=4
                    )
                if nb_lower_3d is not None and nb_upper_3d is not None:
                    bottoms.append(nb_lower_3d)
                    height = (nb_upper_3d - nb_lower_3d)[0, 2]  # [row_index, z_index]
                    heights.append(height)
        
                    cv2_color = list(cv_colors)[(level + 1) % (len(cv_colors) - 1)].value
                    plt_color = [c / 255.0 for c in cv2_color[::-1]]  # Normalize for plt
                    colors.append(plt_color)
                    # Recursive call to process nb
                    process_segment((nb_lower, nb_upper), (nb_lower_3d, nb_upper_3d), marked_image, nb_label, ext_labels, nbs_dict, processed_labels, bottoms, heights, colors, level+1, mark)
                    
def compute_validity_score(lower_face_3d, upper_face_3d, xmax, ymax, zmax):
    """ Compute the fraction of the cube within the valid bounding box """
    
    # Get min/max coordinates from the set of four points
    min_x, min_y, min_z = np.min(lower_face_3d, axis=0)
    max_x, max_y, max_z = np.max(upper_face_3d, axis=0)

    # Compute full cube volume
    full_volume = max(1e-6, (max_x - min_x) * (max_y - min_y) * (max_z - min_z))  # Avoid division by zero

    # Compute constrained bounding box inside the valid region
    clipped_min_x = np.clip(min_x, 0, xmax)
    clipped_max_x = np.clip(max_x, 0, xmax)
    clipped_min_y = np.clip(min_y, 0, ymax)
    clipped_max_y = np.clip(max_y, 0, ymax)
    clipped_min_z = np.clip(min_z, 0, zmax)
    clipped_max_z = np.clip(max_z, 0, zmax)

    inside_volume = max(0, (clipped_max_x - clipped_min_x) * (clipped_max_y - clipped_min_y) * (clipped_max_z - clipped_min_z))

    return inside_volume / full_volume  # Ratio of the part within the box

def process_segment(xmax, ymax, zmax, f_2d, f_3d, marked_image, segment_label, ext_labels, nbs_dict, processed_labels, bottoms, heights, colors, level=0, mark=True, max_level = 10):
    """ Process segment and recursively process nbs """
    
    if level > max_level: return
    # Check if segment was already processed
    # if segment_label in processed_labels:
        # return
    
    # processed_labels[segment_label] = {'iou': 0.0, 'lower_face_3d': None, 'upper_face_3d': None}  # Initialize storage
    
    # # Get mask for the segment
    # mask = (ext_labels == segment_label)
    
    # Extract projected box
    lower_face, upper_face = f_2d
    bottom, top = f_3d
    
    if lower_face is None or upper_face is None:
        return  # Skip if projection failed

    if bottom is None or top is None:
        return  # Skip if projection failed
        

    cube_2d = get_projected_cube_faces(lower_face, upper_face)  # Get all 6 faces of the cube
    cube_3d = get_projected_cube_faces(bottom, top)

    # Process neighboring segments recursively
    nbs = nbs_dict.get(segment_label, {})
    
    # Select relevant faces based on direction
    face_pairs = {
        "left": ("right", "left"),   # Check right of current vs left of nb
        "right": ("left", "right"),  # Check left of current vs right of nb
        "top": ("lower", "upper"),  # Check back of current vs front of nb
        "bottom": ("upper", "lower"),   # Check front of current vs back of nb
    }
    
    for nb_label, directions in nbs.items():
        for direction in directions:
            print(f"direction: {direction}")

            nb_mask = (ext_labels == nb_label)
            nb_lower, nb_upper, _ = get_projected_box(nb_mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)
            nb_lower_3d, nb_upper_3d = None, None
            if nb_lower is None or nb_upper is None:
                continue
                
            draw_cube(
            marked_image,
            nb_lower.astype("int"),
            nb_upper.astype("int"),
            color=cv_colors.BLACK.value,
            thickness=2
            )
            
            nb_cube_2d = get_projected_cube_faces(nb_lower, nb_upper)
            
            if direction in face_pairs:
                face1, face2 = face_pairs[direction]
                iou = faces_iou(cube_2d[face1], nb_cube_2d[face2])
                print(f"IoU: {iou}")
                
                #if iou > 0 and (nb_label not in processed_labels or iou > processed_labels[nb_label]['iou']):# or level < processed_labels[nb_label]['level']):
                if iou > 0 and (nb_label not in processed_labels or level < processed_labels[nb_label]['level']):
                    
                    if direction == "left":
                        # Convert face points to float32 for transformation
                        pts1 = np.float32(cube_2d[face1])

                        arr = cube_3d[face1].copy()
                        # Step 1: Remove the first column (x) and keep only the second and third columns
                        arr = arr[:, 1:]  # This slices the array from the second column onwards (index 1)
                        # Step 2: Swap the second and third columns (y and z)
                        arr[:, [0, 1]] = arr[:, [1, 0]]

                        pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)

                        # Compute Inverse Perspective Mapping (IPM) matrix
                        ipm = cv2.getPerspectiveTransform(pts1, pts2)
                    
                        # Compute distances
                        dist_2d = dist(cube_2d[face1], cube_2d[face2])
                        dist_3d = dist(cube_3d[face1], cube_3d[face2])
                        
                        print(f"2D distance: {dist_2d}")
                        print(f"3D distance: {dist_3d}")

                        nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                    
                        # Compute neighbor's 3D distance using ratio
                        nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
                    
                        # Transform the 2D points of the neighbor's face into BEV (Bird's Eye View)
                        left_3d = map_points_to_BEV(nb_cube_2d[face2], ipm)  # Output is [z, y]
                        
                        # Swap columns to reorder from [z, y] → [y, z]
                        left_3d = left_3d[:, [1, 0]]
                        
                        # Extend to [x, y, z] by adding the correct x-coordinate
                        left_3d = np.column_stack((np.full(left_3d.shape[0], np.array(cube_3d[face1])[0, 0]), left_3d))
                        # Step 2: Compute right_3d correctly (ensure 2D shape is maintained)
                        right_3d = left_3d.copy()  # Copy left_3d structure
                        right_3d[:, 0] += nb_dist_3d  # Add nb_dist_3d to x-coordinates
                        
                        nb_upper_3d = np.array([left_3d[1], right_3d[1], right_3d[2], left_3d[2]])
                        nb_lower_3d = np.array([left_3d[0], right_3d[0], right_3d[3], left_3d[3]])
                        
                        
                        # print("Right face :", cube_3d[face1])
                        # print("NB Left face: ", left_3d)
                        
                    if direction == "right":
                        # Convert face points to float32 for transformation
                        pts1 = np.float32(cube_2d[face1])

                        arr = cube_3d[face1].copy()
                        # Step 1: Remove the first column (x) and keep only the second and third columns
                        arr = arr[:, 1:]  # This slices the array from the second column onwards (index 1)
                        # Step 2: Swap the second and third columns (y and z)
                        arr[:, [0, 1]] = arr[:, [1, 0]]

                        pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)

                        # Compute Inverse Perspective Mapping (IPM) matrix
                        ipm = cv2.getPerspectiveTransform(pts1, pts2)
                    
                        # Compute distances
                        dist_2d = dist(cube_2d[face1], cube_2d[face2])
                        dist_3d = dist(cube_3d[face1], cube_3d[face2])
                        
                        print(f"2D distance: {dist_2d}")
                        print(f"3D distance: {dist_3d}")

                        nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                    
                        # Compute neighbor's 3D distance using ratio
                        nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
                    
                        # Transform the 2D points of the neighbor's face into BEV (Bird's Eye View)
                        right_3d = map_points_to_BEV(nb_cube_2d[face2], ipm)  # Output is [z, y]
                        
                        # Swap columns to reorder from [z, y] → [y, z]
                        right_3d = right_3d[:, [1, 0]]
                        
                        # Extend to [x, y, z] by adding the correct x-coordinate
                        right_3d = np.column_stack((np.full(right_3d.shape[0], np.array(cube_3d[face1])[0, 0]), right_3d))
                        # Step 2: Compute right_3d correctly (ensure 2D shape is maintained)
                        left_3d = right_3d.copy()  # Copy left_3d structure
                        left_3d[:, 0] -= nb_dist_3d  # Add nb_dist_3d to x-coordinates
                        
                        nb_upper_3d = np.array([left_3d[1], right_3d[1], right_3d[2], left_3d[2]])
                        nb_lower_3d = np.array([left_3d[0], right_3d[0], right_3d[3], left_3d[3]])
                        
                        
                        # print("Right face :", cube_3d[face1])
                        # print("NB Left face: ", left_3d)

                    elif direction == "bottom":
                        # Convert face points to float32 for transformation
                        pts1 = np.float32(cube_2d[face1])

                        arr = cube_3d[face1].copy()
                        # Step 1: Remove the first column (x) and keep only the second and third columns
                        arr = arr[:, :2]

                        pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)

                        # Compute Inverse Perspective Mapping (IPM) matrix
                        ipm = cv2.getPerspectiveTransform(pts1, pts2)
                    
                        # Compute distances
                        dist_2d = dist(cube_2d[face1], cube_2d[face2])
                        dist_3d = dist(cube_3d[face1], cube_3d[face2])
                        
                        print(f"2D distance: {dist_2d}")
                        print(f"3D distance: {dist_3d}")

                        nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                    
                        # Compute neighbor's 3D distance using ratio
                        nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
                    
                        # Transform the 2D points of the neighbor's face into BEV (Bird's Eye View)
                        bottom_3d = map_points_to_BEV(nb_cube_2d[face2], ipm)  # Output is [x, y]
                        
                        # Extend to [x, y, z] by adding the correct x-coordinate
                        nb_lower_3d = np.column_stack(( bottom_3d, np.full(bottom_3d.shape[0], np.array(cube_3d[face1])[0, 2])))
                        # Step 2: Compute right_3d correctly (ensure 2D shape is maintained)
                        nb_upper_3d = nb_lower_3d.copy()  # Copy left_3d structure
                        nb_upper_3d[:, 2] += nb_dist_3d  # Add nb_dist_3d to z-coordinates
                        
                    elif direction == "top":
                        # Convert face points to float32 for transformation
                        pts1 = np.float32(cube_2d[face1])

                        arr = cube_3d[face1].copy()
                        # Step 1: Remove the first column (x) and keep only the second and third columns
                        arr = arr[:, :2]

                        pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)

                        # Compute Inverse Perspective Mapping (IPM) matrix
                        ipm = cv2.getPerspectiveTransform(pts1, pts2)
                    
                        # Compute distances
                        dist_2d = dist(cube_2d[face1], cube_2d[face2])
                        dist_3d = dist(cube_3d[face1], cube_3d[face2])
                        
                        print(f"2D distance: {dist_2d}")
                        print(f"3D distance: {dist_3d}")

                        nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                    
                        # Compute neighbor's 3D distance using ratio
                        nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
                    
                        # Transform the 2D points of the neighbor's face into BEV (Bird's Eye View)
                        top_3d = map_points_to_BEV(nb_cube_2d[face2], ipm)  # Output is [x, y]
                        
                        # Extend to [x, y, z] by adding the correct x-coordinate
                        nb_upper_3d = np.column_stack(( top_3d, np.full(top_3d.shape[0], np.array(cube_3d[face1])[0, 2])))
                        # Step 2: Compute right_3d correctly (ensure 2D shape is maintained)
                        nb_lower_3d = nb_upper_3d.copy()  # Copy left_3d structure
                        nb_lower_3d[:, 2] -= nb_dist_3d  # Add nb_dist_3d to z-coordinates

                    else: continue

                    score = compute_validity_score(nb_lower_3d, nb_upper_3d, xmax, ymax, zmax)
                    if score < 0.8: continue
                    print(f"score: {score}")

                    if nb_label not in processed_labels:
                        processed_labels[nb_label] = {'iou': iou, "score": score, 'level': level, 'lower_face': nb_lower, 'upper_face': nb_upper,'lower_face_3d': nb_lower_3d, 'upper_face_3d': nb_upper_3d}
                        process_segment(xmax, ymax, zmax, (nb_lower, nb_upper), (nb_lower_3d, nb_upper_3d), marked_image, nb_label, ext_labels, nbs_dict, processed_labels, bottoms, heights, colors, level+1, mark)
                    else: #if iou + score > processed_labels[nb_label]['iou'] + processed_labels[nb_label]['score']:
                        processed_labels[nb_label].update({
                            'iou': iou,
                            "score": score,
                            'level': level,
                            'lower_face': nb_lower,
                            'upper_face': nb_upper,
                            'lower_face_3d': nb_lower_3d,
                            'upper_face_3d': nb_upper_3d
                        })
                        process_segment(xmax, ymax, zmax, (nb_lower, nb_upper), (nb_lower_3d, nb_upper_3d), marked_image, nb_label, ext_labels, nbs_dict, processed_labels, bottoms, heights, colors, level+1, mark)

        # if nb_label in processed_labels and processed_labels[nb_label]['lower_face_3d'] is not None and processed_labels[nb_label]['upper_face_3d'] is not None:
            
            # bottoms.append(processed_labels[nb_label]['lower_face_3d'])
            # height = (processed_labels[nb_label]['upper_face_3d'] - processed_labels[nb_label]['lower_face_3d'])[0, 2]
            # heights.append(height)
            
            # cv2_color = list(cv_colors)[(level + 1) % (len(cv_colors) - 1)].value
            # plt_color = [c / 255.0 for c in cv2_color[::-1]]  # Normalize for plt
            # colors.append(plt_color)
            
            # draw_cube(
                # marked_image,
                # processed_labels[nb_label]['lower_face'].astype("int"),
                # processed_labels[nb_label]['upper_face'].astype("int"),
                # color=cv2_color,
                # thickness=4
            # )

def compute_average_processed_labels(processed_labels_list):
    averaged_processed_labels = [{} for _ in range(len(processed_labels_list))]  # Same structure

    label_data = {}

    # Collect values from all segments
    for segment_labels in processed_labels_list:
        for label, data in segment_labels.items():
            if label not in label_data:
                label_data[label] = {
                    'iou': [],
                    'score': [],
                    'lower_faces': [],
                    'upper_faces': [],
                    'lower_faces_3d': [],
                    'upper_faces_3d': []
                }
            
            label_data[label]['iou'].append(data['iou'])
            label_data[label]['score'].append(data['score'])

            if data['lower_face'] is not None:
                label_data[label]['lower_faces'].append(data['lower_face'])
            if data['upper_face'] is not None:
                label_data[label]['upper_faces'].append(data['upper_face'])

            if data['lower_face_3d'] is not None:
                label_data[label]['lower_faces_3d'].append(data['lower_face_3d'])
            if data['upper_face_3d'] is not None:
                label_data[label]['upper_faces_3d'].append(data['upper_face_3d'])

    # Compute averages and populate new dictionary
    for segment_index in range(len(processed_labels_list)):
        for label, values in label_data.items():
            avg_iou = np.mean(values['iou'])
            avg_score = np.mean(values['score'])

            # Compute 2D average if available
            avg_lower_face = np.mean(values['lower_faces'], axis=0) if values['lower_faces'] else None
            avg_upper_face = np.mean(values['upper_faces'], axis=0) if values['upper_faces'] else None

            # Compute 3D averages (skip if empty)
            avg_lower_face_3d = np.mean(values['lower_faces_3d'], axis=0) if values['lower_faces_3d'] else None
            avg_upper_face_3d = np.mean(values['upper_faces_3d'], axis=0) if values['upper_faces_3d'] else None

            # Compute height if possible
            height = (avg_upper_face_3d[0, 2] - avg_lower_face_3d[0, 2]) if avg_lower_face_3d is not None and avg_upper_face_3d is not None else None

            averaged_processed_labels[segment_index][label] = {
                'iou': avg_iou,
                'score': avg_score,
                'lower_face': avg_lower_face,  # Now correctly computed
                'upper_face': avg_upper_face,  # Now correctly computed
                'lower_face_3d': avg_lower_face_3d,
                'upper_face_3d': np.array(avg_lower_face_3d) + np.array([0, 0, height]) if height is not None else None
            }

    return averaged_processed_labels


def bottoms_to_centers(bottoms, heights):
    """
    Converts bottom face 3D points to cube centers.

    :param bottoms: List of bottom faces (each is an array of 4 (x, y, z) points).
    :param heights: List of heights for each cube.
    :return: List of computed 3D centers [(x, y, z)].
    """
    centers = []

    for bottom, height in zip(bottoms, heights):
        # Compute the average of the 4 bottom face points
        avg_bottom = np.mean(bottom, axis=0)  # (x, y, z)

        # Compute center by adding half the height in Z direction
        center = avg_bottom + np.array([0, 0, height / 2])
        centers.append(center)


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

        # Convert the binary mask to 3 channels to match the color image
        object_mask_3c = cv2.cvtColor(object_mask, cv2.COLOR_GRAY2BGR)
    
        # Apply the mask to the color image
        background_removed = cv2.bitwise_and(color_image, object_mask_3c)
        
        # Save or display the result
        #output_path = os.path.join(output_path, f"removed_{filename}")
        cv2.imwrite(os.path.join(output_path, f"removed_{filename}"), background_removed)

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
        level = 1  # Start from level 0
        levels_image = color_image_for_slic.copy()
        
        while np.any(levels > 0):# and level < 1:  # Continue until all segments are marked
            # Assign a color for the current level
            cv2_color = list(cv_colors)[level % (len(cv_colors)-1)].value
        
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
        bottoms = []
        pr_boxes = []
        heights = []
        colors = []

        box_image = color_image.copy()
        full_mask = (labels > 0)
        ext_labels = labels #segment_mask(mask, extrem)

        g_xmin = xmin = 0
        g_xmax = xmax = 21
        g_ymin = ymin = 0
        g_object_width = g_ymax = ymax = 7
        g_zmin = zmin = 0
        g_zmax = zmax = 7
        
        marked_image = box_image.copy()
        lower_face, upper_face, extrem = get_projected_box(full_mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)

        
        # faces = get_projected_cube_faces(lower_face, upper_face)
        # #draw_projected_cube(faces)
        # result_image = draw_projected_cube_on_image(marked_image, faces)

        # # Show the result
        # cv2.imshow("Projected Cube", result_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        down = extrem[0] # RED d
        top = extrem[1] # GREEN u
        left = extrem[5] # MINT r
        left_v = extrem[2] # BLUE l
        right_v = extrem[3] # ORANGE r
        right_u = extrem[4] # PURPLE l

        extrem = [down, top, left_v, right_v, left, right_u]

        count = 0
        
        reflect = False#True
        mark = True#False
        
        neighbors_dict = find_neighbors(labels)
        
        # processed_labels = {} #([segment_label])
        
        # num_segments = total_segments  # Define the number of segments
        processed_labels_list = [{} for _ in range(len(extrem))]  # List of dictionaries

        while np.any(full_mask) and count < 1:
            count += 1
            print(f"iteration: {count}")

            object_length = xmax - xmin
            object_width = ymax - ymin
            object_height = zmax - zmin
    
            print(f"xmin: {xmin}")
            print(f"xmax: {xmax}")
            print(f"ymin: {ymin}")
            print(f"ymax: {ymax}")
            print(f"zmin: {zmin}")
            print(f"zmax: {zmax}")
    
    
            print(f"object length: {object_length}")
            print(f"object width: {object_width}")
            print(f"object height: {object_height}")
    
            # Validate computed faces
            if lower_face is not None and upper_face is not None:
                # top down
                pts1 = np.float32(lower_face)
                pts2 = np.float32([(xmin, ymin), (xmin + object_length, ymin), (xmin + object_length, ymin + object_width), (xmin, ymin + object_width)])
                pts3 = np.float32(upper_face)
    
                # Generate the perspective transformation matrices
                ipm_matrix = cv2.getPerspectiveTransform(pts1, pts2)
                inv_ipm_matrix = cv2.getPerspectiveTransform(pts2, pts1)
                top_inv_ipm_matrix = cv2.getPerspectiveTransform(pts1, pts3)
                top_ipm_matrix = cv2.getPerspectiveTransform(pts3, pts2)
                
                # left face
                pts1 = np.float32([lower_face[0], upper_face[0], upper_face[3], lower_face[3]])
                pts2 = np.float32([(ymin, zmin), (ymin, zmin + object_height), (ymin + object_width, zmin + object_height), (ymin + object_width, zmin)])
                pts3 = np.float32([lower_face[1], upper_face[1], upper_face[2], lower_face[2]])
    
                ipm_left = cv2.getPerspectiveTransform(pts1, pts2)
                inv_ipm_left = cv2.getPerspectiveTransform(pts2, pts1)
                top_inv_ipm_left = cv2.getPerspectiveTransform(pts1, pts3)
                top_ipm_left = cv2.getPerspectiveTransform(pts3, pts2)

                #main cube
                bottom = np.array([[0, 0, 0], [g_xmax, 0, 0], [g_xmax, g_ymax, 0], [0, g_ymax, 0]])
                top = np.array([[0, 0, g_zmax], [g_xmax, 0, g_zmax], [g_xmax, g_ymax, g_zmax], [0, g_ymax, g_zmax]])
                cube_2d = get_projected_cube_faces(lower_face, upper_face)  # Get all 6 faces of the cube
                cube_3d = get_projected_cube_faces(bottom, top)
            
                # Process neighboring segments recursively
                #nbs = nbs_dict.get(segment_label, {})
                
                # Select relevant faces based on direction
                face_pairs = {
                    "left": ("right", "left"),   # Check right of current vs left of nb
                    "right": ("left", "right"),  # Check left of current vs right of nb
                    "top": ("lower", "upper"),  # Check back of current vs front of nb
                    "bottom": ("upper", "lower"),   # Check front of current vs back of nb
                }
                
                
                segment_index = 1
                segment_label = find_segment_for_point(labels, contour_mask, extrem[segment_index])
                mask = (ext_labels == segment_label)  # Get all pixels belonging to the segment
                if mark: marked_image[mask] = list(cv_colors)[(segment_index) % (len(cv_colors)-1)].value
    
                ext_labels[ext_labels == segment_label] = 0
                #mask = (ext_labels == segment_index).astype(np.uint8) * 255 
                # Extract the lower and upper faces for the segment
                lower_face, upper_face, _ = get_projected_box(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)
                if lower_face is None or upper_face is None: continue
    
                # top_2d = map_points_to_BEV([top], top_ipm_matrix)[0]
                # top_3d = [top_2d[0], top_2d[1], object_height]
                ##ext_3d.append(top_3d)
    
                pr_upper_face = map_points_to_BEV(upper_face, top_ipm_matrix)
                height = calc_height(lower_face, upper_face, pr_upper_face, inv_ipm_matrix, top_inv_ipm_matrix, object_height)
                z = zmin + object_height - height
    
                #avg_bottom_3d = [[max(0, min(x, object_length)), max(0, min(y, object_width)), z] for x, y in pr_upper_face]
                avg_bottom_3d = [[x, y, z] for x, y in pr_upper_face]
                avg_bottom_3d = np.clip(avg_bottom_3d, [0, 0, 0], [g_xmax, g_ymax, g_zmax])
                avg_bottom_3d = np.clip(avg_bottom_3d, [0, 0, 0], [xmax, ymax, zmax])
                bottoms.append(np.array(avg_bottom_3d))
                pr_boxes.append((lower_face, upper_face))
                heights.append(height)
    
                cv2_color = list(cv_colors)[(segment_index + count) % (len(cv_colors)-1)].value
                plt_color = [c / 255.0 for c in cv2_color[::-1]]  # Normalize for plt
                colors.append(plt_color)
                
    
                # Validate computed faces
                if lower_face is not None and upper_face is not None:
                    # Draw 3D cube
                    draw_cube(
                        marked_image,
                        lower_face.astype("int"),
                        upper_face.astype("int"),
                        color=list(cv_colors)[(segment_index + count) % (len(cv_colors)-1)].value,
                        thickness=4
                    )
                    
                f_2d = (lower_face, upper_face)
                f_3d = (np.array(avg_bottom_3d), np.array(avg_bottom_3d) + np.array([0, 0, height]))

                for i in range(len(extrem)):
                    processed_labels_list[i][segment_label] = {'iou': 1.0, 'score': 1.0, 'level': 0, 'lower_face': lower_face, 'upper_face': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height])}  # Initialize storage
                process_segment(g_xmax, g_ymax, g_zmax, f_2d, f_3d, marked_image, segment_label, ext_labels, neighbors_dict, processed_labels_list[segment_index], bottoms, heights, colors, mark)
    

                segment_index = 0
                segment_label = find_segment_for_point(labels, contour_mask, extrem[segment_index])
                mask = (ext_labels == segment_label)  # Get all pixels belonging to the segment
                if mark: marked_image[mask] = list(cv_colors)[(segment_index) % (len(cv_colors)-1)].value
                #ext_labels[ext_labels == segment_label] = 0
                #mask = (ext_labels == segment_index).astype(np.uint8) * 255  
                # Extract the lower and upper faces for the segment
                lower_face, upper_face, _ = get_projected_box(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)
                if lower_face is None or upper_face is None: continue
    
                pr_lower_face = map_points_to_BEV(lower_face, ipm_matrix)
                height = calc_height(lower_face, upper_face, pr_lower_face, inv_ipm_matrix, top_inv_ipm_matrix, object_height)
                z = zmin
    
                avg_bottom_3d = [[x, y, z] for x, y in pr_lower_face]
                
                # down_2d = map_points_to_BEV([down], ipm_matrix)[0]
                # down_3d = [down_2d[0], down_2d[1], z]
                ##ext_3d.append(down_3d)
                
                avg_bottom_3d = np.clip(avg_bottom_3d, [0, 0, 0], [g_xmax, g_ymax, g_zmax])
                bottoms.append(np.array(avg_bottom_3d))
                pr_boxes.append((lower_face, upper_face))
                heights.append(height)
    
                cv2_color = list(cv_colors)[(0) % (len(cv_colors)-1)].value
                plt_color = [c / 255.0 for c in cv2_color[::-1]]  # Normalize for plt
                colors.append(plt_color)
                
    
                # Validate computed faces
                if lower_face is not None and upper_face is not None:
                    # Draw 3D cube
                    draw_cube(
                        marked_image,
                        lower_face.astype("int"),
                        upper_face.astype("int"),
                        color=list(cv_colors)[(segment_index + count) % (len(cv_colors)-1)].value,
                        thickness=4
                    )

                f_2d = (lower_face, upper_face)
                f_3d = (np.array(avg_bottom_3d), np.array(avg_bottom_3d) + np.array([0, 0, height]))

                for i in range(len(extrem)):
                    processed_labels_list[i][segment_label] = {'iou': 1.0, 'score': 1.0, 'level': 0, 'lower_face': lower_face, 'upper_face': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height])}  # Initialize storage
                process_segment(g_xmax, g_ymax, g_zmax, f_2d, f_3d, marked_image, segment_label, ext_labels, neighbors_dict, processed_labels_list[segment_index], bottoms, heights, colors, mark)
    

                segment_index = 4
                segment_label = find_segment_for_point(labels, contour_mask, extrem[segment_index])
                mask = (ext_labels == segment_label)  # Get all pixels belonging to the segment
                marked_image[mask] = list(cv_colors)[segment_index].value
                ext_labels[ext_labels == segment_label] = 0
                #mask = (ext_labels == segment_index).astype(np.uint8) * 255 
                # Extract the lower and upper faces for the segment
                lower_face, upper_face, _ = get_projected_box(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)
                if lower_face is None or upper_face is None: continue
    
                pr_lower_face = map_points_to_BEV(lower_face, ipm_matrix)
                height = calc_height(lower_face, upper_face, pr_lower_face, inv_ipm_matrix, top_inv_ipm_matrix, object_height)
                z = 0
    
                avg_bottom_3d = [[x, y, z] for x, y in pr_lower_face]
                
                left_2d = map_points_to_BEV([left], ipm_matrix)[0]
                left_3d = [left_2d[0], left_2d[1], z]
                ##ext_3d.append(left_3d)
                
                
                bottoms.append(np.array(avg_bottom_3d))
                pr_boxes.append((lower_face, upper_face))
                heights.append(height)
    
                cv2_color = cv_colors.ORANGE.value
                plt_color = [c / 255.0 for c in cv2_color[::-1]]  # Normalize for plt
                colors.append(plt_color)
                
                    
                # Validate computed faces
                if lower_face is not None and upper_face is not None:
                    # Draw 3D cube
                    draw_cube(
                        marked_image,
                        lower_face.astype("int"),
                        upper_face.astype("int"),
                        color=cv_colors.ORANGE.value,
                        thickness=4
                    )
                    
                f_2d = (lower_face, upper_face)
                f_3d = (np.array(avg_bottom_3d), np.array(avg_bottom_3d) + np.array([0, 0, height]))
                # Start processing from segment_label
                #processed_labels = {} #([segment_label])
                for i in range(len(extrem)):
                    processed_labels_list[i][segment_label] = {'iou': 1.0, 'score': 1.0, 'level': 0, 'lower_face': lower_face, 'upper_face': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height])}  # Initialize storage
                process_segment(g_xmax, g_ymax, g_zmax, f_2d, f_3d, marked_image, segment_label, ext_labels, neighbors_dict, processed_labels_list[segment_index], bottoms, heights, colors, mark)
    

  
    
                segment_index = 2
                segment_label = find_segment_for_point(labels, contour_mask, extrem[segment_index])
                mask = (ext_labels == segment_label)  # Get all pixels belonging to the segment
                marked_image[mask] = list(cv_colors)[segment_index].value
                ext_labels[ext_labels == segment_label] = 0
                #mask = (ext_labels == segment_index).astype(np.uint8) * 255 
                # Extract the lower and upper faces for the segment
                lower_face, upper_face, _ = get_projected_box(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)
                if lower_face is None or upper_face is None: continue
                
                if lower_face is None or upper_face is None: continue

                left_face = np.float32([lower_face[0], upper_face[0], upper_face[3], lower_face[3]])
                right_face = np.float32([lower_face[1], upper_face[1], upper_face[2], lower_face[2]])
                #pts2 = np.float32([(0, 0), (0, object_height), (object_width, object_height), (object_width, 0)])
    
                pr_left_face = map_points_to_BEV(left_face, ipm_left)
                
                width = (pr_left_face[-1] - pr_left_face[0])[0]
                height = (pr_left_face[1] - pr_left_face[0])[1]
                z = pr_left_face[0][1]
                length = calc_height(left_face, right_face, pr_left_face, inv_ipm_left, top_inv_ipm_left, object_length)
    
                avg_bottom_3d = [[0, object_width - width, z], [length, object_width - width, z], [length, object_width, z], [0, object_width, z]]
                #[[x, y, z] for x, y in pr_left_face]
                
                left_2d = map_points_to_BEV([left_v], ipm_left)[0]
                left_3d = [left_2d[0], left_2d[1], z + height/2]
                #ext_3d.append(left_3d)
                
                bottoms.append(np.array(avg_bottom_3d))
                pr_boxes.append((lower_face, upper_face))
                heights.append(height)
    
                cv2_color = cv_colors.BROWN.value
                plt_color = [c / 255.0 for c in cv2_color[::-1]]  # Normalize for plt
                colors.append(plt_color)
                
                    
                # Validate computed faces
                if lower_face is not None and upper_face is not None:
                    # Draw 3D cube
                    draw_cube(
                        marked_image,
                        lower_face.astype("int"),
                        upper_face.astype("int"),
                        color=cv_colors.BROWN.value,
                        thickness=4
                    )
                    
                f_2d = (lower_face, upper_face)
                f_3d = (np.array(avg_bottom_3d), np.array(avg_bottom_3d) + np.array([0, 0, height]))

                for i in range(len(extrem)):
                    processed_labels_list[i][segment_label] = {'iou': 1.0, 'score': 1.0, 'level': 0, 'lower_face': lower_face, 'upper_face': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height])}  # Initialize storage
                process_segment(g_xmax, g_ymax, g_zmax, f_2d, f_3d, marked_image, segment_label, ext_labels, neighbors_dict, processed_labels_list[segment_index], bottoms, heights, colors, mark)
    

                # Find the segment for right vertical point
                # segment_label = find_segment_for_point(labels, contour_mask, right_v)
                # print(f"Segment label: {segment_label}")
                # # Mark the segment in red
                # marked_image = mark_segment(marked_image, labels, segment_label, color=cv_colors.TEAL.value)
                # mask = (labels == segment_label)
                segment_index = 3
                segment_label = find_segment_for_point(labels, contour_mask, extrem[segment_index])
                mask = (ext_labels == segment_label)  # Get all pixels belonging to the segment
                marked_image[mask] = list(cv_colors)[segment_index].value
                ext_labels[ext_labels == segment_label] = 0
                #mask = (ext_labels == segment_index).astype(np.uint8) * 255 
                # Extract the lower and upper faces for the segment
                lower_face, upper_face, _ = get_projected_box(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)
                if lower_face is None or upper_face is None: continue
                
                left_face = np.float32([lower_face[0], upper_face[0], upper_face[3], lower_face[3]])
                right_face = np.float32([lower_face[1], upper_face[1], upper_face[2], lower_face[2]])
                #pts2 = np.float32([(0, 0), (0, object_height), (object_width, object_height), (object_width, 0)])
    
                pr_right_face = map_points_to_BEV(right_face, top_ipm_left)
                print(f"Projected right face: {pr_right_face}")
    
                width = (pr_right_face[-1] - pr_right_face[0])[0]
                height = (pr_right_face[1] - pr_right_face[0])[1]
                z = pr_right_face[0][1]
                length = calc_height(left_face, right_face, pr_right_face, inv_ipm_left, top_inv_ipm_left, object_length)
    
                avg_bottom_3d = [[object_length - length, 0, z], [object_length, 0, z], [object_length, width, z], [object_length - length, width, z]]
                
                right_2d = map_points_to_BEV([right_v], top_ipm_left)[0]
                right_3d = [right_2d[0], right_2d[1], z + height/2]
                #ext_3d.append(right_3d)
                
                bottoms.append(np.array(avg_bottom_3d))
                pr_boxes.append((lower_face, upper_face))
                heights.append(height)
    
                cv2_color = cv_colors.TEAL.value
                plt_color = [c / 255.0 for c in cv2_color[::-1]]  # Normalize for plt
                colors.append(plt_color)
                
                    
                # Validate computed faces
                if lower_face is not None and upper_face is not None:
                    # Draw 3D cube
                    draw_cube(
                        marked_image,
                        lower_face.astype("int"),
                        upper_face.astype("int"),
                        color=cv_colors.TEAL.value,
                        thickness=4
                    )
                    
                f_2d = (lower_face, upper_face)
                f_3d = (np.array(avg_bottom_3d), np.array(avg_bottom_3d) + np.array([0, 0, height]))
                # Start processing from segment_label
                #processed_labels = {} #([segment_label])
                for i in range(len(extrem)):
                    processed_labels_list[i][segment_label] = {'iou': 1.0, 'score': 1.0, 'level': 0, 'lower_face': lower_face, 'upper_face': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height])}  # Initialize storage
                process_segment(g_xmax, g_ymax, g_zmax, f_2d, f_3d, marked_image, segment_label, ext_labels, neighbors_dict, processed_labels_list[segment_index], bottoms, heights, colors, mark)
    
                ###################################
                segment_index = 5 # right_up
                segment_label = find_segment_for_point(labels, contour_mask, extrem[segment_index])
                mask = (ext_labels == segment_label)  # Get all pixels belonging to the segment
                marked_image[mask] = list(cv_colors)[segment_index].value
                ext_labels[ext_labels == segment_label] = 0
                #mask = (ext_labels == segment_index).astype(np.uint8) * 255 
                # Extract the lower and upper faces for the segment
                lower_face, upper_face, _ = get_projected_box(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)
                if lower_face is None or upper_face is None: continue
                
                nb_cube_2d = get_projected_cube_faces(lower_face, upper_face)
                # Convert face points to float32 for transformation
                pts1 = np.float32(cube_2d["right"])
                arr = cube_3d["right"].copy()
                # Step 1: Remove the first column (x) and keep only the second and third columns
                arr = arr[:, 1:]  # This slices the array from the second column onwards (index 1)
                # Step 2: Swap the second and third columns (y and z)
                arr[:, [0, 1]] = arr[:, [1, 0]]

                pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)

                # Compute Inverse Perspective Mapping (IPM) matrix
                ipm = cv2.getPerspectiveTransform(pts1, pts2)
            
                # Compute distances
                dist_2d = dist(cube_2d["left"], cube_2d["right"])
                dist_3d = dist(cube_3d["left"], cube_3d["right"])
                
                print(f"2D distance: {dist_2d}")
                print(f"3D distance: {dist_3d}")

                nb_dist_2d = dist(nb_cube_2d["left"], nb_cube_2d["right"])
            
                # Compute neighbor's 3D distance using ratio
                nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0

                # Transform the 2D points of the neighbor's face into BEV (Bird's Eye View)
                right_3d = map_points_to_BEV(nb_cube_2d["right"], ipm)  # Output is [z, y]
                
                # Swap columns to reorder from [z, y] → [y, z]
                right_3d = right_3d[:, [1, 0]]
                
                # Extend to [x, y, z] by adding the correct x-coordinate
                right_3d = np.column_stack((np.full(right_3d.shape[0], np.array(cube_3d["right"])[0, 0]), right_3d))
                # Step 2: Compute right_3d correctly (ensure 2D shape is maintained)
                left_3d = right_3d.copy()  # Copy left_3d structure
                left_3d[:, 0] -= nb_dist_3d  # Add nb_dist_3d to x-coordinates
                
                nb_upper_3d = np.array([left_3d[1], right_3d[1], right_3d[2], left_3d[2]])
                nb_lower_3d = np.array([left_3d[0], right_3d[0], right_3d[3], left_3d[3]])
                        
                
                bottoms.append(nb_lower_3d)
                pr_boxes.append((lower_face, upper_face))
                heights.append(nb_dist_3d)
    
                cv2_color = cv_colors.TEAL.value
                plt_color = [c / 255.0 for c in cv2_color[::-1]]  # Normalize for plt
                colors.append(plt_color)
                
                    
                # Validate computed faces
                if lower_face is not None and upper_face is not None:
                    # Draw 3D cube
                    draw_cube(
                        marked_image,
                        lower_face.astype("int"),
                        upper_face.astype("int"),
                        color=cv_colors.TEAL.value,
                        thickness=4
                    )

                f_2d = (lower_face, upper_face)
                f_3d = (np.array(avg_bottom_3d), np.array(avg_bottom_3d) + np.array([0, 0, height]))

                for i in range(len(extrem)):
                    processed_labels_list[i][segment_label] = {'iou': 1.0, 'score': 1.0, 'level': 0, 'lower_face': lower_face, 'upper_face': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height])}  # Initialize storage
                process_segment(g_xmax, g_ymax, g_zmax, f_2d, f_3d, marked_image, segment_label, ext_labels, neighbors_dict, processed_labels_list[segment_index], bottoms, heights, colors, mark)
    
        #averaged_processed_labels = compute_average_processed_labels(processed_labels_list)

        bottoms = []
        heights = []
        colors = []
        
        # Dictionary to store lists of 3D lower/upper faces for averaging
        segment_3d_data = {}
        
        for e in range(len(extrem)):
            for segment_index in (processed_labels_list[e]):  # Iterate over segment indices
                lower_face = processed_labels_list[e][segment_index]['lower_face']
                upper_face = processed_labels_list[e][segment_index]['upper_face']
                lower_face_3d = processed_labels_list[e][segment_index]['lower_face_3d']
                upper_face_3d = processed_labels_list[e][segment_index]['upper_face_3d']
                
                if lower_face_3d is not None and upper_face_3d is not None:
                    bottoms.append(lower_face_3d)

                if lower_face_3d is not None and upper_face_3d is not None:
                    if segment_index not in segment_3d_data:
                        segment_3d_data[segment_index] = {'lower_faces_3d': [], 'upper_faces_3d': []}
                    
                    segment_3d_data[segment_index]['lower_faces_3d'].append(lower_face_3d)
                    segment_3d_data[segment_index]['upper_faces_3d'].append(upper_face_3d)
                    
                    # Compute height using the Z-coordinates
                    height = (upper_face_3d - lower_face_3d)[0, 2]
                    heights.append(height)
                    cv2_color = list(cv_colors)[e % (len(cv_colors)-1)].value
                    plt_color = [c / 255.0 for c in cv2_color[::-1]]  # Normalize for plt
                    colors.append(plt_color)
                    
                    draw_cube(
                    marked_image,
                    lower_face.astype("int"),
                    upper_face.astype("int"),
                    color=cv2_color,
                    thickness=4
                )
            

        draw_cubes_with_bounding_image(marked_image, bottoms, heights, colors)
        
        reflect = True
        bottoms = []
        heights = []
        colors = []
        # Compute the average for each segment index
        for segment_index, data in segment_3d_data.items():
            avg_lower_face_3d = np.mean(data['lower_faces_3d'], axis=0)
            avg_upper_face_3d = np.mean(data['upper_faces_3d'], axis=0)
        
            # Compute height using Z-coordinates
            height = (avg_upper_face_3d - avg_lower_face_3d)[0, 2]
        
            # Append computed values
            bottoms.append(avg_lower_face_3d)
            heights.append(height)
            
            cv2_color = cv_colors.BLUE.value
            plt_color = [c / 255.0 for c in cv2_color[::-1]]  # Normalize for plt
            colors.append(plt_color)
            
            
            if reflect:
                # Reflect the lower face and add the reflected data
                reflected_bottom = reflect_segment(np.array(avg_lower_face_3d), g_object_width)
                bottoms.append(reflected_bottom)
                heights.append(height)  # Symmetric object, height remains the same
                colors.append(plt_color)  # Use the same color for symmetry
            
        draw_cubes_with_bounding_image(marked_image, bottoms, heights, colors)
        
        # process_segment(f_2d, f_3d, marked_image, segment_label, ext_labels, neighbors_dict, processed_labels, bottoms, heights, colors, mark)

        # Draw SLIC contours on the original image
        slic_contour_mask = slic.getLabelContourMask(thick_line=False)
        segmented_output = current_image.copy()
        marked_image[slic_contour_mask > 0] = 255  # Mark contours in white

        # Save the output images
        cv2.imwrite(os.path.join(output_path, f"abs_diff_{filename}"), abs_diff)
        cv2.imwrite(os.path.join(output_path, f"object_mask_{filename}"), object_mask)
        cv2.imwrite(os.path.join(output_path, f"slic_segments_{filename}"), labels_combined_color)
        cv2.imwrite(os.path.join(output_path, f"levels_{filename}"), levels_image)
        cv2.imwrite(os.path.join(output_path, f"box_{filename}"), box_image)

        print(f"Processed {filename}: Results saved to {output_path}")
        
        if len(bottoms) > 0:# and frame_count == 80:
            #closest_points_3d = find_closest_3d_point(extrem, pr_boxes, bottoms, heights)
            #draw_cubes_in_3d(bottoms, heights, colors, closest_points_3d)
            #draw_3d_points_with_hull(#ext_3d)
            #draw_cubes_with_bounding_image(marked_image, bottoms, heights, colors) #, closest_points_3d)
            centers = bottoms_to_centers(bottoms, heights)
            draw_interpolated_surface(centers, resolution=100, method='cubic')


# Example usage
# reference_image_path = "/home/dmytrozhuravlov/cv/data/img/reference.JPG"  # Replace with your reference image path
# # folder_path = "/home/dmytrozhuravlov/cv/data/img/out/"  # Replace with your folder path
# # output_path = "/home/dmytrozhuravlov/cv/data/img/out/out"  # Replace with your output folder path

# # reference_image_path = "/home/dmytrozhuravlov/cv/data/img/out/reference.JPG"  # Replace with your reference image path
# folder_path = "/home/dmytrozhuravlov/cv/data/img/test/"  # Replace with your folder path
# output_path = "/home/dmytrozhuravlov/cv/data/img/test/out/"  # Replace with your output folder path


reference_image_path = "/home/dzhura/ComputerVision/data/img/reference.JPG"  # Replace with your reference image path
folder_path = "/home/dzhura/ComputerVision/data/img/test/"  # Replace with your folder path
output_path = "/home/dzhura/ComputerVision/data/img/test/out"  # Replace with your output folder path

threshold_value = 25  # Adjust threshold value as needed

process_images(reference_image_path, folder_path, output_path, region_size=40*4, ruler=30*4, method="otsu")
