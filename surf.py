import cv2
import numpy as np
import os

from collections import defaultdict
from collections import deque

import copy

from opengl_drawer import *
from scipy.spatial import ConvexHull
import scipy.ndimage

from matplotlib.lines import Line2D


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

 

def scale_mask(mask, scale_factor=1.1):
    """
    Scales a binary mask by a given factor, keeping its center fixed.

    :param mask: Input binary mask (numpy array, dtype=bool or uint8)
    :param scale_factor: Factor to scale the mask (e.g., 1.1 for 10% expansion)
    :return: Scaled binary mask (same shape as input)
    """
    height, width = mask.shape

    # Compute new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize the mask using OpenCV with nearest-neighbor interpolation
    scaled_mask = cv2.resize(mask.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Crop or pad to match original dimensions
    if scale_factor > 1.0:
        # Crop the center region to original size
        start_x = (new_width - width) // 2
        start_y = (new_height - height) // 2
        scaled_mask = scaled_mask[start_y:start_y + height, start_x:start_x + width]
    else:
        # Pad the image to keep the same size
        pad_x = (width - new_width) // 2
        pad_y = (height - new_height) // 2
        scaled_mask = np.pad(scaled_mask, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')

    return scaled_mask.astype(bool)


def process_segment_var(
    segment_label, ext_labels, nbs_dict, processed_labels,
    vert_vp, hor_left_vp, hor_right_vp,
    xmax, ymax, zmax,
    reflect=False, clip=False, epsilon=np.array([1, 1, 1]),
    color=None, avg=False
):
    """Process segment and recursively process neighbors. Supports optional reflection and averaging."""

    if segment_label in processed_labels:
        print(f"Already processed label: {segment_label}")
        return processed_labels[segment_label]['lower_face_3d'], processed_labels[segment_label]['upper_face_3d']

    full_mask = (ext_labels > 0)
    mask = (ext_labels == segment_label)
    lower_face, upper_face, _ = get_projected_box(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)

    if lower_face is None or upper_face is None:
        processed_labels[segment_label] = {
            'lower_face_2d': None, 'upper_face_2d': None,
            'lower_face_3d': None, 'upper_face_3d': None,
            'color': color
        }
        return None, None

    # Save 2D faces
    processed_labels[segment_label] = {
        'lower_face_2d': lower_face, 'upper_face_2d': upper_face,
        'lower_face_3d': None, 'upper_face_3d': None,
        'color': color
    }
    processed_labels[-segment_label] = processed_labels[segment_label]  # Optional reflection entry

    cube_2d = get_projected_cube_faces(lower_face, upper_face)
    face_pairs = {
        "left": ("left", "right"),
        "right": ("right", "left"),
        "top": ("upper", "lower"),
        "bottom": ("lower", "upper"),
    }

    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([xmax, ymax, zmax])

    collected_cubes = []
    ref_cubes = [] if reflect else None

    # ❗ Initialize these to None so we avoid UnboundLocalError later
    lower_3d = None
    upper_3d = None

    nbs = nbs_dict.get(segment_label, {})
    for nb_label, directions in nbs.items():
        print(f"Processing neighbor label: {nb_label}")

        for direction in face_pairs:
            face1, face2 = face_pairs[direction]
            nb_mask = (ext_labels == nb_label)
            nb_lower, nb_upper, _ = get_projected_box(nb_mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)

            if nb_lower is None or nb_upper is None:
                continue

            nb_cube_2d = get_projected_cube_faces(nb_lower, nb_upper)
            iou = faces_intersection(cube_2d[face1], nb_cube_2d[face2]).area
            print(f"IoU with {direction}: {iou}")

            if iou == 0:
                continue

            nb_lower_3d, nb_upper_3d = process_segment(
                nb_label, ext_labels, nbs_dict, processed_labels,
                vert_vp, hor_left_vp, hor_right_vp,
                xmax, ymax, zmax,
                reflect=reflect, clip=clip, epsilon=epsilon, color=color, avg=avg
            )

            if nb_lower_3d is None or nb_upper_3d is None:
                continue

            nb_cube_3d = get_projected_cube_faces(nb_lower_3d, nb_upper_3d)

            # Compute IPM for transformation
            pts1 = np.float32(nb_cube_2d[face2])
            arr = nb_cube_3d[face2].copy()
            arr = arr[:, 1:] if direction in ["left", "right"] else arr[:, :2]
            if direction in ["left", "right"]:
                arr[:, [0, 1]] = arr[:, [1, 0]]
            pts2 = np.float32(arr)
            ipm = cv2.getPerspectiveTransform(pts1, pts2)

            dist_2d = dist(cube_2d[face1], cube_2d[face2])
            nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
            nb_dist_3d = dist(nb_cube_3d[face1], nb_cube_3d[face2])
            dist_3d = (nb_dist_3d / nb_dist_2d) * dist_2d if nb_dist_2d != 0 else 0

            bev = map_points_to_BEV(cube_2d[face1], ipm)
            if direction in ["left", "right"]:
                bev = bev[:, [1, 0]]
                axis = 0
                base = nb_cube_3d[face2][0, 0]
            else:
                axis = 1
                base = nb_cube_3d[face2][0, 2]

            base_pts = np.column_stack([
                np.full(bev.shape[0], base) if axis == 0 else bev[:, 0],
                bev[:, 0] if axis == 0 else bev[:, 1],
                bev[:, 1] if axis == 0 else np.full(bev.shape[0], base)
            ])

            if direction in ["right", "bottom"]:
                other_pts = base_pts.copy()
                other_pts[:, axis] -= dist_3d
            else:
                other_pts = base_pts.copy()
                other_pts[:, axis] += dist_3d

            lower_3d = np.array([other_pts[0], base_pts[0], base_pts[3], other_pts[3]])
            upper_3d = np.array([other_pts[1], base_pts[1], base_pts[2], other_pts[2]])

            if reflect:
                ref_cubes.append((lower_3d, upper_3d))

            collected_cubes.append((lower_3d, upper_3d))

    if avg and collected_cubes:
        lower_3d = np.mean([pair[0] for pair in collected_cubes], axis=0)
        upper_3d = np.mean([pair[1] for pair in collected_cubes], axis=0)

    if collected_cubes or (lower_3d is not None and upper_3d is not None):
        processed_labels[segment_label]['lower_face_3d'] = lower_3d
        processed_labels[segment_label]['upper_face_3d'] = upper_3d
        processed_labels[-segment_label]['lower_face_3d'] = lower_3d
        processed_labels[-segment_label]['upper_face_3d'] = upper_3d

    return lower_3d, upper_3d


def process_segment(segment_label, ext_labels, nbs_dict, processed_labels, vert_vp, hor_left_vp, hor_right_vp, xmax, ymax, zmax, reflect = False, clip = False, epsilon = np.array([1,1,1]), color = None):
    """ Process segment and recursively process neighbors. """
    
    if segment_label in processed_labels and processed_labels[segment_label]['visited']:
        print(f"Found proccessed label: {segment_label}")
        return processed_labels[segment_label]['lower_face_3d'], processed_labels[segment_label]['upper_face_3d']
    
    full_mask = (ext_labels > 0)
    #full_mask = scale_mask(full_mask, scale_factor=1.2)  # Scale up by 20%
    # Get mask for the segment
    mask = (ext_labels == segment_label)
    lower_face, upper_face, _ = get_projected_box(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)
    
    if lower_face is None or upper_face is None:
        processed_labels[segment_label] = {'visited': True, 'lower_face_2d': None, 'upper_face_2d': None, 'lower_face_3d': None, 'upper_face_3d': None, 'color': color}  # Initialize storage
        return None, None  # Skip if projection failed
        
    processed_labels[segment_label] = {'visited': True, 'lower_face_2d': lower_face, 'upper_face_2d': upper_face, 'lower_face_3d': None, 'upper_face_3d': None, 'color': color}  # Initialize storage
    if reflect:
        processed_labels[-segment_label] = {'visited': True, 'lower_face_2d': lower_face, 'upper_face_2d': upper_face, 'lower_face_3d': None, 'upper_face_3d': None, 'color': color}  # Initialize storage
    
    
    cube_2d = get_projected_cube_faces(lower_face, upper_face)  # Get all 6 faces of the cube
    # processed_labels[segment_label]['lower_face_2d'] = lower_face
    # processed_labels[segment_label]['upper_face_2d'] = upper_face

    # Process neighboring segments recursively
    nbs = nbs_dict.get(segment_label, {})
    
    #print(nbs)
    
    # nb is on direction
    face_pairs = {
        "left": ("left", "right"),
        "right":  ("right", "left"), #("left", "right"),
        "top": ("upper", "lower"),
        "bottom": ("lower", "upper"),
    }
    
    collected_cubes = []
    if reflect:
        ref_cubes = []
    
    # Define bounds
    # lower_bound = np.array([0 - epsilon, 0 - epsilon, 0 - epsilon])
    # upper_bound = np.array([xmax + epsilon, ymax + epsilon, zmax + epsilon])
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([xmax, ymax, zmax])


    
    for nb_label, directions in nbs.items():
        print(f"nb label : {nb_label}")
        for direction in directions:

            lower_3d = None
            upper_3d = None

            if reflect:
                ref_lower_2d = None
                ref_upper_2d = None
                ref_lower_3d = None
                ref_upper_3d = None

            nb_mask = (ext_labels == nb_label)
            nb_lower, nb_upper, _ = get_projected_box(nb_mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)
            if nb_lower is None or nb_upper is None:
                continue
            
            nb_cube_2d = get_projected_cube_faces(nb_lower, nb_upper)
            
            if direction in face_pairs:
                face1, face2 = face_pairs[direction]
                iou = faces_iou(cube_2d[face1], nb_cube_2d[face2])
                print(f"IoU: {iou}")

                if iou > 0:
                    #print(f"IoU: {iou}")
                    nb_lower_3d, nb_upper_3d = process_segment(nb_label, ext_labels, nbs_dict, processed_labels, vert_vp, hor_left_vp, hor_right_vp, xmax, ymax, zmax, reflect, clip, epsilon, color = color)
                    
                    if nb_lower_3d is None or nb_upper_3d is None:
                        continue
                        
                    # Check if values are within bounds
                    # if not (np.all(nb_lower_3d >= lower_bound) and np.all(nb_upper_3d <= upper_bound)):
                        # continue
                    
                    nb_cube_3d = get_projected_cube_faces(nb_lower_3d, nb_upper_3d)

                    print(direction)
                    if direction == "right":
                        # Compute perspective transform for 3D estimation
                        pts1 = np.float32(nb_cube_2d[face2])
                        arr = nb_cube_3d[face2].copy()
                        arr = arr[:, 1:]
                        arr[:, [0, 1]] = arr[:, [1, 0]]
                        pts2 = np.float32(arr)  # Convert to [y, z]
                        
                        ipm = cv2.getPerspectiveTransform(pts1, pts2)
                    
                        # Compute distances
                        nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                        nb_dist_3d = dist(nb_cube_3d[face1], nb_cube_3d[face2])
                        
                        print(f"NB 2D distance: {nb_dist_2d}")
                        print(f"NB 3D distance: {nb_dist_3d}")

                        dist_2d = dist(cube_2d[face1], cube_2d[face2])
                    
                        # Compute neighbor's 3D distance using ratio
                        dist_3d = (nb_dist_3d / nb_dist_2d) * dist_2d if nb_dist_2d != 0 else 0
                        
                        print(f"2D distance: {dist_2d}")
                        print(f"3D distance: {dist_3d}")

                        r_3d = map_points_to_BEV(cube_2d[face1], ipm)
                        r_3d = r_3d[:, [1, 0]]  # Swap [z, y] to [y, z]
                        r_3d = np.column_stack((np.full(r_3d.shape[0], nb_cube_3d[face2][0, 0]), r_3d))

                        l_3d = r_3d.copy()
                        l_3d[:, 0] -= dist_3d

                        upper_3d = np.array([l_3d[1], r_3d[1], r_3d[2], l_3d[2]])
                        lower_3d = np.array([l_3d[0], r_3d[0], r_3d[3], l_3d[3]])
                        
                    if direction == "left":
                        pts1 = np.float32(nb_cube_2d[face2])
                        arr = nb_cube_3d[face2].copy()
                        arr = arr[:, 1:]
                        arr[:, [0, 1]] = arr[:, [1, 0]]
                        pts2 = np.float32(arr)  # Convert to [y, z]
                        
                        ipm = cv2.getPerspectiveTransform(pts1, pts2)
                    
                        # Compute distances
                        nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                        nb_dist_3d = dist(nb_cube_3d[face1], nb_cube_3d[face2])
                        
                        print(f"NB 2D distance: {nb_dist_2d}")
                        print(f"NB 3D distance: {nb_dist_3d}")

                        dist_2d = dist(cube_2d[face1], cube_2d[face2])
                    
                        # Compute neighbor's 3D distance using ratio
                        dist_3d = (nb_dist_3d / nb_dist_2d) * dist_2d if nb_dist_2d != 0 else 0
                        
                        print(f"2D distance: {dist_2d}")
                        print(f"3D distance: {dist_3d}")

                        l_3d = map_points_to_BEV(cube_2d[face1], ipm)
                        l_3d = l_3d[:, [1, 0]]  # Swap [z, y] to [y, z]
                        l_3d = np.column_stack((np.full(l_3d.shape[0], nb_cube_3d[face2][0, 0]), l_3d))

                        r_3d = l_3d.copy()
                        r_3d[:, 0] += dist_3d

                        upper_3d = np.array([l_3d[1], r_3d[1], r_3d[2], l_3d[2]])
                        lower_3d = np.array([l_3d[0], r_3d[0], r_3d[3], l_3d[3]])
                        
                    elif direction == "top":
                        # Convert face points to float32 for transformation
                        pts1 = np.float32(nb_cube_2d[face2])
                        arr = nb_cube_3d[face2].copy()
                        # Step 1: Remove the first column (x) and keep only the second and third columns
                        arr = arr[:, :2]
                        pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)

                        # Compute Inverse Perspective Mapping (IPM) matrix
                        ipm = cv2.getPerspectiveTransform(pts1, pts2)
                    
                        # Compute distances
                        nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                        nb_dist_3d = dist(nb_cube_3d[face1], nb_cube_3d[face2])
                        
                        print(f"2D distance: {nb_dist_2d}")
                        print(f"3D distance: {nb_dist_3d}")

                        dist_2d = dist(cube_2d[face1], cube_2d[face2])
                    
                        # Compute neighbor's 3D distance using ratio
                        dist_3d = (nb_dist_3d / nb_dist_2d) * dist_2d if nb_dist_2d != 0 else 0
                    
                        # Transform the 2D points of the neighbor's face into BEV (Bird's Eye View)
                        u_3d = map_points_to_BEV(cube_2d[face1], ipm)  # Output is [x, y]
                        
                        # Extend to [x, y, z] by adding the correct x-coordinate
                        upper_3d = np.column_stack(( u_3d, np.full(u_3d.shape[0], np.array(nb_cube_3d[face2])[0, 2])))
                        # Step 2: Compute right_3d correctly (ensure 2D shape is maintained)
                        lower_3d = upper_3d.copy()  # Copy left_3d structure
                        lower_3d[:, 2] -= dist_3d  # Add nb_dist_3d to z-coordinates
                        
                    elif direction == "bottom":
                        # Convert face points to float32 for transformation
                        pts1 = np.float32(nb_cube_2d[face2])
                        arr = nb_cube_3d[face2].copy()
                        # Step 1: Remove the first column (x) and keep only the second and third columns
                        arr = arr[:, :2]
                        pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)

                        # Compute Inverse Perspective Mapping (IPM) matrix
                        ipm = cv2.getPerspectiveTransform(pts1, pts2)
                        inv_ipm = cv2.getPerspectiveTransform(pts2, pts1)

                    
                        # Compute distances
                        nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                        nb_dist_3d = dist(nb_cube_3d[face1], nb_cube_3d[face2])
                        
                        print(f"2D distance: {nb_dist_2d}")
                        print(f"3D distance: {nb_dist_3d}")

                        dist_2d = dist(cube_2d[face1], cube_2d[face2])
                    
                        # Compute neighbor's 3D distance using ratio
                        dist_3d = (nb_dist_3d / nb_dist_2d) * dist_2d if nb_dist_2d != 0 else 0
                    
                        # Transform the 2D points of the neighbor's face into BEV (Bird's Eye View)
                        d_3d = map_points_to_BEV(cube_2d[face1], ipm)  # Output is [x, y]
                        
                        # Extend to [x, y, z] by adding the correct x-coordinate
                        lower_3d = np.column_stack((d_3d, np.full(d_3d.shape[0], np.array(nb_cube_3d[face2])[0, 2])))
                        # Step 2: Compute right_3d correctly (ensure 2D shape is maintained)
                        upper_3d = lower_3d.copy()  # Copy left_3d structure
                        upper_3d[:, 2] += dist_3d  # Add nb_dist_3d to z-coordinates

               
                    if lower_3d is not None and upper_3d is not None:
                        cube_points = np.vstack((lower_3d, upper_3d))
                        if clip and not (np.all(cube_points >= lower_bound - epsilon) and np.all(cube_points <= upper_bound + epsilon)): continue
                        if clip:
                            # Ensure the values are within bounds
                            lower_3d = np.clip(lower_3d, lower_bound, upper_bound)
                            upper_3d = np.clip(upper_3d, lower_bound, upper_bound)
                        if not reflect:
                            collected_cubes.append({'lower_face_2d': lower_face.copy(), 'upper_face_2d': upper_face.copy(), 'lower_face_3d': lower_3d.copy(), 'upper_face_3d': upper_3d.copy()})
                        else:
                            ref_lower_3d = reflect_segment(lower_3d, ymax)
                            ref_upper_3d = reflect_segment(upper_3d, ymax)
                            
                            # Convert face points to float32 for transformation
                            pts1 = np.float32(lower_face)
                            arr = lower_3d.copy()
                            # Step 1: Remove the first column (x) and keep only the second and third columns
                            arr = arr[:, :2]
                            pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)
    
                            # Compute Inverse Perspective Mapping (IPM) matrix
                            ipm = cv2.getPerspectiveTransform(pts1, pts2)
                            inv_ipm = cv2.getPerspectiveTransform(pts2, pts1)

                            arr = ref_lower_3d.copy()
                            arr = arr[:, :2]
                            pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)
                            ref_lower_2d = map_points_to_BEV(pts2, inv_ipm)


                            # Convert face points to float32 for transformation
                            pts1 = np.float32(upper_face)
                            arr = upper_3d.copy()
                            # Step 1: Remove the first column (x) and keep only the second and third columns
                            arr = arr[:, :2]
                            pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)
    
                            # Compute Inverse Perspective Mapping (IPM) matrix
                            ipm = cv2.getPerspectiveTransform(pts1, pts2)
                            inv_ipm = cv2.getPerspectiveTransform(pts2, pts1)

                            # Convert face points to float32 for transformation
                            arr = ref_upper_3d.copy()
                            # Step 1: Remove the first column (x) and keep only the second and third columns
                            arr = arr[:, :2]
                            pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)

                            ref_upper_2d = map_points_to_BEV(pts2, inv_ipm)
                            
                            ref_center = get_intersect(ref_lower_2d[0], ref_upper_2d[2], ref_lower_2d[1], ref_upper_2d[3])
                            if ref_center is not None:
                                y = np.clip(int(ref_center[1]), 0, full_mask.shape[0] - 1)
                                x = np.clip(int(ref_center[0]), 0, full_mask.shape[1] - 1)
                                if full_mask[y, x]:
                                    collected_cubes.append({'lower_face_2d': lower_face, 'upper_face_2d': upper_face, 'lower_face_3d': lower_3d, 'upper_face_3d': upper_3d})
                                    ref_cubes.append({'lower_face_2d': ref_lower_2d, 'upper_face_2d': ref_upper_2d, 'lower_face_3d': ref_lower_3d, 'upper_face_3d': ref_upper_3d})
                  
                        
    if collected_cubes:
        # processed_labels[segment_label] = {'lower_face_2d': lower_face, 'upper_face_2d': upper_face, 'lower_face_3d': None, 'upper_face_3d': None}

        # processed_labels[segment_label]['lower_face'] = lower_face
        # processed_labels[segment_label]['upper_face'] = upper_face
        avg_lower = np.mean([cube['lower_face_3d'] for cube in collected_cubes], axis=0)
        avg_upper = np.mean([cube['upper_face_3d'] for cube in collected_cubes], axis=0)
        processed_labels[segment_label]['lower_face_3d'] = avg_lower.copy()
        processed_labels[segment_label]['upper_face_3d'] = avg_upper.copy()
        
    if reflect and ref_cubes:
        # Filter out None values
        valid_lower_faces_3d = [cube['lower_face_3d'] for cube in ref_cubes if cube['lower_face_3d'] is not None]
        valid_upper_faces_3d = [cube['upper_face_3d'] for cube in ref_cubes if cube['upper_face_3d'] is not None]
        
        valid_lower_faces_2d = [cube['lower_face_2d'] for cube in ref_cubes if cube['lower_face_2d'] is not None]
        valid_upper_faces_2d = [cube['upper_face_2d'] for cube in ref_cubes if cube['upper_face_2d'] is not None]
    
        if valid_lower_faces_3d and valid_upper_faces_3d:  # Ensure non-empty lists
            avg_lower_3d = np.mean(valid_lower_faces_3d, axis=0)
            avg_upper_3d = np.mean(valid_upper_faces_3d, axis=0)
        else:
            avg_lower_3d, avg_upper_3d = None, None  # Handle missing values safely
    
        if valid_lower_faces_2d and valid_upper_faces_2d:
            avg_lower_2d = np.mean(valid_lower_faces_2d, axis=0)
            avg_upper_2d = np.mean(valid_upper_faces_2d, axis=0)
        else:
            avg_lower_2d, avg_upper_2d = None, None
    
        # # Ensure processed_labels[-segment_label] exists
        # if -segment_label not in processed_labels:
            # processed_labels[-segment_label] = {}
    
        processed_labels[-segment_label]['lower_face_3d'] = avg_lower_3d.copy()
        processed_labels[-segment_label]['upper_face_3d'] = avg_upper_3d.copy()
        processed_labels[-segment_label]['lower_face_2d'] = avg_lower_2d.copy()
        processed_labels[-segment_label]['upper_face_2d'] = avg_upper_2d.copy()
    
    if processed_labels[segment_label]['lower_face_3d'] is not None:
        return processed_labels[segment_label]['lower_face_3d'].copy(), processed_labels[segment_label]['upper_face_3d'].copy()
    
    return None, None

def process_segment_new(segment_label, ext_labels, nbs_dict, processed_labels, vert_vp, hor_left_vp, hor_right_vp, xmax, ymax, zmax, reflect = False, clip = False, epsilon = np.array([1,1,1]), color = None):
    """ Process segment and recursively process neighbors. """
    
    if segment_label in processed_labels and processed_labels[segment_label]['visited']:# and processed_labels[segment_label]['lower_face_3d'] is not None and processed_labels[segment_label]['upper_face_3d'] is not None:
        print(f"Found proccessed label: {segment_label}")
        return processed_labels[segment_label]['lower_face_3d'], processed_labels[segment_label]['upper_face_3d']
    
    full_mask = (ext_labels > 0)
    #full_mask = scale_mask(full_mask, scale_factor=1.2)  # Scale up by 20%
    # Get mask for the segment
    mask = (ext_labels == segment_label)
    lower_face, upper_face, _ = get_projected_box(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)


    if lower_face is None or upper_face is None:
        processed_labels[segment_label] = {'visited': True, 'lower_face_2d': None, 'upper_face_2d': None, 'lower_face_3d': None, 'upper_face_3d': None, 'color': color}  # Initialize storage
        return None, None  # Skip if projection failed
        
    processed_labels[segment_label] = {'visited': True, 'lower_face_2d': lower_face, 'upper_face_2d': upper_face, 'lower_face_3d': None, 'upper_face_3d': None, 'color': color}  # Initialize storage
    processed_labels[-segment_label] = {'visited': True, 'lower_face_2d': lower_face, 'upper_face_2d': upper_face, 'lower_face_3d': None, 'upper_face_3d': None, 'color': color}  # Initialize storage
    
    cube_2d = get_projected_cube_faces(lower_face, upper_face)  # Get all 6 faces of the cube
    # processed_labels[segment_label]['lower_face_2d'] = lower_face
    # processed_labels[segment_label]['upper_face_2d'] = upper_face

    # Process neighboring segments recursively
    nbs = nbs_dict.get(segment_label, {})
    
    #print(nbs)
    
    # nb is on direction
    face_pairs = {
        "left": ("left", "right"),
        "right":  ("right", "left"), #("left", "right"),
        "top": ("upper", "lower"),
        "bottom": ("lower", "upper"),
    }
    
    collected_cubes = []
    if reflect:
        ref_cubes = []
    
    # Define bounds
    # lower_bound = np.array([0 - epsilon, 0 - epsilon, 0 - epsilon])
    # upper_bound = np.array([xmax + epsilon, ymax + epsilon, zmax + epsilon])
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([xmax, ymax, zmax])


    
    for nb_label, directions in nbs.items():
        print(f"nb label : {nb_label}")
        for direction in directions: #["left", "right", "top", "bottom"]: #directions:

            lower_3d = None
            upper_3d = None

            if reflect:
                ref_lower_2d = None
                ref_upper_2d = None
                ref_lower_3d = None
                ref_upper_3d = None

            nb_mask = (ext_labels == nb_label)
            nb_lower, nb_upper, _ = get_projected_box(nb_mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)
            if nb_lower is None or nb_upper is None:
                continue
            
            nb_cube_2d = get_projected_cube_faces(nb_lower, nb_upper)
            
            if direction in face_pairs:
                face1, face2 = face_pairs[direction]
                iou = faces_iou(cube_2d[face1], nb_cube_2d[face2])
                print(f"IoU: {iou}")

                if iou > 0:
                    #print(f"IoU: {iou}")
                    nb_lower_3d, nb_upper_3d = process_segment(nb_label, ext_labels, nbs_dict, processed_labels, vert_vp, hor_left_vp, hor_right_vp, xmax, ymax, zmax, reflect, clip, epsilon, color = color)
                    
                    if nb_lower_3d is None or nb_upper_3d is None:
                        continue
                        
                    # Check if values are within bounds
                    # if not (np.all(nb_lower_3d >= lower_bound) and np.all(nb_upper_3d <= upper_bound)):
                        # continue
                    
                    nb_cube_3d = get_projected_cube_faces(nb_lower_3d, nb_upper_3d)

                    print(direction)
                    if direction == "right":
                        # Compute perspective transform for 3D estimation
                        pts1 = np.float32(nb_cube_2d[face2])
                        arr = nb_cube_3d[face2].copy()
                        arr = arr[:, 1:]
                        arr[:, [0, 1]] = arr[:, [1, 0]]
                        pts2 = np.float32(arr)  # Convert to [y, z]
                        
                        ipm = cv2.getPerspectiveTransform(pts1, pts2)
                    
                        # Compute distances
                        nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                        nb_dist_3d = dist(nb_cube_3d[face1], nb_cube_3d[face2])
                        
                        print(f"NB 2D distance: {nb_dist_2d}")
                        print(f"NB 3D distance: {nb_dist_3d}")

                        dist_2d = dist(cube_2d[face1], cube_2d[face2])
                    
                        # Compute neighbor's 3D distance using ratio
                        dist_3d = (nb_dist_3d / nb_dist_2d) * dist_2d if nb_dist_2d != 0 else 0
                        
                        print(f"2D distance: {dist_2d}")
                        print(f"3D distance: {dist_3d}")

                        r_3d = map_points_to_BEV(cube_2d[face1], ipm)
                        r_3d = r_3d[:, [1, 0]]  # Swap [z, y] to [y, z]
                        r_3d = np.column_stack((np.full(r_3d.shape[0], nb_cube_3d[face2][0, 0]), r_3d))

                        l_3d = r_3d.copy()
                        l_3d[:, 0] -= dist_3d

                        upper_3d = np.array([l_3d[1], r_3d[1], r_3d[2], l_3d[2]])
                        lower_3d = np.array([l_3d[0], r_3d[0], r_3d[3], l_3d[3]])
                        
                    if direction == "left":
                        pts1 = np.float32(nb_cube_2d[face2])
                        arr = nb_cube_3d[face2].copy()
                        arr = arr[:, 1:]
                        arr[:, [0, 1]] = arr[:, [1, 0]]
                        pts2 = np.float32(arr)  # Convert to [y, z]
                        
                        ipm = cv2.getPerspectiveTransform(pts1, pts2)
                    
                        # Compute distances
                        nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                        nb_dist_3d = dist(nb_cube_3d[face1], nb_cube_3d[face2])
                        
                        print(f"NB 2D distance: {nb_dist_2d}")
                        print(f"NB 3D distance: {nb_dist_3d}")

                        dist_2d = dist(cube_2d[face1], cube_2d[face2])
                    
                        # Compute neighbor's 3D distance using ratio
                        dist_3d = (nb_dist_3d / nb_dist_2d) * dist_2d if nb_dist_2d != 0 else 0
                        
                        print(f"2D distance: {dist_2d}")
                        print(f"3D distance: {dist_3d}")

                        l_3d = map_points_to_BEV(cube_2d[face1], ipm)
                        l_3d = l_3d[:, [1, 0]]  # Swap [z, y] to [y, z]
                        l_3d = np.column_stack((np.full(l_3d.shape[0], nb_cube_3d[face2][0, 0]), l_3d))

                        r_3d = l_3d.copy()
                        r_3d[:, 0] += dist_3d

                        upper_3d = np.array([l_3d[1], r_3d[1], r_3d[2], l_3d[2]])
                        lower_3d = np.array([l_3d[0], r_3d[0], r_3d[3], l_3d[3]])
                        
                    elif direction == "top":
                        # Convert face points to float32 for transformation
                        pts1 = np.float32(nb_cube_2d[face2])
                        arr = nb_cube_3d[face2].copy()
                        # Step 1: Remove the first column (x) and keep only the second and third columns
                        arr = arr[:, :2]
                        pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)

                        # Compute Inverse Perspective Mapping (IPM) matrix
                        ipm = cv2.getPerspectiveTransform(pts1, pts2)
                    
                        # Compute distances
                        nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                        nb_dist_3d = dist(nb_cube_3d[face1], nb_cube_3d[face2])
                        
                        print(f"2D distance: {nb_dist_2d}")
                        print(f"3D distance: {nb_dist_3d}")

                        dist_2d = dist(cube_2d[face1], cube_2d[face2])
                    
                        # Compute neighbor's 3D distance using ratio
                        dist_3d = (nb_dist_3d / nb_dist_2d) * dist_2d if nb_dist_2d != 0 else 0
                    
                        # Transform the 2D points of the neighbor's face into BEV (Bird's Eye View)
                        u_3d = map_points_to_BEV(cube_2d[face1], ipm)  # Output is [x, y]
                        
                        # Extend to [x, y, z] by adding the correct x-coordinate
                        upper_3d = np.column_stack(( u_3d, np.full(u_3d.shape[0], np.array(nb_cube_3d[face2])[0, 2])))
                        # Step 2: Compute right_3d correctly (ensure 2D shape is maintained)
                        lower_3d = upper_3d.copy()  # Copy left_3d structure
                        lower_3d[:, 2] -= dist_3d.copy()  # Add nb_dist_3d to z-coordinates
                        
                    elif direction == "bottom":
                        # Convert face points to float32 for transformation
                        pts1 = np.float32(nb_cube_2d[face2])
                        arr = nb_cube_3d[face2].copy()
                        # Step 1: Remove the first column (x) and keep only the second and third columns
                        arr = arr[:, :2]
                        pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)

                        # Compute Inverse Perspective Mapping (IPM) matrix
                        ipm = cv2.getPerspectiveTransform(pts1, pts2)
                        inv_ipm = cv2.getPerspectiveTransform(pts2, pts1)

                    
                        # Compute distances
                        nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                        nb_dist_3d = dist(nb_cube_3d[face1], nb_cube_3d[face2])
                        
                        print(f"2D distance: {nb_dist_2d}")
                        print(f"3D distance: {nb_dist_3d}")

                        dist_2d = dist(cube_2d[face1], cube_2d[face2])
                    
                        # Compute neighbor's 3D distance using ratio
                        dist_3d = (nb_dist_3d / nb_dist_2d) * dist_2d if nb_dist_2d != 0 else 0
                    
                        # Transform the 2D points of the neighbor's face into BEV (Bird's Eye View)
                        d_3d = map_points_to_BEV(cube_2d[face1], ipm)  # Output is [x, y]
                        
                        # Extend to [x, y, z] by adding the correct x-coordinate
                        lower_3d = np.column_stack((d_3d, np.full(d_3d.shape[0], np.array(nb_cube_3d[face2])[0, 2])))
                        # Step 2: Compute right_3d correctly (ensure 2D shape is maintained)
                        upper_3d = lower_3d.copy()  # Copy left_3d structure
                        upper_3d[:, 2] += dist_3d  # Add nb_dist_3d to z-coordinates

               
                    if lower_3d is not None and upper_3d is not None:
                        processed_labels[segment_label]['lower_face_3d'] = lower_3d.copy()
                        processed_labels[segment_label]['upper_face_3d'] = upper_3d.copy()
                        return lower_3d.copy(), upper_3d.copy()
                        
                        cube_points = np.vstack((lower_3d, upper_3d))
                        if clip and not (np.all(cube_points >= lower_bound - epsilon) and np.all(cube_points <= upper_bound + epsilon)): continue
                        # Ensure the values are within bounds
                        if clip:
                            lower_3d = np.clip(lower_3d, lower_bound, upper_bound)
                            upper_3d = np.clip(upper_3d, lower_bound, upper_bound)
                        if not reflect:
                            #collected_cubes.append({'lower_face_2d': lower_face, 'upper_face_2d': upper_face, 'lower_face_3d': lower_3d, 'upper_face_3d': upper_3d})
                            #return lower_3d.copy(), upper_3d.copy()
                            processed_labels[segment_label]['lower_face_3d'] = lower_3d
                            processed_labels[segment_label]['upper_face_3d'] = upper_3d
                            return lower_3d.copy(), upper_3d.copy()
                            
                        else:
                            ref_lower_3d = reflect_segment(lower_3d, ymax)
                            ref_upper_3d = reflect_segment(upper_3d, ymax)

                            # Convert face points to float32 for transformation
                            pts1 = np.float32(lower_face)
                            arr = lower_3d.copy()
                            # Step 1: Remove the first column (x) and keep only the second and third columns
                            arr = arr[:, :2]
                            pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)

                            # Compute Inverse Perspective Mapping (IPM) matrix
                            ipm = cv2.getPerspectiveTransform(pts1, pts2)
                            inv_ipm = cv2.getPerspectiveTransform(pts2, pts1)

                            arr = ref_lower_3d.copy()
                            arr = arr[:, :2]
                            pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)
                            ref_lower_2d = map_points_to_BEV(pts2, inv_ipm)


                            # Convert face points to float32 for transformation
                            pts1 = np.float32(upper_face)
                            arr = upper_3d.copy()
                            # Step 1: Remove the first column (x) and keep only the second and third columns
                            arr = arr[:, :2]
                            pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)

                            # Compute Inverse Perspective Mapping (IPM) matrix
                            ipm = cv2.getPerspectiveTransform(pts1, pts2)
                            inv_ipm = cv2.getPerspectiveTransform(pts2, pts1)

                            # Convert face points to float32 for transformation
                            arr = ref_upper_3d.copy()
                            # Step 1: Remove the first column (x) and keep only the second and third columns
                            arr = arr[:, :2]
                            pts2 = np.float32(arr)  # Remove 3rd coordinate (only x, y)

                            ref_upper_2d = map_points_to_BEV(pts2, inv_ipm)
                            
                            ref_center = get_intersect(ref_lower_2d[0], ref_upper_2d[2], ref_lower_2d[1], ref_upper_2d[3])
                            if ref_center is not None:
                                y = np.clip(int(ref_center[1]), 0, full_mask.shape[0] - 1)
                                x = np.clip(int(ref_center[0]), 0, full_mask.shape[1] - 1)
                                if full_mask[y, x]:
                                    collected_cubes.append({'lower_face_2d': lower_face, 'upper_face_2d': upper_face, 'lower_face_3d': lower_3d, 'upper_face_3d': upper_3d})
                                    ref_cubes.append({'lower_face_2d': ref_lower_2d, 'upper_face_2d': ref_upper_2d, 'lower_face_3d': ref_lower_3d, 'upper_face_3d': ref_upper_3d})
                                    
    return None, None

    # if len(collected_cubes) > 0:

        # # processed_labels[segment_label] = {'lower_face_2d': lower_face, 'upper_face_2d': upper_face, 'lower_face_3d': None, 'upper_face_3d': None}

        # # processed_labels[segment_label]['lower_face'] = lower_face
        # # processed_labels[segment_label]['upper_face'] = upper_face
        # avg_lower = np.mean([cube['lower_face_3d'] for cube in collected_cubes], axis=0)
        # avg_upper = np.mean([cube['upper_face_3d'] for cube in collected_cubes], axis=0)
        # processed_labels[segment_label]['lower_face_3d'] = avg_lower
        # processed_labels[segment_label]['upper_face_3d'] = avg_upper
        
        # if reflect and ref_cubes:
            # # Filter out None values
            # valid_lower_faces_3d = [cube['lower_face_3d'] for cube in ref_cubes if cube['lower_face_3d'] is not None]
            # valid_upper_faces_3d = [cube['upper_face_3d'] for cube in ref_cubes if cube['upper_face_3d'] is not None]
            
            # valid_lower_faces_2d = [cube['lower_face_2d'] for cube in ref_cubes if cube['lower_face_2d'] is not None]
            # valid_upper_faces_2d = [cube['upper_face_2d'] for cube in ref_cubes if cube['upper_face_2d'] is not None]
        
            # if valid_lower_faces_3d and valid_upper_faces_3d:  # Ensure non-empty lists
                # avg_lower_3d = np.mean(valid_lower_faces_3d, axis=0)
                # avg_upper_3d = np.mean(valid_upper_faces_3d, axis=0)
            # else:
                # avg_lower_3d, avg_upper_3d = None, None  # Handle missing values safely
        
            # if valid_lower_faces_2d and valid_upper_faces_2d:
                # avg_lower_2d = np.mean(valid_lower_faces_2d, axis=0)
                # avg_upper_2d = np.mean(valid_upper_faces_2d, axis=0)
            # else:
                # avg_lower_2d, avg_upper_2d = None, None
        
            # # # Ensure processed_labels[-segment_label] exists
            # # if -segment_label not in processed_labels:
                # # processed_labels[-segment_label] = {}
        
            # processed_labels[-segment_label]['lower_face_3d'] = avg_lower_3d
            # processed_labels[-segment_label]['upper_face_3d'] = avg_upper_3d
            # processed_labels[-segment_label]['lower_face_2d'] = avg_lower_2d
            # processed_labels[-segment_label]['upper_face_2d'] = avg_upper_2d
            


        # return processed_labels[segment_label]['lower_face_3d'].copy(), processed_labels[segment_label]['upper_face_3d'].copy()

    # else:
        # return None, None


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

def find_central_label(labels):
    # Create a binary mask where labels > 0
    full_mask = (labels > 0)

    # Get coordinates of all nonzero (True) pixels
    y_coords, x_coords = np.where(full_mask)

    # Compute centroid of the mask
    centroid_y, centroid_x = np.mean(y_coords), np.mean(x_coords)

    # Find the label closest to the centroid
    labeled_mask, num_labels = scipy.ndimage.label(full_mask)
    object_labels = np.unique(labels[full_mask])  # Extract unique labels in mask

    min_distance = float('inf')
    central_label = None

    for lbl in object_labels:
        # Get coordinates of pixels belonging to the current label
        y, x = np.where(labels == lbl)

        # Compute distance from centroid to each pixel in the label
        distances = np.sqrt((y - centroid_y) ** 2 + (x - centroid_x) ** 2)

        # Find the minimum distance for this label
        avg_distance = np.mean(distances)

        # Update central label if it's the closest so far
        if avg_distance < min_distance:
            min_distance = avg_distance
            central_label = lbl

    return central_label
    
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
    
    return centers

def draw_debug_overlay(box_image, contour_mask, centers, neighbors_dict, debug=True):
    overlay = box_image.copy()

    # Optional: draw white contour over superpixels
    #overlay[contour_mask != 0] = [255, 255, 255]

    for label, (y, x) in centers.items():
        cv2.circle(overlay, (int(x), int(y)), 8, (0, 255, 255), -1)  # Yellow dot

    if debug:
        for label1, neighbors in neighbors_dict.items():
            y0, x0 = centers[label1]
            for label2 in neighbors:
                y1, x1 = centers[label2]
                pt1 = (int(x0), int(y0))
                pt2 = (int(x1), int(y1))
                cv2.line(overlay, pt1, pt2, (255, 255, 0), 5)  # Cyan line

    return overlay

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
        #camera_position = np.float32([width/3, height])
        camera_position = np.float32([width/2, height])
        level = 1  # Start from level 0
        levels_image = color_image_for_slic.copy()
        
        ground_level_segments = set()
        
        while np.any(levels > 0) and level < 2:  # Continue until all segments are marked
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
                            ground_level_segments.add(segment_label)
        
            # Remove border segments from segmentation and color them
            for label in border_segments:
                levels_image[levels  == label] = cv2_color
                levels [levels  == label] = level  # Remove the labeled segments
        
            # Increment the level for the next color
            level += 1
        
        # Final visualization
        levels_image[0 < contour_mask] = cv_colors.BLACK.value

        # Final visualization
        # levels_image[0 < contour_mask] = cv_colors.BLACK.value
        # for seg in ground_level_segments:
            # levels_image[labels == seg] = cv_colors.RED.value

        # levels_image[0 < contour_mask] = cv_colors.BLACK.value
        # cv2.imshow("Final Colored Segmentation", levels_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # Initialize dictionaries and lists
        segments = {}
        bottoms = []
        pr_boxes = []
        heights = []
        colors = []

        box_image = color_image.copy()
        full_mask = (labels > 0)
        ext_labels = labels #segment_mask(mask, extrem)
        
        cube = False #True
        pig = True #False

        if pig:
            g_xmin = xmin = 0
            g_xmax = xmax = 21
            g_ymin = ymin = 0
            g_object_width = g_ymax = ymax = 7
            g_zmin = zmin = 0
            g_zmax = zmax = 7
        elif cube:
            g_xmin = xmin = 0
            g_xmax = xmax = 21
            g_ymin = ymin = 0
            g_object_width = g_ymax = ymax = 21
            g_zmin = zmin = 0
            g_zmax = zmax = 21
        
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
        
        print(f"Extrem: {extrem}")
        
        extrem_labels = []
        for e in extrem:
            extrem_labels.append(find_segment_for_point(labels, contour_mask, e))
            
        print(f"Extrem labels: {extrem_labels}")
        

        count = 0
        
        reflect = False#True
        mark = True#False
        
        #processed_labels = {}
        slic_contour_mask = slic.getLabelContourMask(thick_line=False)
        neighbors_dict, centers = find_neighbors(labels, slic_contour_mask)

        # Generate and show overlay

        box_image[slic_contour_mask > 0] = 255  # Mark contours in white
        marked_image = draw_debug_overlay(box_image, contour_mask, centers, neighbors_dict, debug=True)
        cv2.imshow("Superpixel Centers and Neighbor Connections", resize_to_height(marked_image, 1980//3))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
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
                
                initial_segments = set()
                processed_labels = {}
                
                
                # TODO ground levels
                # for segment_label in ground_level_segments:

                    # processed_labels[segment_label] = {}

                    # mask = (ext_labels == segment_label)
                    # lower_face, upper_face, _ = get_projected_box(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)
                    # if lower_face is None or upper_face is None: continue

                    # initial_segments.add(segment_label)
                    # pr_lower_face = map_points_to_BEV(lower_face, ipm_matrix)
                    # height = calc_height(lower_face, upper_face, pr_lower_face, inv_ipm_matrix, top_inv_ipm_matrix, object_height)
                    # z = zmin
                    # avg_bottom_3d = [[x, y, z] for x, y in pr_lower_face]
                    # processed_labels[segment_label][segment_label] = {'lower_face_2d': lower_face, 'upper_face_2d': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height]), 'color': cv_colors.RED.value}
    
                    
                #for segment_index in range(len(extrem)):
                    
                segment_index = 1
                segment_label = find_segment_for_point(labels, contour_mask, extrem[segment_index])
                initial_segments.add(segment_label)
                mask = (ext_labels == segment_label)  # Get all pixels belonging to the segment
                if mark: marked_image[mask] = list(cv_colors)[(segment_index) % (len(cv_colors)-1)].value
    
                lower_face, upper_face, _ = get_projected_box(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False)
                if lower_face is None or upper_face is None: continue
    
    
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
                    
                # f_2d = (lower_face, upper_face)
                # f_3d = (np.array(avg_bottom_3d), np.array(avg_bottom_3d) + np.array([0, 0, height]))
                processed_labels[segment_index] = {}  # Initialize as a dictionary
                processed_labels[segment_index][segment_label] = {'visited': False, 'lower_face_2d': lower_face, 'upper_face_2d': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height]), 'color': cv_colors.RED.value}
                # for i in range(len(extrem)):
                    # processed_labels_list[i][segment_label] = {'iou': 1.0, 'score': 1.0, 'level': 0, 'lower_face': lower_face, 'upper_face': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height])}  # Initialize storage
                # process_segment(g_xmax, g_ymax, g_zmax, f_2d, f_3d, marked_image, segment_label, ext_labels, neighbors_dict, processed_labels_list[segment_index], bottoms, heights, colors, mark)
    

                segment_index = 0
                segment_label = find_segment_for_point(labels, contour_mask, extrem[segment_index])
                initial_segments.add(segment_label)
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

                # f_2d = (lower_face, upper_face)
                # f_3d = (np.array(avg_bottom_3d), np.array(avg_bottom_3d) + np.array([0, 0, height]))

                # for i in range(len(extrem)):
                    # processed_labels_list[i][segment_label] = {'iou': 1.0, 'score': 1.0, 'level': 0, 'lower_face': lower_face, 'upper_face': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height])}  # Initialize storage
                # process_segment(g_xmax, g_ymax, g_zmax, f_2d, f_3d, marked_image, segment_label, ext_labels, neighbors_dict, processed_labels_list[segment_index], bottoms, heights, colors, mark)
                processed_labels[segment_index] = {}
                processed_labels[segment_index][segment_label] = {'visited': False, 'lower_face_2d': lower_face, 'upper_face_2d': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height]), 'color': cv_colors.RED.value}

                segment_index = 4
                segment_label = find_segment_for_point(labels, contour_mask, extrem[segment_index])
                initial_segments.add(segment_label)
                mask = (ext_labels == segment_label)  # Get all pixels belonging to the segment
                marked_image[mask] = list(cv_colors)[segment_index].value
                #ext_labels[ext_labels == segment_label] = 0
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
                    
                # f_2d = (lower_face, upper_face)
                # f_3d = (np.array(avg_bottom_3d), np.array(avg_bottom_3d) + np.array([0, 0, height]))
                # # Start processing from segment_label
                # #processed_labels = {} #([segment_label])
                # for i in range(len(extrem)):
                    # processed_labels_list[i][segment_label] = {'iou': 1.0, 'score': 1.0, 'level': 0, 'lower_face': lower_face, 'upper_face': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height])}  # Initialize storage
                # process_segment(g_xmax, g_ymax, g_zmax, f_2d, f_3d, marked_image, segment_label, ext_labels, neighbors_dict, processed_labels_list[segment_index], bottoms, heights, colors, mark)
                processed_labels[segment_index] = {}
                processed_labels[segment_index][segment_label] = {'visited': False, 'lower_face_2d': lower_face, 'upper_face_2d': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height]), 'color': cv_colors.RED.value}

  
    
                segment_index = 2
                segment_label = find_segment_for_point(labels, contour_mask, extrem[segment_index])
                initial_segments.add(segment_label)
                mask = (ext_labels == segment_label)  # Get all pixels belonging to the segment
                marked_image[mask] = list(cv_colors)[segment_index].value
                #ext_labels[ext_labels == segment_label] = 0
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
                    
                # f_2d = (lower_face, upper_face)
                # f_3d = (np.array(avg_bottom_3d), np.array(avg_bottom_3d) + np.array([0, 0, height]))

                # for i in range(len(extrem)):
                    # processed_labels_list[i][segment_label] = {'iou': 1.0, 'score': 1.0, 'level': 0, 'lower_face': lower_face, 'upper_face': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height])}  # Initialize storage
                # process_segment(g_xmax, g_ymax, g_zmax, f_2d, f_3d, marked_image, segment_label, ext_labels, neighbors_dict, processed_labels_list[segment_index], bottoms, heights, colors, mark)
                processed_labels[segment_index] = {}
                processed_labels[segment_index][segment_label] = {'visited': False, 'lower_face_2d': lower_face, 'upper_face_2d': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height]), 'color': cv_colors.RED.value}

                # Find the segment for right vertical point
                # segment_label = find_segment_for_point(labels, contour_mask, right_v)
                # print(f"Segment label: {segment_label}")
                # # Mark the segment in red
                # marked_image = mark_segment(marked_image, labels, segment_label, color=cv_colors.TEAL.value)
                # mask = (labels == segment_label)
                segment_index = 3
                segment_label = find_segment_for_point(labels, contour_mask, extrem[segment_index])
                initial_segments.add(segment_label)

                mask = (ext_labels == segment_label)  # Get all pixels belonging to the segment
                marked_image[mask] = list(cv_colors)[segment_index].value
                #ext_labels[ext_labels == segment_label] = 0
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
                    
                # f_2d = (lower_face, upper_face)
                # f_3d = (np.array(avg_bottom_3d), np.array(avg_bottom_3d) + np.array([0, 0, height]))
                # # Start processing from segment_label
                # #processed_labels = {} #([segment_label])
                # for i in range(len(extrem)):
                    # processed_labels_list[i][segment_label] = {'iou': 1.0, 'score': 1.0, 'level': 0, 'lower_face': lower_face, 'upper_face': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height])}  # Initialize storage
                # process_segment(g_xmax, g_ymax, g_zmax, f_2d, f_3d, marked_image, segment_label, ext_labels, neighbors_dict, processed_labels_list[segment_index], bottoms, heights, colors, mark)
                processed_labels[segment_index] = {}
                processed_labels[segment_index][segment_label] = {'visited': False, 'lower_face_2d': lower_face, 'upper_face_2d': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height]), 'color': cv_colors.RED.value}
                 
                 
                ###################################
                segment_index = 5 # right_up

                segment_label = find_segment_for_point(labels, contour_mask, extrem[segment_index])
                initial_segments.add(segment_label)
                right_segment_label = segment_label
                mask = (ext_labels == segment_label)  # Get all pixels belonging to the segment
                #marked_image[mask] = list(cv_colors)[segment_index].value
                #ext_labels[ext_labels == segment_label] = 0
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
                        
                
                # bottoms.append(nb_lower_3d)
                # pr_boxes.append((lower_face, upper_face))
                # heights.append(nb_dist_3d)
    
                # cv2_color = cv_colors.TEAL.value
                # plt_color = [c / 255.0 for c in cv2_color[::-1]]  # Normalize for plt
                # colors.append(plt_color)
                
                    
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

                # f_2d = (lower_face, upper_face)
                # f_3d = (np.array(avg_bottom_3d), np.array(avg_bottom_3d) + np.array([0, 0, height]))

                # for i in range(len(extrem)):
                    # processed_labels_list[i][segment_label] = {'iou': 1.0, 'score': 1.0, 'level': 0, 'lower_face': lower_face, 'upper_face': upper_face, 'lower_face_3d': np.array(avg_bottom_3d), 'upper_face_3d': np.array(avg_bottom_3d) + np.array([0, 0, height])}  # Initialize storage
                # process_segment(g_xmax, g_ymax, g_zmax, f_2d, f_3d, marked_image, segment_label, ext_labels, neighbors_dict, processed_labels_list[segment_index], bottoms, heights, colors, mark)
                processed_labels[segment_index] = {}
                processed_labels[segment_index][segment_label] = {'visited': False, 'lower_face_2d': lower_face, 'upper_face_2d': upper_face, 'lower_face_3d': np.array(nb_lower_3d), 'upper_face_3d': np.array(nb_lower_3d) + np.array([0, 0, nb_dist_3d]), 'color': cv_colors.RED.value}

        #averaged_processed_labels = compute_average_processed_labels(processed_labels_list)
        
        #marked_image[full_mask] = cv_colors.GRAY.value
        unique_labels = np.unique(labels[full_mask])
        
        
        reflect = False #True
        processed_labels_full = processed_labels.copy()
        
        bottoms = []
        tops = []
        heights = []
        colors = []
        
        centers = []
        
  
        
        for i, initial_label in enumerate(extrem):   #(initial_segments):
            
            bottoms = []
            tops = []
            heights = []
            colors = []
            marked_image_copy = marked_image.copy()
            
            
            color_i= list(cv_colors)[i % (len(cv_colors)-1)].value
            #marked_image_copy[(labels == initial_label) ] = color_i

            i_label = find_segment_for_point(labels, contour_mask, extrem[i])
            processed_labels[i][i_label]['visited'] =  True
            for segment_label in unique_labels:
                print(f"Segment label : {segment_label}")
                # if segment_label in processed_labels[i] and processed_labels[i][segment_label]['visited']:
                    # continue  # Skip if already processed
                lower_face_3d, upper_face_3d = process_segment(segment_label, labels, neighbors_dict, processed_labels[i], vert_vp, hor_left_vp, hor_right_vp, g_xmax, g_ymax, g_zmax, reflect= reflect, clip= False, color = color_i)
                if lower_face_3d is None or upper_face_3d is None:
                    print("Error computing: ", processed_labels[i])
                    continue
            # segment_label = find_segment_for_point(labels, contour_mask, extrem[i])
            # #visited = set()
            # process_segment(segment_label, labels, neighbors_dict, processed_labels[i], vert_vp, hor_left_vp, hor_right_vp, g_xmax, g_ymax, g_zmax, reflect= reflect, clip= False, color = color_i)
    
            # for segment_label, segment_data in processed_labels[i].items():
                # print(f"processed_labels[{i}]: { processed_labels[i]}")

                # # Get lower and upper face data safely
                # lower_face = segment_data['lower_face_2d']#, None)
                # upper_face = segment_data['upper_face_2d']#, None)
                # lower_face_3d = segment_data['lower_face_3d']#, None)
                # upper_face_3d = segment_data['upper_face_3d']#, None)
                
                # Skip if missing
                # if lower_face is None or upper_face is None:
                    # #marked_image_copy[labels == segment_label] = cv_colors.BLACK.value
                    # continue
                
                # # Store color if available
                # if 'color' in segment_data:
                    # cv2_color = segment_data['color']
                # else:
                    # cv2_color = cv_colors.BLUE.value
                    
                cv2_color = color_i
                    
                # draw_cube(
                    # marked_image_copy,
                    # lower_face.astype("int"),
                    # upper_face.astype("int"),
                    # color=color_i, #cv2_color,
                    # thickness=4
                # )
            
                # Skip if missing
                if lower_face_3d is None or upper_face_3d is None:
                    # draw_cube(
                    # marked_image_copy,
                    # lower_face.astype("int"),
                    # upper_face.astype("int"),
                    # color=cv_colors.GRAY.value,
                    # thickness=4
                    # )
                    print("Error computing: ", segment_data)
                    continue
                   
                   
                # Convert to numpy arrays
                # lower_face = np.array(lower_face)
                # upper_face = np.array(upper_face)
                # Draw 3D cube

                # draw_cube(
                    # marked_image_copy,
                    # lower_face.astype("int"),
                    # upper_face.astype("int"),
                    # color=color_i, #cv2_color,
                    # thickness=4
                # )
    
                    
                # Convert to numpy arrays
                lower_face_3d = np.array(lower_face_3d)
                upper_face_3d = np.array(upper_face_3d)
        
                height = (upper_face_3d - lower_face_3d)[0, 2]
        
                bottoms.append(lower_face_3d.copy())
                tops.append(upper_face_3d.copy())
                heights.append(height.copy())
                
                centers.append(upper_face_3d[0])
        
    
            
                plt_color = [c / 255.0 for c in cv2_color[::-1]]  # Normalize for plt
                #plt_color = [c / 255.0 for c in cv_colors.BLUE.value[::-1]] 
                colors.append(plt_color)
                    
                if reflect and segment_label in initial_segments:
                    # Reflect the lower face and add the reflected data
                    reflected_bottom = reflect_segment(np.array(lower_face_3d), g_object_width)
                    reflected_top = reflect_segment(np.array(upper_face_3d), g_object_width)
                    bottoms.append(reflected_bottom)
                    tops.append(reflected_top)
                    heights.append(height)  # Symmetric object, height remains the same
                    colors.append(plt_color)  # Use the same color for symmetry
                    
                    centers.append(upper_face_3d[-1])
                    
            draw_cubes_with_bounding_image(marked_image_copy, bottoms, heights, colors, alpha = 0.15) #, closest_points_3d)
            plt.show()

                    
        slic_contour_mask = slic.getLabelContourMask(thick_line=False)
        marked_image[slic_contour_mask > 0] = 255  # Mark contours in white
                
        draw_cubes_with_bounding_image(marked_image, bottoms, heights, colors, alpha = 0.25) #, closest_points_3d)
        plt.show()
        
        
        draw_interpolated_surface(centers)
        plt.show()
        
        # verts, tris = build_mesh_from_cuboids(bottoms, tops)
        # draw_mesh_matplotlib(verts, tris)
        # plt.show()
        
        # bottoms = []
        # tops = []
        # heights = []
        # colors = []
        
        # centers = []
        
        # # Dictionary to accumulate values for averaging
        # face_data = defaultdict(lambda: {"lower": [], "upper": []})
        
        # # Collect all occurrences of each segment_label across initial_labels
        # for initial_label in processed_labels:
            # for segment_label in processed_labels[initial_label]:
                # lower_face = processed_labels[initial_label][segment_label]['lower_face_3d']
                # upper_face = processed_labels[initial_label][segment_label]['upper_face_3d']
        
                # if lower_face is not None and upper_face is not None:
                    # face_data[segment_label]["lower"].append(lower_face)
                    # face_data[segment_label]["upper"].append(upper_face)
           
        # color = [c / 255.0 for c in cv_colors.BLUE.value[::-1]]     

        # # Compute average and update processed_labels
        # for segment_label, faces in face_data.items():
            # try:
                # #print(f"[DEBUG] Processing segment_label '{segment_label}' with {len(faces['lower'])} lower and {len(faces['upper'])} upper faces.")
        
                # if not faces["lower"]:
                    # #print(f"[DEBUG] Skipping avg_lower for segment_label '{segment_label}' as faces['lower'] is empty.")
                    # avg_lower = None
                # else:
                    # avg_lower = np.mean(faces["lower"], axis=0)
                    # #print(f"[DEBUG] segment_label '{segment_label}': avg_lower computed with shape {avg_lower.shape}")
        
                # if not faces["upper"]:
                   # # print(f"[DEBUG] Skipping avg_upper for segment_label '{segment_label}' as faces['upper'] is empty.")
                    # avg_upper = None
                # else:
                    # avg_upper = np.mean(faces["upper"], axis=0)
                    # #print(f"[DEBUG] segment_label '{segment_label}': avg_upper computed with shape {avg_upper.shape}")
                    
                # height = (avg_upper - avg_lower)[0, 2]
    
                # tops.append(avg_upper)
                # heights.append(height)
                # bottoms.append(avg_lower)
                # colors.append(color)
                


        
            # except Exception as e:
                # print(f"[ERROR] Failed to compute average for segment_label '{segment_label}': {e}")
                # print(f"[DEBUG] faces['lower'] content: {faces['lower']}")
                # print(f"[DEBUG] faces['upper'] content: {faces['upper']}")
            

        # draw_cubes_with_bounding_image(marked_image, bottoms, heights, colors, alpha = 0.25) #, closest_points_3d)
        # plt.show()
        
        

        
        # centers, triangles = build_triangulation_from_upper_face_centers(tops)
        # draw_mesh_matplotlib(np.array(centers), triangles)
                
        
        # verts, tris = build_mesh_from_cuboids(bottoms, tops)
        # draw_mesh_matplotlib(verts, tris)
        # plt.show()
        
        


            #marked_image[(labels == segment_label) ] = cv_colors.GRAY.value
            
            

           
           
           
            
        #draw_cubes_with_bounding_image(marked_image, bottoms, heights, colors)
        
        # process_segment(f_2d, f_3d, marked_image, segment_label, ext_labels, neighbors_dict, processed_labels, bottoms, heights, colors, mark)
# and frame_count == 80:

        # Draw SLIC contours on the original image #, closest_points_3d)
        
        slic_contour_mask = slic.getLabelContourMask(thick_line=False)
        segmented_output = current_image.copy()
        marked_image[slic_contour_mask > 0] = 255  # Mark contours in white

        # Save the output images

        print(f"Processed {filename}: Results saved to {output_path}")
        
        if len(bottoms) > 0:# and frame_count == 80:


        
            plt.show()



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

process_images(reference_image_path, folder_path, output_path, region_size=40*4*2, ruler=30*4*2, method="otsu")
#process_images(reference_image_path, folder_path, output_path, region_size=50*4, ruler=40*4, method="otsu")
