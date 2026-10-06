import numpy as np
import cv2
from shapely.geometry import Point, LineString
from scipy.spatial.distance import cdist

import numpy as np
from shapely.geometry import LineString
from skimage.draw import line  # Efficient line tracing

def compute_centers(labels):
    """
    Compute the center of each segment in the label mask.

    Args:
        labels (np.array): Label mask from SLIC.

    Returns:
        dict: {label: (y, x) center}
    """
    centers = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == 0:  # Ignore background
            continue
        coords = np.argwhere(labels == label)
        center = np.mean(coords, axis=0)
        centers[label] = (center[0], center[1])  # (y, x) format
    return centers


def is_valid_neighbor(label1, label2, labels, centers=None):
    """
    Check if label1 and label2 are direct neighbors by checking if any pixels of the two segments touch.

    Args:
        label1 (int): First segment label.
        label2 (int): Second segment label.
        labels (np.array): Label mask from SLIC.
        centers (dict, optional): Not used in this version but kept for compatibility.

    Returns:
        bool: True if any pixels of the two segments are adjacent (distance 0), False otherwise.
    """
    # Get coordinates of all pixels belonging to each label
    coords1 = np.argwhere(labels == label1)
    coords2 = np.argwhere(labels == label2)

    # Compute pairwise distances between all pixels
    dists = cdist(coords1, coords2, metric='cityblock')  # faster than euclidean

    # Check if any pair has distance 1 (4-connectivity) or 0 (overlap)
    return np.any(dists <= 1)

def is_valid_neighbor_contour(label1, label2, labels, contour_mask):
    """
    Check if label1 and label2 are direct neighbors by testing if they both appear near
    the same contour pixel in the contour mask.

    Args:
        label1 (int): First segment label.
        label2 (int): Second segment label.
        labels (np.array): Superpixel label map.
        contour_mask (np.array): Binary mask from slic.getLabelContourMask().

    Returns:
        bool: True if the two labels are adjacent in the image, False otherwise.
    """
    H, W = labels.shape

    # Iterate only over contour pixels (nonzero mask values)
    ys, xs = np.nonzero(contour_mask)

    for y, x in zip(ys, xs):
        # Check 8-connected neighborhood around contour pixel
        neighborhood = labels[max(y-1,0):min(y+2,H), max(x-1,0):min(x+2,W)]
        unique_labels = np.unique(neighborhood)

        # If both label1 and label2 are present near the contour pixel, they are neighbors
        if label1 in unique_labels and label2 in unique_labels:
            return True

    return False


def is_valid_neighbor_center(label1, label2, labels, centers):
    """
    Check if label1 and label2 are direct neighbors by verifying if the line between their centers crosses other segments.

    Args:
        label1 (int): First segment label.
        label2 (int): Second segment label.
        labels (np.array): Label mask from SLIC.

    Returns:
        bool: True if valid neighbors, False otherwise.
    """
    # Compute centers
    center1 = centers[label1] #np.mean(np.argwhere(labels == label1), axis=0)
    center2 = centers[label2] #np.mean(np.argwhere(labels == label2), axis=0)

    # Convert (y, x) → (x, y) for Shapely
    center1_xy = (center1[1], center1[0])
    center2_xy = (center2[1], center2[0])

    # Create a line between centers
    connection_line = LineString([center1_xy, center2_xy])

    # Get all points along the line (integer pixel locations)
    rr, cc = line(int(center1[0]), int(center1[1]), int(center2[0]), int(center2[1]))

    # Check if any of the traversed pixels belong to another segment
    for r, c in zip(rr, cc):
        if labels[r, c] not in {0, label1, label2}:  # Ignore background (0) and same labels
            return False  # Another segment obstructs the connection

    return True  # No obstruction, valid neighbor

# def get_relative_position(center1, center2):
    # """
    # Determine the relative position of center2 with respect to center1.

    # Args:
        # center1 (tuple): (y, x) coordinates of first center.
        # center2 (tuple): (y, x) coordinates of second center.

    # Returns:
        # str: "left", "right", "top", "bottom"
    # """
    # dy = center2[0] - center1[0]
    # dx = center2[1] - center1[1]
 
    # directions = set()

    # if abs(dx) > abs(dy):  # More horizontal movement
        # return "right" if dx > 0 else "left"
    # else:  # More vertical movement
        # return "bottom" if dy > 0 else "top"

def get_relative_position(center1, center2):
    """
    Determine the relative position of center2 with respect to center1.

    Args:
        center1 (tuple): (y, x) coordinates of first center.
        center2 (tuple): (y, x) coordinates of second center.

    Returns:
        set: A set containing possible directions ("left", "right", "top", "bottom").
    """
    dy = center2[0] - center1[0]
    dx = center2[1] - center1[1]

    directions = set()

    if dy >= 0:
        directions.add("bottom")
    elif dy < 0:
        directions.add("top")

    if dx >= 0:
        directions.add("right")
    elif dx < 0:
        directions.add("left")

    return directions

# def opposite_direction(direction):
    # """
    # Get the opposite of a given direction.

    # Args:
        # direction (str): One of "left", "right", "top", "bottom".

    # Returns:
        # str: Opposite direction.
    # """
    # opposites = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
    # return opposites[direction]
    
def opposite_direction(directions):
    """
    Get the opposite of given directions.

    Args:
        directions (set): A set containing any of "left", "right", "top", "bottom".

    Returns:
        set: A set containing the opposite directions.
    """
    opposites = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
    return {opposites[direction] for direction in directions}

def find_neighbors(labels, contour):
    """
    Identify neighboring segments and their relative positions.

    Args:
        labels (np.array): Label mask from SLIC.

    Returns:
        dict: {label1: {label2: "direction"}}
    """
    centers = compute_centers(labels)
    unique_labels = list(centers.keys())
    
    neighbors_dict = {label: {} for label in unique_labels}

    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i + 1:]:  # Avoid duplicate comparisons
            # if is_valid_neighbor(label1, label2, labels, centers):
            if is_valid_neighbor_contour(label1, label2, labels, contour):
                direction = get_relative_position(centers[label1], centers[label2])
                neighbors_dict[label1][label2] = direction
                neighbors_dict[label2][label1] = opposite_direction(direction)

    return neighbors_dict, centers



def find_neighbors_copy(labels):
    """
    Identify neighboring segments and their relative positions.

    Args:
        labels (np.array): Label mask from SLIC.

    Returns:
        dict: {label1: {label2: "direction"}}
    """
    centers = compute_centers(labels)
    unique_labels = list(centers.keys())
    
    neighbors_dict = {label: {} for label in unique_labels}

    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i + 1:]:  # Avoid duplicate comparisons
            if is_valid_neighbor(label1, label2, labels):
                direction = get_relative_position(centers[label1], centers[label2])
                neighbors_dict[label1][label2] = direction
                neighbors_dict[label2][label1] = opposite_direction(direction)

    return neighbors_dict

# def find_neighbors(labels):
    # """
    # Find neighboring segments by checking if the line between centers is obstructed.
    
    # Args:
        # labels (np.array): Label map from cv2.ximgproc.createSuperpixelSLIC
    
    # Returns:
        # dict: A dictionary where keys are segment labels and values are their neighboring labels with directions.
    # """
    # unique_labels = np.unique(labels)
    # centers = {}

    # # Compute segment centers
    # for label in unique_labels:
        # if label == 0:  # Skip background
            # continue
        # yx_coords = np.column_stack(np.where(labels == label))
        # centers[label] = yx_coords.mean(axis=0)  # Compute centroid

    # neighbors_dict = {label: {} for label in unique_labels}

    # # Check segment connectivity
    # for label1, center1 in centers.items():
        # for label2, center2 in centers.items():
            # if label1 >= label2:
                # continue  # Avoid duplicate comparisons
            
            # # Create a line between segment centers
            # connection_line = LineString([center1[::-1], center2[::-1]])  # Convert (y, x) → (x, y)
            
            # # Check if any other segment intersects the line
            # obstructed = False
            # for other_label, other_center in centers.items():
                # if other_label not in {label1, label2}:
                    # if connection_line.intersects(Point(other_center[::-1]).buffer(2)):  # Small buffer
                        # obstructed = True
                        # break
            
            # # If no obstruction, they are neighbors
            # if not obstructed:
                # dx, dy = center2 - center1
                # direction = (
                    # "right" if dx > 5 else "left" if dx < -5 else
                    # "bottom" if dy > 5 else "top"
                # )
                
                # # Store neighbor with direction
                # neighbors_dict[label1][label2] = direction
                # neighbors_dict[label2][label1] = direction  # Ensure bidirectionality

    # return neighbors_dict


# def find_neighbors(labels, e=1):
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
                # 'top': [(y - e, x), (y - e, x - e), (y - e, x + e)],
                # 'bottom': [(y + e, x), (y + e, x - e), (y + e, x + e)],
                # 'left': [(y, x - e), (y - e, x - e), (y + e, x - e)],
                # 'right': [(y, x + e), (y - e, x + e), (y + e, x + e)],
            # }

            # # Iterate through the neighboring directions
            # for direction, positions in neighbors.items():
                # for ny, nx in positions:
                    # # Skip neighbors that are out of bounds
                    # if ny < 0 or ny >= labels.shape[0] or nx < 0 or nx >= labels.shape[1]:
                        # continue

                    # neighbor_label = labels[ny, nx]
                    # if neighbor_label == 0 or neighbor_label == current_label:
                        # continue

                    # # Add neighbor if different from current label
                    # if neighbor_label not in neighbors_dict[current_label]:
                        # neighbors_dict[current_label][neighbor_label] = set()
                    
                    # # Record the direction of this neighbor relative to current label
                    # neighbors_dict[current_label][neighbor_label].add(direction)

    # return neighbors_dict
