import cv2
import numpy as np
import os
from opengl_drawer import *
from shapely.geometry import Polygon
import flowiz as fz

from lifting import *
from enum import Enum

import os

# Define your capture source and output directory
output_dir = "output_frames"
output_bev_dir = "output_bev_frames"
os.makedirs(output_dir, exist_ok=True)

# Initialize video capture and output
# /home/dmytrozhuravlov/cv/data/
# '/home/dzhura/mount/cv/data/
# /home/dzhura/ComputerVision/data
input_video_path = '/home/dzhura/ComputerVision/data/4kStreetView Cctv.mp4'
output_video_path = 'output_with_cubes.mp4'



class cv_colors(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    PURPLE = (247, 44, 200)
    ORANGE = (44, 162, 247)
    MINT = (239, 255, 66)
    YELLOW = (2, 255, 250)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)


debug = True
use_slick = True  #False
show_warp = True
resize_shape = (1600, 900)
screen_width, screen_height = 1920//2, 1080//2
roi_corners = None
min_square = 100
flow_treshold = 0.1

region_size = 70  #200 #100 #15  #10#30
ruler = 60  #150 # 100 #20 #14
slic_iterations = 10


def unwarp(img, roi_corners):
    src = np.float32(roi_corners)
    warped_size = (img.shape[1], img.shape[0])
    offset = int(warped_size[0] / 3.0)

    dst = np.float32([[offset, warped_size[1]], [offset, 0],
                      [warped_size[0] - offset, 0],
                      [warped_size[0] - offset, warped_size[1]]])

    Mpersp = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, Mpersp, dsize=warped_size)
    return warped


def select_points(image):
    points = []
    lines = []

    def select_point(event, x, y, flags, param):
        nonlocal points, lines
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            if len(points) > 1:
                lines.append((points[-2], points[-1]))
            if len(points) == 4:
                lines.append((points[-1], points[0]))
            for line in lines:
                cv2.line(image, line[0], line[1], (0, 255, 0), 2)
            for point in points:
                cv2.circle(image, point, 5, (0, 0, 255), -1)
            cv2.imshow('Select Points', image)
            if len(points) == 4:
                cv2.waitKey(0)

    cv2.imshow('Select Points', image)
    cv2.setMouseCallback('Select Points', select_point)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.array(points, dtype=np.float32)


def compute_cuboid_dimensions(lower_face, upper_face, lower_center, bottom_bev,
                              optical_center):
    """
    Computes the approximate dimensions (length, width, height) of a cuboid
    considering BEV projection for width and length scaling and using the
    lower and upper faces in the plain view for height.
    
    Parameters:
    - lower_face: np.array of shape (4, 2), coordinates of the four corners of the lower face in plain view.
    - upper_face: np.array of shape (4, 2), coordinates of the four corners of the upper face in plain view.
    - bottom_bev: np.array of shape (4, 2), coordinates of the four corners of the lower face in BEV projection.
    - optical_center: np.array of shape (2,), coordinates of the optical center in the image.

    Returns:
    - length: Estimated length of the cuboid.
    - width: Estimated width of the cuboid.
    - height: Estimated height of the cuboid.
    """

    # Compute BEV distances (real-world lengths) for length and width
    bev_length = np.linalg.norm(bottom_bev[0] - bottom_bev[1])
    bev_width = np.linalg.norm(bottom_bev[0] - bottom_bev[3])

    # # Calculate projected distances in plain view for length and width
    # proj_length = np.linalg.norm(lower_face[0] - lower_face[1])
    # proj_width = np.linalg.norm(lower_face[0] - lower_face[3])

    # # Scaling factors based on BEV
    # length_scale = bev_length / proj_length if proj_length else 1
    # width_scale = bev_width / proj_width if proj_width else 1

    # # Apply scaling factors
    # length = proj_length * length_scale
    # width = proj_width * width_scale

    # # Calculate height as average of vertical distances between lower and upper face points
    # height_edges = [np.linalg.norm(lower_face[i] - upper_face[i]) for i in range(4)]
    # height = np.mean(height_edges)

    # Calculate height from lower and upper faces using perspective adjustment
    # Get the average distance between lower and upper face points
    avg_height_projection = np.mean(
        [np.linalg.norm(lower_face[i] - upper_face[i]) for i in range(4)])

    # Calculate the perspective scaling factor based on distance to the optical center
    distance_to_optical_center = np.linalg.norm(lower_center - optical_center)
    scaling_factor = 1 + (distance_to_optical_center /
                          np.linalg.norm(lower_face[0] - lower_face[2]))

    # Compute height by scaling the projected height difference
    height = avg_height_projection * scaling_factor

    return bev_length, bev_width, height


def generate_perspective_matrix_leeds(img, roi_corners):
    pst2 = np.float32([[-36., -290.], [1154., 420.], [400., 918.],
                       [600., 948.]])
    pst1 = np.float32([[0., 0.], [960., 0.], [0., 540.], [960., 540.]])

    persp = cv2.getPerspectiveTransform(pst1, pst2)
    inv = cv2.getPerspectiveTransform(pst2, pst1)

    bev_image_height = 1028
    bev_image_width = 1048

    return persp, inv, (bev_image_width, bev_image_height)


def generate_perspective_matrix(ratio=0.5):
   
    seeds = [[1, 508], [747, 235], [1401, 215], [1082, 701]]


    widthA = np.sqrt(((seeds[3][0] - seeds[0][0]) ** 2) + ((seeds[3][1] - seeds[0][1]) ** 2))
    widthB = np.sqrt(((seeds[1][0] - seeds[2][0]) ** 2) + ((seeds[1][1] - seeds[2][1]) ** 2))
    maxWidth = max(int(widthA), int(widthB)) 
    heightA = np.sqrt(((seeds[1][0] - seeds[0][0]) ** 2) + ((seeds[1][1] - seeds[0][1]) ** 2))
    heightB = np.sqrt(((seeds[3][0] - seeds[2][0]) ** 2) + ((seeds[3][1] - seeds[2][1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    pts1 = np.float32(seeds)    
    pts2 = ratio * np.float32([ [0, maxHeight] , [0, 0], [maxWidth, 0], [maxWidth, maxHeight] ])

    # Generate the perspective transformation matrices
    persp = cv2.getPerspectiveTransform(pts1, pts2)
    inv = cv2.getPerspectiveTransform(pts2, pts1)


    return persp, inv, (int(ratio*maxWidth) + 1, int(ratio*maxHeight)+1)


def crop_warp(warped):
    return warped
    top_padding, left_padding = 1000000, 0
    warped_size = (warped.shape[1], warped.shape[0])
    offset = 0

    return warped[top_padding:warped_size[1],
                  left_padding + offset:warped_size[0] - left_padding - offset]


def compute_average_frame(video_path, avg_frame_path):
    if os.path.exists(avg_frame_path):
        avg_frame = cv2.imread(avg_frame_path)
        return avg_frame

    cap = cv2.VideoCapture(video_path)
    avg_frame = None
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if avg_frame is None:
            avg_frame = np.float32(frame)
        else:
            cv2.accumulate(frame, avg_frame)
        count += 1

    cap.release()
    avg_frame /= count
    avg_frame = cv2.convertScaleAbs(avg_frame)

    cv2.imwrite(avg_frame_path, avg_frame)
    return avg_frame

def get_motion_mask(flow_mag, motion_thresh=1, kernel=np.ones((7,7))):
    """ Obtains Detection Mask from Optical Flow Magnitude
        Inputs:
            flow_mag (array) Optical Flow magnitude
            motion_thresh - thresold to determine motion
            kernel - kernal for Morphological Operations
        Outputs:
            motion_mask - Binray Motion Mask
        """
    motion_mask = np.uint8(flow_mag > motion_thresh)*255

    motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return motion_mask

def compute_optical_flow1(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    # blurr image
    prev_gray = cv2.GaussianBlur(prev_gray, dst=None, ksize=(3,3), sigmaX=5)
    next_gray = cv2.GaussianBlur(next_gray, dst=None, ksize=(3,3), sigmaX=5)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15,
                                        3, 5, 1.2, 0)
    return flow

def compute_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    # Preprocess with bilateral filter for edge preservation
    prev_gray = cv2.bilateralFilter(prev_gray, d=5, sigmaColor=50, sigmaSpace=50)
    next_gray = cv2.bilateralFilter(next_gray, d=5, sigmaColor=50, sigmaSpace=50)

    # Calculate optical flow with fine-tuned parameters
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                        pyr_scale=0.5,   # pyramid scale
                                        levels=5,        # pyramid levels
                                        winsize=21,      # window size
                                        iterations=5,    # iterations per level
                                        poly_n=7,        # neighborhood size
                                        poly_sigma=1.5,  # standard deviation
                                        flags=0)
    return flow


def mark_boundaries(image,
                    labels,
                    boundary_color=cv_colors.BLACK.value,
                    alpha=0.3):
    boundaries = np.copy(image)  # Blank boundary image

    # Loop through the image and check for boundary conditions
    for y in range(1, labels.shape[0] - 1):
        for x in range(1, labels.shape[1] - 1):
            if (labels[y, x] != labels[y - 1, x]
                    or labels[y, x] != labels[y + 1, x]
                    or labels[y, x] != labels[y, x - 1]
                    or labels[y, x] != labels[y, x + 1]):

                boundaries[y, x] = boundary_color

    # Blend the boundaries over the original image
    blended_image = cv2.addWeighted(image, 1 - alpha, boundaries, alpha, 0)

    return blended_image


def color_neighbors(image, labels, neighbors_dict, moving_mask):
    # Convert labels image to an RGB color image for display
    colored_image = 255 * np.ones((*labels.shape, 3), dtype=np.uint8)

    # Dictionary to store colors for each label
    label_colors = {}

    # Define a recursive helper function for flood-filling connected neighbors
    def flood_fill(label, color):
        if label not in label_colors and label != 0:
            label_colors[label] = color
            colored_image[labels == label] = color

            # Recursively flood-fill neighbors
            for neighbor_label in neighbors_dict.get(label, {}):
                if neighbor_label not in label_colors:
                    flood_fill(neighbor_label, color)

    # Iterate over each label and apply flood-fill
    for label in neighbors_dict.keys():
        if label not in label_colors:
            random_color = np.random.randint(0, 255, 3)
            flood_fill(label, random_color)

    # Mark pixels outside the moving mask as white
    #colored_image[moving_mask == 0] = cv_colors.WHITE.value

    return colored_image


# def color_neighbors(image, labels, neighbors_dict, moving_mask):
# # Convert labels image to an RGB color image for display
# colored_image = 255 * np.ones((*labels.shape, 3), dtype=np.uint8)

# # Get the total number of unique labels
# num_labels = labels.max() + 1

# # Generate random colors for each label and store them in an array
# label_colors = np.random.randint(0, 255, (num_labels, 3), dtype=np.uint8)

# # Define a recursive helper function for flood-filling connected neighbors
# def flood_fill(label):
# if label not in label_colors and label != 0:
# color = label_colors[label]
# colored_image[labels == label] = color
# cv2.imshow('Colored Flood', resize_to_match_height(colored_image, screen_height))
# cv2.waitKey(0)

# # Recursively flood-fill neighbors
# for neighbor_label in neighbors_dict.get(label, {}):
# if neighbor_label != 0 and np.all(colored_image[labels == neighbor_label] == 255):
# flood_fill(neighbor_label)

# # Iterate over each label and apply flood-fill
# for label in neighbors_dict.keys():
# if np.all(colored_image[labels == label] == 255):  # if label has not been colored yet
# flood_fill(label)

# # # Mark pixels outside the moving mask as white
# # colored_image[moving_mask == 0] = [255, 255, 255]

# return colored_image


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


# def find_neighbors(image, labels, e = 1):
# # Initialize the dictionary with each unique label
# neighbors_dict = {label: {} for label in range(1, np.max(labels) + 1)}

# # Iterate through each pixel in the labels array
# for y in range(labels.shape[0]):
# for x in range(labels.shape[1]):
# current_label = labels[y, x]

# # Define neighbor pixel offsets in (y, x) format
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
# if ny < 0 or ny >= labels.shape[0] or nx < 0 or nx >= labels.shape[1]:
# continue

# neighbor_label = labels[ny, nx]

# # Only add if neighbor label is different from current label
# if neighbor_label != current_label:
# if neighbor_label not in neighbors_dict[current_label]:
# neighbors_dict[current_label][neighbor_label] = set()

# # Record the direction of this neighbor relative to current label
# neighbors_dict[current_label][neighbor_label].add(direction)

# return neighbors_dict

# def find_neighbors(image, labels, e=50):
# # Initialize the dictionary with each unique label
# neighbors_dict = {label: {} for label in range(1, np.max(labels) + 1)}

# # Iterate through each unique label to find its neighbors
# unique_labels = np.unique(labels)
# for label in unique_labels:
# # Get the mask for the current label
# label_mask = (labels == label).astype(np.uint8)

# # Find the bounding box around the current label mask to limit the search area
# y_coords, x_coords = np.where(label_mask)
# y_min, y_max = max(np.min(y_coords) - e, 0), min(np.max(y_coords) + e + 1, labels.shape[0])
# x_min, x_max = max(np.min(x_coords) - e, 0), min(np.max(x_coords) + e + 1, labels.shape[1])

# # Create a neighborhood mask within the epsilon range
# for y in range(y_min, y_max):
# for x in range(x_min, x_max):
# neighbor_label = labels[y, x]
# if neighbor_label != label:
# # Determine relative position
# dy, dx = y - y_coords.mean(), x - x_coords.mean()
# direction = None
# if dy < -e / 2 and dx < -e / 2:
# direction = 'top-left'
# elif dy < -e / 2 and dx > e / 2:
# direction = 'top-right'
# elif dy > e / 2 and dx < -e / 2:
# direction = 'bottom-left'
# elif dy > e / 2 and dx > e / 2:
# direction = 'bottom-right'
# elif dy < -e / 2:
# direction = 'top'
# elif dy > e / 2:
# direction = 'bottom'
# elif dx < -e / 2:
# direction = 'left'
# elif dx > e / 2:
# direction = 'right'

# if direction:
# # Initialize neighbor entry if not already present
# if neighbor_label not in neighbors_dict[label]:
# neighbors_dict[label][neighbor_label] = set()
# # Add the direction of this neighbor relative to current label
# neighbors_dict[label][neighbor_label].add(direction)

# return neighbors_dict


def color_segments(image, labels):
    output_image = np.copy(image)
    num_labels = np.max(labels) + 1
    for label in range(1, num_labels):
        mask = labels == label
        mean_color = np.mean(image[mask], axis=0)
        output_image[mask] = mean_color
    return output_image


def draw_flow_vectors(image, labels, flow, flow_threshold=2.0):
    num_labels = np.max(labels) + 1
    for label in range(1, num_labels):
        mask = labels == label
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        center_x, center_y = np.mean(xs), np.mean(ys)
        avg_flow = np.mean(flow[mask], axis=0)
        flow_x, flow_y = avg_flow[0], avg_flow[1]

        if np.linalg.norm(avg_flow) > flow_threshold:
            cv2.arrowedLine(image, (int(center_x), int(center_y)),
                            (int(center_x + flow_x), int(center_y + flow_y)),
                            (0, 255, 0), 2)
    return image


def draw_flow_blob(image, labels, flow, flow_threshold=2.0):
    num_labels = np.max(labels) + 1
    for label in range(1, num_labels):
        mask = labels == label
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        center_x, center_y = np.mean(xs), np.mean(ys)
        avg_flow = np.mean(flow[mask], axis=0)
        flow_x, flow_y = avg_flow[0], avg_flow[1]

        if np.linalg.norm(avg_flow) > flow_threshold:
            cv2.arrowedLine(image, (int(center_x), int(center_y)),
                            (int(center_x + flow_x), int(center_y + flow_y)),
                            (0, 255, 0), 2)
    return image


def draw_bounding_boxes(image, labels, flow, flow_threshold=2.0):
    """Draw bounding boxes over segments where flow > threshold."""
    num_labels = np.max(labels) + 1
    for label in range(1, num_labels):
        mask = labels == label
        if np.mean(np.linalg.norm(flow[mask], axis=1)) > flow_threshold:
            ys, xs = np.where(mask)
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0),
                          1)
    return image


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


def draw_quadrangles_in_BEV(bev_image, bev_points):
    """Draw quadrangles (projected bounding boxes) in the BEV."""
    bev_points = np.int32(bev_points)
    cv2.polylines(bev_image, [bev_points],
                  isClosed=True,
                  color=(0, 255, 255),
                  thickness=2)
    return bev_image


def add_padding(image, padding_size):
    """
    Adds padding to an image.
    """
    height, width = image.shape[:2]
    padded_image = np.zeros(
        (height + 2 * padding_size, width + 2 * padding_size, 3),
        dtype=np.uint8)
    padded_image[padding_size:padding_size + height,
                 padding_size:padding_size + width] = image
    return padded_image


def map_points_to_BEV_with_padding(box_corners, ipm_matrix, padding_size):
    """
    Maps bounding box corners to BEV coordinates and accounts for padding in the BEV image.
    """
    bev_points = map_points_to_BEV(box_corners, ipm_matrix)
    # Adjust points for padding
    bev_points = [(x + padding_size, y + padding_size) for x, y in bev_points]
    return bev_points


def compute_bounding_box_centroid(corners):
    """
    Computes the centroid of a bounding box from its corner coordinates.
    """
    x_coords = [p[0] for p in corners]
    y_coords = [p[1] for p in corners]
    centroid_x = sum(x_coords) / len(x_coords)
    centroid_y = sum(y_coords) / len(y_coords)
    return centroid_x, centroid_y


def match_bounding_boxes(current_boxes, previous_boxes, threshold=50):
    """
    Match bounding boxes across frames based on centroid distance.
    Returns a list of matched box pairs (current_box, previous_box).
    """
    matches = []
    for cur_box in current_boxes:
        cur_centroid = compute_bounding_box_centroid(cur_box)
        best_match = None
        min_distance = float('inf')

        for prev_box in previous_boxes:
            prev_centroid = compute_bounding_box_centroid(prev_box)
            distance = np.sqrt((cur_centroid[0] - prev_centroid[0])**2 +
                               (cur_centroid[1] - prev_centroid[1])**2)

            if distance < threshold and distance < min_distance:
                min_distance = distance
                best_match = prev_box

        if best_match:
            matches.append((cur_box, best_match))

    return matches


def draw_bottom(image, lower_face, color=cv_colors.RED.value, thickness=1):
    # Draw the lower face (as a quadrilateral)
    cv2.polylines(image, [lower_face],
                  isClosed=True,
                  color=color,
                  thickness=thickness)


def generate_random_color():
    """Generates a random color and returns it in both BGR (for cv2) and RGB (for plt) formats."""
    color_rgb = np.random.randint(
        0, 256, 3).tolist()  # Random RGB color in 0-255 range
    color_bgr = color_rgb[::-1]  # Reverse for BGR format for cv2
    color_rgb_normalized = [c / 255.0 for c in color_rgb]  # Normalize for plt
    return color_bgr, color_rgb_normalized


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


def compute_avg_cube(lower_face, upper_face):
    """
    Compute an average cube from the lower and upper faces, adjusting for mismatched indices and ensuring vertical lines.
    
    Args:
    - lower_face: np.array of shape (4, 2), lower face of the cube.
    - upper_face: np.array of shape (4, 2), upper face of the cube.
    
    Returns:
    - avg_lower_face: np.array of shape (4, 2), the averaged lower face.
    - avg_upper_face: np.array of shape (4, 2), the averaged upper face.
    """
    lower_face = np.array(lower_face)
    upper_face = np.array(upper_face)

    # Define index mapping between lower and upper face
    index_mapping = [(0, 0), (1, 1), (2, 2), (3, 3)]

    avg_lower_face = []
    avg_upper_face = []

    # Compute the average cube by adjusting the coordinates
    for lower_idx, upper_idx in index_mapping:
        # Get the lower and upper points
        lower_point = lower_face[lower_idx]
        upper_point = upper_face[upper_idx]

        # Compute the average x-coordinate to make vertical lines vertical
        avg_x = (lower_point[0] + upper_point[0]) / 2

        # The lower face y remains the same (closer to the ground)
        avg_lower_face.append([avg_x, lower_point[1]])

        # The upper face y remains the same (higher up in the cube)
        avg_upper_face.append([avg_x, upper_point[1]])

    # Convert the averaged faces to numpy arrays
    avg_lower_face = np.array(avg_lower_face, dtype=np.int32)
    avg_upper_face = np.array(avg_upper_face, dtype=np.int32)

    return avg_lower_face, avg_upper_face


def compute_avg_cube_new(lower_face, upper_face):
    """
    Compute an average cuboid from the lower and upper face projections, ensuring vertical lines and aligned diagonals.
    
    Args:
    - lower_face: np.array of shape (4, 2), lower face of the cube projection.
    - upper_face: np.array of shape (4, 2), upper face of the cube projection.
    
    Returns:
    - avg_lower_face: np.array of shape (4, 2), the averaged lower face projection.
    - avg_upper_face: np.array of shape (4, 2), the averaged upper face projection.
    """
    lower_face = np.array(lower_face)
    upper_face = np.array(upper_face)

    # Calculate the center points of the lower and upper face projections
    lower_center = np.mean(lower_face, axis=0)
    upper_center = np.mean(upper_face, axis=0)

    # Calculate the average center
    avg_center = (lower_center + upper_center) / 2

    avg_lower_face = []
    avg_upper_face = []

    # Adjust each pair of corresponding points
    for i in range(4):
        # Compute average x-coordinates for vertical alignment
        avg_x = (lower_face[i, 0] + upper_face[i, 0]) / 2

        # Use the original y-coordinates for the lower and upper faces
        avg_lower_point = [avg_x, lower_face[i, 1]]
        avg_upper_point = [avg_x, upper_face[i, 1]]

        avg_lower_face.append(avg_lower_point)
        avg_upper_face.append(avg_upper_point)

    # Ensure that diagonals of both faces intersect at the avg_center
    avg_lower_face = np.array(avg_lower_face)
    avg_upper_face = np.array(avg_upper_face)

    # Force the diagonals to intersect at the same center
    avg_lower_face = adjust_to_center(avg_lower_face, avg_center)
    avg_upper_face = adjust_to_center(avg_upper_face, avg_center)

    return avg_lower_face, avg_upper_face


def adjust_to_center(face, center):
    """
    Adjust the vertices of the face projection so that the diagonals intersect at the center.
    
    Args:
    - face: np.array of shape (4, 2), the face to adjust.
    - center: np.array of shape (2,), the center point where diagonals should intersect.
    
    Returns:
    - adjusted_face: np.array of shape (4, 2), the adjusted face with aligned diagonals.
    """
    adjusted_face = []

    for point in face:
        # Adjust the point's distance to maintain symmetry around the center
        adjusted_point = center + (point - center)
        adjusted_face.append(adjusted_point)

    return np.array(adjusted_face, dtype=np.int32)


def draw_bounding_box_for_blob(image, flow, component, flow_threshold=2.0):
    """Draw a single bounding box around the entire moving blob and its average flow direction."""

    if np.any(component):
        ys, xs = np.where(component)
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)

        # Draw the bounding box around the blob
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
                      cv_colors.GREEN.value, 2)

        # Calculate the center of the bounding box (center of the blob)
        center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2

        # Calculate the average flow within the blob
        avg_flow = np.mean(flow[ys, xs],
                           axis=0)  # flow[y, x] corresponds to (dx, dy)
        flow_x, flow_y = avg_flow[0], avg_flow[1]

        # Scale the flow vector for better visualization
        scale = 50
        flow_x_scaled, flow_y_scaled = scale * flow_x, scale * flow_y

        # Only draw the flow arrow if the average flow magnitude is above the threshold
        if np.linalg.norm(avg_flow) > flow_threshold:
            cv2.arrowedLine(
                image, (int(center_x), int(center_y)),
                (int(center_x + flow_x_scaled), int(center_y + flow_y_scaled)),
                cv_colors.RED.value, 2)

    return image


def convex_hull_of_faces(lower_face, upper_face):
    """
    Compute the convex hull from the combined lower and upper face points.
    
    Args:
    - lower_face: np.array of shape (4, 2), the lower face of the cube projection.
    - upper_face: np.array of shape (4, 2), the upper face of the cube projection.
    
    Returns:
    - convex_hull: np.array of shape (N, 2), the convex hull points of the cube.
    """
    # Combine lower and upper face points
    points = np.vstack([lower_face, upper_face])

    # Compute convex hull
    hull = cv2.convexHull(points)
    return hull


def compute_iou(mask, convex_hull_mask):
    """
    Compute the Intersection over Union (IoU) between two binary masks.
    
    Args:
    - mask: np.array, the original binary mask.
    - convex_hull_mask: np.array, the convex hull binary mask.
    
    Returns:
    - iou: float, the IoU between the two masks.
    """
    intersection = np.logical_and(mask, convex_hull_mask)
    union = np.logical_or(mask, convex_hull_mask)

    iou = np.sum(intersection) / np.sum(union)
    return iou


def create_mask(image_shape, points):
    """
    Create a binary mask from a set of points.
    
    Args:
    - image_shape: tuple, shape of the mask (height, width).
    - points: np.array of shape (N, 2), the points to fill in the mask.
    
    Returns:
    - mask: np.array, binary mask filled according to the points.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask


# Function to convert optical flow to a visual image (HSV or magnitude/angle representation)
def flow_to_image(flow):
    """Convert dense optical flow to an image-like format for SLIC segmentation."""
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    # Set HSV channels
    hsv[..., 0] = np.uint8(angle * 180 / np.pi / 2)  # Hue: angle
    hsv[..., 1] = 255  # Saturation
    hsv[...,
        2] = np.uint8(cv2.normalize(magnitude, None, 0, 255,
                                    cv2.NORM_MINMAX))  # Value: magnitude

    # Convert HSV to BGR for display or further processing
    flow_image_bev = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_image_bev


# # Utility function to check face overlap in BEV
# def faces_overlap(face1, face2):
# # TODO that is not true
# """ Check if two faces overlap in the image or BEV space """
# # Simple overlap condition: check bounding box intersection for the two faces
# # x_min1, x_max1 = np.min(face1[:, 0]), np.max(face1[:, 0])
# # y_min1, y_max1 = np.min(face1[:, 1]), np.max(face1[:, 1])
# # x_min2, x_max2 = np.min(face2[:, 0]), np.max(face2[:, 0])
# # y_min2, y_max2 = np.min(face2[:, 1]), np.max(face2[:, 1])

# # return not (x_max1 < x_min2 or x_min1 > x_max2 or y_max1 < y_min2 or y_min1 > y_max2)
# p1 = Polygon([(rect1[0],rect1[1]), (rect1[1],rect1[1]),(rect1[2],rect1[3]),(rect1[2],rect1[1])])
# p2 = Polygon([(rect2[0],rect2[1]), (rect2[1],rect2[1]),(rect2[2],rect2[3]),(rect2[2],rect2[1])])
# return(p1.intersects(p2))


# Utility function to check face overlap in BEV
def faces_overlap(face1, face2):
    """
    Check if two faces overlap in the BEV or image space.

    Parameters:
    - face1, face2: Each should be an array of four corner points, 
                    each point formatted as (x, y).

    Returns:
    - True if the faces overlap, False otherwise.
    """
    # Define polygons for each face based on the corner points
    p1 = Polygon([face1[0], face1[1], face1[2], face1[3]])
    p2 = Polygon([face2[0], face2[1], face2[2], face2[3]])

    # Check if the polygons intersect (overlap)
    return p1.intersects(p2)


# def faces_overlap(face1, face2, epsilon=10.0):
# """
# Check if two faces overlap in the BEV or image space, with optional epsilon proximity.

# Parameters:
# - face1, face2: Arrays of four corner points for each face, formatted as [(x, y), ...].
# - epsilon: Distance buffer around each polygon for proximity overlap.

# Returns:
# - True if the faces overlap or are within epsilon distance, False otherwise.
# """
# # Define polygons for each face based on the corner points
# p1 = Polygon([face1[0], face1[1], face1[2], face1[3]])
# p2 = Polygon([face2[0], face2[1], face2[2], face2[3]])

# # Apply the epsilon buffer to both polygons
# p1_buffered = p1.buffer(epsilon)
# p2_buffered = p2.buffer(epsilon)

# # Check if the buffered polygons intersect
# return p1_buffered.intersects(p2_buffered)


def combined_segmentation(next_frame, flow, region_size, ruler,
                          slic_iterations):
    # Perform superpixel segmentation on the image frame
    slic_image = cv2.ximgproc.createSuperpixelSLIC(
        next_frame,
        algorithm=cv2.ximgproc.MSLIC,
        region_size=region_size,
        ruler=ruler)
    slic_image.iterate(slic_iterations)
    mask_image = slic_image.getLabelContourMask()
    labels_image = slic_image.getLabels()

    # Convert optical flow to an image-like format for segmentation
    flow_image_bev = flow_to_image(flow)

    region_size = 20  #200 #100 #15  #10#30
    ruler = 10  #150 # 100 #20 #14
    # Perform superpixel segmentation on the flow image
    slic_flow = cv2.ximgproc.createSuperpixelSLIC(flow_image_bev,
                                                  algorithm=cv2.ximgproc.MSLIC,
                                                  region_size=region_size,
                                                  ruler=ruler)
    slic_flow.iterate(slic_iterations)
    mask_flow = slic_flow.getLabelContourMask()
    labels_flow = slic_flow.getLabels()

    # Combine the segmentations
    combined_labels = combine_labels(labels_image, labels_flow)

    return combined_labels, mask_image, mask_flow


def combine_labels(labels_image, labels_flow):
    """Combine labels from image and flow segmentations"""
    # Combine segmentations, here we use element-wise minimum as an example.
    combined_labels = np.minimum(labels_image, labels_flow)
    return combined_labels


def bev_labels_to_plain(labels_bev, inv_ipm_matrix):
    """
    Converts BEV labels to plain projection using inverse mapping.

    Args:
    - labels_bev (numpy array): BEV labels.
    - inv_ipm_matrix (numpy array): Inverse perspective transformation matrix.

    Returns:
    - labels_plain (numpy array): Labels projected onto the original plane.
    """
    # Initialize an empty array for the plain-projected labels
    labels_plain = np.zeros_like(labels_bev)

    # Loop through unique labels in the BEV
    for label in np.unique(labels_bev):
        if label == 0:
            continue  # Skip background if 0 is background

        # Extract all points belonging to the current label in BEV
        points_yx = np.column_stack(np.where(labels_bev == label))

        # Add homogeneous coordinate for transformation
        points_bev_homogeneous = np.hstack(
            [points_yx, np.ones((points_yx.shape[0], 1))])

        # Apply inverse IPM to map points back to plain projection
        points_plain_homogeneous = points_bev_homogeneous @ inv_ipm_matrix.T
        points_plain = points_plain_homogeneous[:, :
                                                2] / points_plain_homogeneous[:,
                                                                              2][:,
                                                                                 None]

        # Convert back to integer coordinates
        points_plain = np.round(points_plain).astype(int)

        # Ensure coordinates are within the plain image bounds
        points_plain = points_plain[
            (points_plain[:, 0] >= 0)
            & (points_plain[:, 0] < labels_plain.shape[1]) &
            (points_plain[:, 1] >= 0) &
            (points_plain[:, 1] < labels_plain.shape[0])]

        # Update the labels in the plain-projected image
        labels_plain[points_plain[:, 1], points_plain[:, 0]] = label

    return labels_plain


# Define the recursive function to traverse and update segment properties from the bottom upwards
def traverse_segments_from_bottom(segment_label, segments_dict, neighbors_dict,
                                  visited):
    if segment_label in visited:
        return segments_dict[segment_label]['height'], segments_dict[
            segment_label]['cv2_color'], segments_dict[segment_label][
                'plt_color'], segments_dict[segment_label]['z']

    segment = segments_dict[segment_label]
    visited.add(segment_label)

    max_height = 0
    top_color = segment['cv2_color']
    top_plt_color = segment['plt_color']
    max_z = 0

    for neighbor_label, relative_positions in neighbors_dict[
            segment_label].items():
        if 'top' in relative_positions or 'top-left' in relative_positions or 'top-right' in relative_positions:
            neighbor_height, neighbor_color, neighbor_plt_color, neighbor_z = traverse_segments_from_bottom(
                neighbor_label, segments_dict, neighbors_dict, visited)

            if neighbor_z + neighbor_height > max_z + max_height:
                max_height = neighbor_height
                max_z = neighbor_z
                top_color = neighbor_color
                top_plt_color = neighbor_plt_color

    segment['height'] = max_height + segment['height']
    segment['cv2_color'] = top_color
    segment['plt_color'] = top_plt_color
    segment['z'] = max_z + segment['height']

    return segment['height'], segment['cv2_color'], segment[
        'plt_color'], segment['z']


def join_segmentations_cv2(labels1, labels2):
    """
    Mimics skimage's join_segmentations by creating a combined label map 
    based on unique pairs of labels from two input segmentations.
    """
    # Ensure labels1 and labels2 are the same shape
    if labels1.shape != labels2.shape:
        raise ValueError("Input label arrays must have the same shape.")

    # Generate unique label pairs by combining the two label maps
    max_label1 = np.max(labels1) + 1
    combined_labels = labels1 * max_label1 + labels2

    # Assign new unique labels to each unique pair
    unique_labels, new_labels = np.unique(combined_labels, return_inverse=True)

    # Reshape the new labels to match the original image shape
    joined_labels = new_labels.reshape(labels1.shape)

    return joined_labels

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

def reenumerate_connected_labels(labels):
    unique_labels = np.unique(labels)
    max_label = 0
    output_labels = np.zeros_like(labels, dtype=np.int32)

    # Re-label each unique segment
    for label in unique_labels:
        if label == 0:  # skip background label
            continue
        mask = (labels == label).astype(np.uint8)
        num_labels, labeled_mask = cv2.connectedComponents(mask)
        
        # Assign new labels starting from the maximum label used so far
        for component in range(1, num_labels):
            output_labels[labeled_mask == component] = max_label + component
        max_label += num_labels - 1

    return output_labels


def main(video_path, draw_boundaries=True, debug=False):

    ipm_matrix, inv_mat, target_shape = generate_perspective_matrix(ratio = 0.5)

    cap = cv2.VideoCapture(input_video_path)
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) // 2
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Can't read video.")
        cap.release()
        return

    prev_bev = cv2.warpPerspective(prev_frame,
                                   ipm_matrix,
                                   target_shape,
                                   flags=cv2.INTER_CUBIC)
    previous_flow_bev = np.zeros((prev_bev.shape[0], prev_bev.shape[1], 2), dtype=np.float32)
    avg_frame = compute_average_frame(video_path, 'avg_frame.png')

    avg_bev = cv2.warpPerspective(avg_frame,
                                  ipm_matrix,
                                  target_shape,
                                  flags=cv2.INTER_LANCZOS4)  #,
    # Display the result
    # cv2.imshow("AVG BEV",
               # resize_to_match_height(avg_bev, screen_height))
    # cv2.waitKey(0)
    #borderMode=cv2.BORDER_TRANSPARENT)

    frame_count = 0

    while True:
        ret, next_frame = cap.read()
        frame_count += 1

        if not ret:
            break

        print(f"Processing {frame_count} frame")
        if frame_count < 75:
            prev_frame = next_frame
            #prev_bev = next_bev
            continue

        #if frame_count % 2 != 0: continue

        # Prepare BEV for next frame
        prev_bev = cv2.warpPerspective(prev_frame,
                                       ipm_matrix,
                                       target_shape,
                                       flags=cv2.INTER_CUBIC)
        next_bev = cv2.warpPerspective(next_frame,
                                       ipm_matrix,
                                       target_shape,
                                       flags=cv2.INTER_CUBIC)
                                       

        # Compute the difference and create the moving mask
        diff = cv2.absdiff(next_bev, avg_bev)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        ratio = 0.1
        blurred_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
        _, moving_mask = cv2.threshold(blurred_diff, 20, 255, cv2.THRESH_BINARY)
    
        #moving_mask = cv2.dilate(moving_mask, None, iterations=1)
        #moving_mask_bin = moving_mask // 255
        # Get connected components within the mask
        num_labels, labels = cv2.connectedComponents(moving_mask)

        # Create an empty color image to display components
        colored_components = np.zeros((*moving_mask.shape, 3), dtype=np.uint8)

        # Assign a unique color to each label
        for label in range(
                1, num_labels):  # Start from 1 to ignore the background
            mask = (labels == label)
            color = np.random.randint(0, 255, 3)  # Generate a random color
            colored_components[mask] = color  # Apply color to the component

        # Display the result
        cv2.imshow("Connected Components",
                   resize_to_match_height(colored_components, screen_height))
        #cv2.waitKey(0)

        # Display the moving mask
        cv2.imshow('Moving Mask',
                   resize_to_match_height(moving_mask, screen_height))
        #cv2.waitKey(0)
        
        if False and debug:
            flow = compute_optical_flow(prev_frame, next_frame)
            flow_image = fz.convert_from_flow(flow)
            
            # Display the flow visualization
            cv2.imshow("Optical Flow ", resize_to_match_height(flow_image, screen_height//2))
            cv2.waitKey(0)

        # Compute optical flow and warp it to BEV
        flow_bev = compute_optical_flow(prev_bev, next_bev)
        smoothed_flow = (previous_flow_bev + flow_bev) / 2  # Adjust to average over more frames if needed
        #flow_image_bev = flow_to_image(flow)
        flow_image_bev = fz.convert_from_flow(smoothed_flow)

        # Set pixels outside the moving mask to white
        #flow_image_bev[moving_mask == 0] = cv_colors.WHITE.value  #[255, 255, 255]

        # Display the flow visualization
        cv2.imshow("Optical Flow Visualization",
                   resize_to_match_height(flow_image_bev, screen_height))
        #cv2.waitKey(0)

        region_size = 2*5*10  #200 #100 #15  #10#30
        ruler = 2*5*10  #150 # 100 #20 #14

        slic1 = cv2.ximgproc.createSuperpixelSLIC(flow_image_bev,
                                                  algorithm=cv2.ximgproc.MSLIC,
                                                  region_size=region_size,
                                                  ruler=ruler)
        slic1.iterate(slic_iterations)
        #slic1.enforceLabelConnectivity(min_element_size=region_size)

        slic2 = cv2.ximgproc.createSuperpixelSLIC(next_bev,
                                                  algorithm=cv2.ximgproc.MSLIC,
                                                  region_size=region_size,
                                                  ruler=ruler)
        slic2.iterate(slic_iterations)
        #slic1.enforceLabelConnectivity(min_element_size=region_size)

        labels1 = slic1.getLabels() + 1
        labels2 = slic2.getLabels() + 1

        labels = join_segmentations_cv2(labels1, labels2)
        num_labels = np.max(labels) + 1
        
        # If debug is enabled

        bev_output = flow_image_bev.copy()
        contour_mask1 = slic1.getLabelContourMask(False)
        contour_mask2 = slic2.getLabelContourMask(False)
        bev_output[0 < contour_mask1] = cv_colors.RED.value
        bev_output[0 < contour_mask2] = cv_colors.BLUE.value
        #cv2.imshow("Countour mask", bev_image)

        if False and debug:        
            # labels1[moving_mask == 0] = 0 
            # labels2[moving_mask == 0] = 0
            # labels[moving_mask == 0] = 0 
            # Convert labels to color images for visualization
            # labels1_color = labels_to_color(labels1)
            # labels2_color = labels_to_color(labels2)
            labels_color = labels_to_color(labels)
        
            # Display each segmentation using cv2
            # cv2.imshow("SLIC1 Segmentation", labels1_color)
            # cv2.imshow("SLIC2 Segmentation", labels2_color)
            cv2.imshow("Joint Segmentation", labels_color)
        
            # Print statistics
            print("Number of unique labels in SLIC1:", np.unique(labels1).size)
            print("Number of unique labels in SLIC2:", np.unique(labels2).size)
            print("Number of unique labels in joint segmentation:", np.unique(labels).size)
            print("Total labels in joint segmentation:", num_labels)
        
            # Wait until any key is pressed, then close all windows
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # Apply connected components labeling to separate unconnected components
        #num_labels, labels = cv2.connectedComponents(labels.astype(np.uint8), connectivity=8)

        # Remove labels outside the moving mask by setting them to 0
        #labels[moving_mask == 0] = 0  # Set areas outside the moving mask to label 0

        labels = reenumerate_connected_labels(labels)
        num_labels = np.max(labels)
        # Debug visualization (optional)
        if False and debug:
            labels_combined_color = labels_to_color(labels)
            cv2.imshow("Re-enumerated Combined Segmentation", labels_combined_color)
            print("Total unique labels in combined segmentation:", num_labels)
        
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # contour_mask = slic1.getLabelContourMask(False)
        # bev_image = flow_image_bev.copy()
        # bev_image[0 < contour_mask] = cv_colors.WHITE.value
        # cv2.imshow('getLabelContourMask', resize_to_match_height(result, screen_height))
        # cv2.waitKey(0)

        #colored_segments = color_segments(flow_image_bev.copy(), labels)
        bounding_box_image = next_frame.copy()
        #neighbors_dict =  find_neighbors(flow_image_bev.copy(), labels)

        neighbors_dict = find_neighbors_within_mask(labels)
        # alpha = 0.5
        # bounding_box_image = cv2.addWeighted(next_frame.copy(), 1 - alpha,
        # bounding_box_image, alpha, 0)

        # cv2.imshow('Segmentation',
                   # resize_to_match_height(bev_image, screen_height))

        # colored_image = color_neighbors(flow_image_bev.copy(), labels, neighbors_dict, moving_mask)
        # cv2.imshow('Colored', resize_to_match_height(colored_image, screen_height))
        # cv2.waitKey(0)

        #cv2.destroyAllWindows()

        # prev_frame = next_frame
        # continue

        segments = []

        lower_faces = []  # Initialize as an empty list
        heights = []  # Similarly, initialize upper_faces if needed
        colors = []

        for label in range(1, num_labels):
            mask = (labels == label)

            # # Find the coordinates of non-zero pixels in the mask
            # points = np.column_stack(np.where(mask.astype(np.uint8) > 0))
        
            # # Check if there are enough points to form a hull
            # if len(points) < 3:
                # continue  # Convex hull requires at least 3 points
        
            # # Compute the convex hull of these points
            # hull_points = cv2.convexHull(points)
        
            # # Create an empty mask to draw the convex hull
            # convex_hull_mask = np.zeros_like(mask, dtype=np.uint8)
        
            # # Draw the convex hull as a filled polygon
            # cv2.fillConvexPoly(convex_hull_mask, hull_points, 255)
            # mask = convex_hull_mask


            # Calculate the total number of pixels in this component
            total_component_pixels = np.sum(mask)
            # Calculate the number of moving pixels in this component
            moving_pixels_in_component = np.sum(moving_mask[mask]) / 255

            # Calculate the percentage of moving pixels in this component
            moving_percentage = moving_pixels_in_component / total_component_pixels

            if moving_percentage >= 0.5:
                # Find the bounding box for the current segment
                ys, xs = np.where(mask)
                if xs.size > 0 and ys.size > 0:
                    x_min, x_max = np.min(xs), np.max(xs)
                    y_min, y_max = np.min(ys), np.max(ys)
                    top_point = (xs[np.argmin(ys)], y_min)  # Top-most point
                    bottom_point = (xs[np.argmax(ys)], y_max
                                    )  # Bottom-most point
                    left_point = (x_min, ys[np.argmin(xs)])  # Left-most point
                    right_point = (x_max, ys[np.argmax(xs)]
                                   )  # Right-most point

                    # Extract the flow vectors inside the box
                    flow_in_box = smoothed_flow[mask]  # Flow vectors within the box
                    # Compute the average flow vector in the box
                    avg_flow_vector = np.mean(flow_in_box, axis=0)

                    # Store segment data (bounding box, mask, etc.)
                    segments.append({
                        'label': label,
                        'avg_flow_vector': avg_flow_vector,
                        'top_point': top_point,
                        'bottom_point': bottom_point,
                        'left_point': left_point,
                        'right_point': right_point,
                        'height': None,  # Store the height of the segment
                        'bottom_center': None,  # check IV
                        'top_center': None,  # check IV
                        'upper_face': None,
                        'bottom': None,
                        'top': None,
                        'corner_ind': None,
                        'cv2_color': None,
                        'plt_color': None
                    })

        # Sort segments by their vertical position (y_min), bottom to top
        segments = sorted(segments,
                          key=lambda s: (s['bottom_point'], s['top_point']),
                          reverse=True)

        for i, segment in enumerate(segments):
            segment_label = segment['label']
            if segment_label == 0: continue
            # Generate a unique random color for this segment
            cv2_color, plt_color = generate_random_color()

            top_point = projection_on_bird(inv_mat, segment['top_point'])
            bottom_point = projection_on_bird(inv_mat, segment['bottom_point'])
            left_point = projection_on_bird(inv_mat, segment['left_point'])
            right_point = projection_on_bird(inv_mat, segment['right_point'])

            x_min = left_point[0]
            x_max = right_point[0]
            y_min = top_point[1]
            y_max = bottom_point[1]

            avg_flow_vector = segment['avg_flow_vector']

            box_corners = [[x_min, y_max], [x_min, y_min], [x_max, y_min],
                           [x_max, y_max]]

            # Compute the BEV flow vector
            avg_flow_vector_bev = avg_flow_vector

            # Compute the angle/orientation of the box based on the BEV flow vector
            avg_flow_angle = np.arctan2(
                avg_flow_vector_bev[1],
                avg_flow_vector_bev[0])  # Angle in radians

            avg_flow_angle_degrees = np.degrees(
                avg_flow_angle)  # Convert to degrees

            print(
                f"Box {label}: Average flow angle (degrees): {avg_flow_angle_degrees}"
            )

            bev_points = map_points_to_BEV(box_corners, ipm_matrix)

            bev_points_iv = to_iv(bev_points)

            if False and debug:
                draw_bottom(bounding_box_image,
                            np.array(box_corners, dtype=np.int32),
                            color=cv_colors.BLUE.value,
                            thickness=2)
                draw_bottom(bev_image,
                            np.array(bev_points, dtype=np.int32),
                            color=cv_colors.BLUE.value,
                            thickness=1)

            angle_radians = move_to_second_quadrant(-avg_flow_angle)

            bottom_point_bev = projection_on_bird(ipm_matrix, bottom_point)
            bottom_point_bev_iv = iv(bottom_point_bev)
            bottom_iv = compute_bottom_rectangle_with(
                bev_points_iv, angle_radians, bottom_point_bev_iv,
                debug)  # Ensure compute_bottom_rectangle is implemented

            x1 = np.linalg.norm(bottom_iv[1] - bottom_iv[2])
            y1 = np.linalg.norm(bottom_iv[2] - bottom_iv[3])
            bottom_center_iv = bottom_iv[0]
            bottom = to_iv(bottom_iv)
            center = bottom[0]

            bottom = bottom[1:]
            lower_face = map_points_to_BEV(bottom, inv_mat)
            e = 100
            if True or np.all((lower_face[:, 0] + e >= x_min)
                              & (lower_face[:, 0] <= x_max + e)
                              & (lower_face[:, 1] + e >= y_min)
                              & (lower_face[:, 1] <= y_max + e)):

                top_point_bev = projection_on_bird(ipm_matrix, top_point)
                top_point_bev_iv = iv(top_point_bev)

                top_iv = compute_top_rectangle_with(
                    bev_points_iv, -np.pi / 2 + angle_radians,
                    top_point_bev_iv,
                    debug)  # Ensure compute_top_rectangle is implemented
                if bottom is not None and top_iv is not None:

                    y2 = np.linalg.norm(top_iv[2] - top_iv[3])
                    x2 = np.linalg.norm(top_iv[3] - top_iv[4])

                    center_top_iv = top_iv[0]
                    top = to_iv(top_iv)
                    center_top = top[0]
                    #bottom = camera.convert_bev_to_world(bottom)
                    #det.centers = [[bottom[0], det.angle_radians]]

                    top = top[1:]
                    # TODO check if avg better
                    x = max(x1, x2)
                    y = max(y1, y2)

                    avg_bootom_iv, avg_corner_ind = compute_rectangle_points(
                        bottom_center_iv, angle_radians, y, x)
                    avg_top_iv, avg_corner_ind_1 = compute_rectangle_points(
                        center_top_iv, angle_radians, y, x)

                    assert avg_corner_ind == avg_corner_ind_1, "Top and bottom corner should be the same"

                    avg_bottom = to_iv(avg_bootom_iv)
                    avg_top = to_iv(avg_top_iv)

                    rot_bootom_iv, rot_corner_ind = compute_rectangle_points(
                        bottom_center_iv, 0, y, x)
                    rot_bottom = to_iv(rot_bootom_iv)
                    rot_lower_face = map_points_to_BEV(rot_bottom, inv_mat)

                    rot_top_iv, rot_corner_ind_1 = compute_rectangle_points(
                        center_top_iv, 0, y, x)
                    rot_top = to_iv(rot_top_iv)
                    rot_upper_face = map_points_to_BEV(rot_top, inv_mat)

                    assert rot_corner_ind == rot_corner_ind_1, "Top and bottom corner should be the same"

                    avg_lower_face = map_points_to_BEV(avg_bottom, inv_mat)

                    avg_upper_face = map_points_to_BEV(avg_top, inv_mat)

                    bev_length = np.linalg.norm(rot_bootom_iv[rot_corner_ind] -
                                                rot_bootom_iv[
                                                    (rot_corner_ind + 1) % 4])
                    proj_length = np.linalg.norm(
                        rot_lower_face[rot_corner_ind] -
                        rot_lower_face[(rot_corner_ind + 1) % 4])
                    plain_height = np.linalg.norm(
                        avg_lower_face[avg_corner_ind] -
                        avg_upper_face[avg_corner_ind])
                    scaling_factor = bev_length / proj_length
                    height = plain_height * scaling_factor

                    # cv2.line(bounding_box_image, rot_lower_face[rot_corner_ind].astype("int"), rot_lower_face[(rot_corner_ind + 1) % 4].astype("int"), cv_colors.GREEN.value, 5)
                    # cv2.line(bev_image, rot_bottom[rot_corner_ind].astype("int"), rot_bottom[(rot_corner_ind + 1) % 4].astype("int"), cv_colors.GREEN.value, 5)
                    # cv2.line(bounding_box_image, avg_lower_face[avg_corner_ind].astype("int"), avg_upper_face[avg_corner_ind].astype("int"), cv_colors.BLUE.value, 5)

                    segment['height'] = height
                    segment['top_center'] = center_top_iv
                    segment['bottom_center'] = bottom_center_iv
                    segment['upper_face'] = avg_upper_face
                    segment['corner_ind'] = rot_corner_ind
                    # segment['top'] = avg_top_iv
                    # segment['cv2_color'] = cv2_color
                    # segment['plt_color'] = plt_color

                    z = 0

                    # Loop through earlier segments (which are below in the image)
                    if False: #for j in range(i):
                        other_segment = segments[j]
                        other_label = other_segment['label']
                        other_top = other_segment['top']
                        other_bottom = other_segment['bottom']

                        # Check if other_segment is a neighbor of segment
                        if other_label in neighbors_dict[segment_label]:
                            relative_positions = neighbors_dict[segment_label][
                                other_label]
                            # cv2_color = other_segment['cv2_color']
                            # plt_color = other_segment['plt_color']

                            # Check if `other_segment` is directly above `segment`
                            if ('bottom' in relative_positions
                                    or 'bottom-left' in relative_positions
                                    or 'bottom-right' in relative_positions):
                                print(
                                    f"Segment {segment_label} is below neighbor segment {other_label} (which is above it)"
                                )

                                cv2_color = other_segment['cv2_color']
                                plt_color = other_segment['plt_color']

                                assert rot_corner_ind == other_segment['corner_ind'], "Top and bottom corner should be the same"

                                pts1 = np.float32(other_segment['upper_face'])   
                                pts2 = np.float32(to_iv(other_segment['bottom']))

                                # Generate the perspective transformation matrices
                                loc_persp = cv2.getPerspectiveTransform(pts1, pts2)
                                #loc_inv = cv2.getPerspectiveTransform(pts2, pts1)
                                
                                #avg_bootom_iv = to_iv(map_points_to_BEV(avg_lower_face, loc_persp))
                                avg_bootom_iv = map_points_to_BEV(avg_lower_face, loc_persp)

                                z = other_segment['z'] + other_segment['height']

                                break

                    segment['top'] = avg_top_iv
                    segment['bottom'] = avg_bootom_iv
                    segment['cv2_color'] = cv2_color
                    segment['plt_color'] = plt_color

                    # draw_bottom(
                    # bev_image,
                    # avg_bottom.astype("int"),
                    # color=cv2_color,  # cv_colors.BLUE.value,
                    # thickness=1)
                    # draw_bottom(bev_image,
                    # avg_top.astype("int"),
                    # color=cv_colors.GREEN.value,
                    # thickness=3)

                    draw_cube(
                        bev_output,
                        avg_bottom.astype("int"),
                        avg_top.astype("int"),
                        color=cv2_color,  #cv_colors.ORANGE.value,
                        thickness=1)

                    draw_cube(
                        bounding_box_image,
                        avg_lower_face.astype("int"),
                        avg_upper_face.astype("int"),
                        color=cv2_color,  #cv_colors.ORANGE.value,
                        thickness=2)


                    segment['z'] = z
                    #Extend avg_upper_face by adding height to each point
                    avg_bottom_3d = []
                    for point in avg_bootom_iv:
                        x_2d, y_2d = point
                        avg_bottom_3d.append(
                            [x_2d, y_2d, z])  # Add height as the third coordinate

                    avg_bottom_3d = np.array(
                        avg_bottom_3d)  # Convert to a NumPy array if needed
                    #segment['bottom'] = avg_bottom

                    #Collect lower and upper faces for 3D rendering
                    lower_faces.append(
                        avg_bottom_3d
                    )  # Lower face remains 2D (or use zeros for Z if required)
                    heights.append(height)  # Now a 3D face
                    colors.append(plt_color)

        out.write(bounding_box_image)
        resized_image = resize_to_match_height(bounding_box_image,
                                               screen_height)
        # Resize BEV image to match the height of bounding_box_image
        if show_warp:
            cv2.imshow('BEV', resize_to_match_height(bev_output, screen_height))
            # bev_image_resized = resize_to_match_height(
            # crop_warp(bev_image),
            # screen_height)  #bounding_box_image.shape[0])
            # combined_image = np.hstack((resized_image, bev_image_resized))
            # resized_image = combined_image  #cv2.resize(combined_image, (screen_width, screen_height))

        prev_frame = next_frame
        previous_flow_bev = flow_bev
        cv2.imshow('Bounding Boxes', resized_image)

        if cv2.waitKey(30) & 0xFF == 27:  # Esc key to stop
            break

        # Call your function to draw cubes
        if debug and frame_count == 80:
            draw_cubes_in_3d(lower_faces, heights, colors)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def draw_tracking_line(image,
                       prev_box,
                       cur_box,
                       color=(0, 0, 255),
                       thickness=2):
    """
    Draws a line between the centroids of the previous and current bounding boxes.
    
    Parameters:
    - image: The image on which to draw the line.
    - prev_box: The previous bounding box (as a list of four corner points).
    - cur_box: The current bounding box (as a list of four corner points).
    - color: The color of the line (default is green).
    - thickness: The thickness of the line (default is 2).
    """
    # Compute centroids for both bounding boxes
    prev_centroid = compute_bounding_box_centroid(prev_box)
    cur_centroid = compute_bounding_box_centroid(cur_box)

    # Draw the line between the centroids
    cv2.line(image, (int(prev_centroid[0]), int(prev_centroid[1])),
             (int(cur_centroid[0]), int(cur_centroid[1])), color, thickness)


def resize_to_match_height(image, target_height):
    height, width = image.shape[:2]
    scaling_factor = target_height / height
    new_width = int(width * scaling_factor)
    resized_image = cv2.resize(image, (new_width, target_height))
    return resized_image


if __name__ == "__main__":
    # /home/dmytrozhuravlov/cv/data/
    # '/home/dzhura/mount/cv/data/
    video_path = input_video_path
    main(video_path, draw_boundaries=True, debug=True)
