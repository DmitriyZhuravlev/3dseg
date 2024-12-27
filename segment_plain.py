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

import os



# Define your capture source and output directory
output_dir = "output_frames"
output_bev_dir = "output_bev_frames"
os.makedirs(output_dir, exist_ok=True)

# Initialize video capture and output
# /home/dmytrozhuravlov/cv/data/
# '/home/dzhura/mount/cv/data/
# /home/dzhura/ComputerVision/data
input_video_path = '/home/dmytrozhuravlov/cv/data/4kStreetViewCctv.mp4'
output_video_path = 'output_with_cubes.mp4'

# vertical vp
line1 = [[1132,11], [1210,843]]
line2 = [[360,16], [503, 867]]

# left horizontal vp 
line3 = [[183, 545], [772, 645]]
line4 = [[429, 427], [1094, 443]]

# right horizontal vp 
line5 = [[1170, 385], [1206, 348]]
line6 = [[1140, 83], [1213, 96]]


# Define input lines as NumPy arrays
lines = {
    "vertical_lines": np.array([[[1132, 11], [1210, 843]], [[360, 16], [503, 867]]]),
    "left_hor_lines": np.array([[[183, 545], [772, 645]], [[429, 427], [1094, 443]]]),
    "right_hor_lines": np.array([[[1169, 385], [1204, 347]], [[1140, 83], [1213, 95]]]),
}

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

# Compute the ground intersection (origin) using the left horizontal and vertical VPs
origin = compute_intersection(
    [[hor_left_vp[0], hor_left_vp[1]], [hor_right_vp[0], hor_right_vp[1]]],
    [[vert_vp[0], vert_vp[1]], [hor_right_vp[0], hor_right_vp[1]]],
)

def line_intersection_with_bounds(point1, point2, img_shape):
    """
    Find the intersection of a line with the image boundaries.
    Args:
        point1: Starting point of the line (x1, y1).
        point2: Ending point of the line (x2, y2).
        img_shape: Shape of the image (height, width).
    Returns:
        List of valid intersection points (x, y) on the image boundary.
    """
    h, w = img_shape[:2]

    # Define the image boundaries as lines
    boundaries = [
        [[0, 0], [w, 0]],  # Top edge
        [[0, 0], [0, h]],  # Left edge
        [[0, h], [w, h]],  # Bottom edge
        [[w, 0], [w, h]],  # Right edge
    ]

    def intersect(line1, line2):
        l1 = np.cross(
            [line1[0][0], line1[0][1], 1], [line1[1][0], line1[1][1], 1]
        )
        l2 = np.cross(
            [line2[0][0], line2[0][1], 1], [line2[1][0], line2[1][1], 1]
        )
        intersection = np.cross(l1, l2)
        if intersection[2] != 0:
            return [intersection[0] / intersection[2], intersection[1] / intersection[2]]
        return None

    intersections = []
    for boundary in boundaries:
        intersect_point = intersect([point1, point2], boundary)
        if intersect_point:
            # Check if the intersection is within the boundary segment
            x, y = intersect_point
            if 0 <= x <= w and 0 <= y <= h:
                intersections.append([x, y])

    return intersections


def draw_line_segment(image, point1, point2, color, thickness):
    """
    Draw a line segment between two points, clipped to the image boundary.
    Args:
        image: Input image to draw on.
        point1: Starting point of the line (x1, y1).
        point2: Ending point of the line (x2, y2).
        color: Line color (BGR tuple).
        thickness: Line thickness.
    Returns:
        Modified image with the line drawn.
    """
    intersections = line_intersection_with_bounds(point1, point2, image.shape)
    if len(intersections) >= 2:  # Draw only if both endpoints are valid
        cv2.line(
            image,
            tuple(map(int, intersections[0])),
            tuple(map(int, intersections[1])),
            color,
            thickness,
        )
    return image


def draw_3d_grid(image, vert_vp, hor_left_vp, hor_right_vp, grid_size=10):
    """
    Draw a 3D grid on an image using the provided vanishing points.
    Args:
        image: Input image to draw on.
        vert_vp: Vertical vanishing point [x, y].
        hor_left_vp: Left horizontal vanishing point [x, y].
        hor_right_vp: Right horizontal vanishing point [x, y].
        grid_size: Number of grid lines to draw.
    Returns:
        Image with the 3D grid drawn.
    """
    h, w, _ = image.shape
    grid_image = image.copy()

    # Bottom edge of the frame (assume as ground plane)
    bottom_line = [[0, h], [w, h]]

    # Draw vertical lines converging to the vertical vanishing point
    for i in np.linspace(0, w, grid_size):
        line_start_i = [i, h]  # Start from the bottom
        grid_image = draw_line_segment(grid_image, line_start_i, vert_vp, (0, 255, 0), 1)
        grid_image = draw_line_segment(grid_image, line_start_i, hor_right_vp, (0, 0, 255), 1)
        grid_image = draw_line_segment(grid_image, [i, 0], hor_right_vp, (0, 0, 255), 1)

    # Draw horizontal lines converging to left and right horizontal vanishing points
    for j in np.linspace(0, h, grid_size):
        line_start_j = [0, j]  # From left to the horizontal VP
        grid_image = draw_line_segment(grid_image, line_start_j, hor_left_vp, (255, 0, 0), 1)
        grid_image =  cv2.line(grid_image, [0, int(j)], [hor_right_vp[0].astype("int"), hor_right_vp[1].astype("int")], (0, 0, 255), 1) #draw_line_segment(grid_image, [0, j], hor_right_vp, (0, 0, 255), 1)
            
            # if j % grid_size == 0:
                # line_start = compute_intersection([line_start_j, hor_left_vp], [line_start_i, vert_vp])
                # if line_start is not None:
                   # grid_image = draw_line_segment(grid_image, line_start, hor_right_vp, (0, 0, 255), 1)

    return grid_image
    
def draw_grid_lines(image, origin, van_point, grid_size, color, thickness):
    """
    Draws grid lines between an origin and a vanishing point.
    Args:
        image: Input image.
        origin: Origin point (x, y).
        van_point: Vanishing point (x, y).
        grid_size: Number of grid lines to draw.
        color: Line color (BGR).
        thickness: Line thickness.
    Returns:
        Image with grid lines.
    """
    h, w = image.shape[:2]
    grid_image = image.copy()

    # Draw lines from the origin to the vanishing point
    for i in range(grid_size):
        # Offset the starting point along the x and y axes
        offset_x = i * w // grid_size
        offset_y = i * h // grid_size

        # Draw lines to vanishing point
        cv2.line(
            grid_image,
            (int(origin[0] - offset_x), int(origin[1])),
            (int(van_point[0]), int(van_point[1])),
            color,
            thickness,
        )
        cv2.line(
            grid_image,
            (int(origin[0] + offset_x), int(origin[1])),
            (int(van_point[0]), int(van_point[1])),
            color,
            thickness,
        )

    return grid_image

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
   
    #seeds = [[1, 508], [747, 245], [1401, 215], [1082, 701]]
    seeds = [[1, 508], [747, 245], [1401, 115], [1082, 701]]


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

# def compute_optical_flow(prev_frame, next_frame):
    # prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    # # Preprocess with bilateral filter for edge preservation
    # prev_gray = cv2.bilateralFilter(prev_gray, d=5, sigmaColor=50, sigmaSpace=50)
    # next_gray = cv2.bilateralFilter(next_gray, d=5, sigmaColor=50, sigmaSpace=50)

    # # Calculate optical flow with fine-tuned parameters
    # flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                        # pyr_scale=0.5,   # pyramid scale
                                        # levels=5,        # pyramid levels
                                        # winsize=21,      # window size
                                        # iterations=5,    # iterations per level
                                        # poly_n=7,        # neighborhood size
                                        # poly_sigma=1.5,  # standard deviation
                                        # flags=0)
    # return flow


def compute_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    # Preprocess with bilateral filter for edge preservation
    # prev_gray = cv2.bilateralFilter(prev_gray, d=5, sigmaColor=50, sigmaSpace=50)
    # next_gray = cv2.bilateralFilter(next_gray, d=5, sigmaColor=50, sigmaSpace=50)

    # Calculate optical flow with fine-tuned parameters
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                        pyr_scale=0.5,   # pyramid scale
                                        levels=5,        # pyramid levels
                                        winsize=21,      # window size
                                        iterations=5,    # iterations per level
                                        poly_n=7,        # neighborhood size
                                        poly_sigma=1.5,  # standard deviation
                                        flags=0)

    # Identify areas where velocity > 0 (valid flow regions)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    valid_mask = magnitude > 1e-2  # Threshold for motion presence

    # Invert the mask to find holes
    holes_mask = ~valid_mask

    # Fill holes using a blur-based approach
    flow_filled = fill_holes_with_blur(flow, holes_mask)

    return flow_filled


def fill_holes_with_blur(flow, holes_mask, blur_kernel=(551, 551)):
    """
    Fill holes in the flow field using a blur operation.

    Args:
        flow (ndarray): Optical flow field of shape (H, W, 2).
        holes_mask (ndarray): Binary mask of holes where flow needs to be filled.
        blur_kernel (tuple): Kernel size for the blur (e.g., (15, 15)).

    Returns:
        ndarray: Flow field with holes filled.
    """
    # Separate x and y components of the flow
    flow_x, flow_y = flow[..., 0], flow[..., 1]

    # Apply Gaussian blur to the entire flow components
    blurred_x = cv2.GaussianBlur(flow_x, blur_kernel, 0)
    blurred_y = cv2.GaussianBlur(flow_y, blur_kernel, 0)

    # Combine masks for each flow component
    filled_x = np.where(holes_mask, blurred_x, flow_x)
    filled_y = np.where(holes_mask, blurred_y, flow_y)

    # Stack x and y components back together
    flow_filled = np.stack([filled_x, filled_y], axis=-1)

    return flow_filled

def fill_holes_with_border_value(flow, holes_mask):
    """
    Fill holes in the flow field using the values from the border of the hole.
    
    Args:
        flow (ndarray): Optical flow field of shape (H, W, 2).
        holes_mask (ndarray): Binary mask of holes where flow needs to be filled.
    
    Returns:
        ndarray: Flow field with holes filled.
    """
    # Create a mask for valid flow regions (inverse of holes_mask)
    valid_mask = ~holes_mask

    # Separate x and y components of the flow
    flow_x, flow_y = flow[..., 0], flow[..., 1]

    # Fill holes for x and y components independently
    flow_x_filled = cv2.inpaint(flow_x.astype(np.float32), holes_mask.astype(np.uint8), inpaintRadius=30, flags=cv2.INPAINT_TELEA)
    flow_y_filled = cv2.inpaint(flow_y.astype(np.float32), holes_mask.astype(np.uint8), inpaintRadius=30, flags=cv2.INPAINT_TELEA)

    # Combine filled x and y components back into a single flow field
    flow_filled = np.stack([flow_x_filled, flow_y_filled], axis=-1)

    return flow_filled

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

def adjust_points_to_collinear(center_bottom, center_top, int_bev):

    # Calculate the direction vector from int_bev to the midpoint of center_bottom and center_top
    midpoint = (center_bottom + center_top) / 2
    direction = midpoint - int_bev

    # Normalize the direction vector
    direction_normalized = direction / np.linalg.norm(direction)

    # Calculate the distances of center_bottom and center_top from int_bev along the direction
    dist_bottom = np.dot(center_bottom - int_bev, direction_normalized)
    dist_top = np.dot(center_top - int_bev, direction_normalized)

    # Adjust the points to lie along the same line
    adjusted_bottom = int_bev + direction_normalized * dist_bottom
    adjusted_top = int_bev + direction_normalized * dist_top

    return adjusted_bottom, adjusted_top

def adjust_upper_face(lower_face, upper_face, hor_van, ver_van):
    # Ensure input faces are valid
    assert lower_face.shape == upper_face.shape, "Lower and upper faces must have the same number of points."
    assert lower_face.shape[1] == 2, "Face points should be in (x, y) format."

    # Initialize adjusted upper face
    adjusted_upper_face = upper_face.copy()

    # Adjust each point of the upper face
    for i in range(lower_face.shape[0]):
        lower_point = lower_face[i]
        upper_point = upper_face[i]

        # Compute the line from the lower face point to the horizontal vanishing point
        line_to_hor_van = np.cross(
            np.array([lower_point[0], lower_point[1], 1]),
            np.array([hor_van[0], hor_van[1], 1])
        )

        # Compute the line from the upper face point to the vertical vanishing point
        line_to_ver_van = np.cross(
            np.array([upper_point[0], upper_point[1], 1]),
            np.array([ver_van[0], ver_van[1], 1])
        )

        # Find the adjusted upper point as the intersection of the two lines
        adjusted_point_homogeneous = np.cross(line_to_hor_van, line_to_ver_van)
        if adjusted_point_homogeneous[2] != 0:
            adjusted_point = adjusted_point_homogeneous[:2] / adjusted_point_homogeneous[2]  # Convert to Cartesian
        else:
            adjusted_point = upper_point  # Fallback in case of no intersection (shouldn't happen)

        # Update the adjusted upper face point
        adjusted_upper_face[i] = adjusted_point

    return adjusted_upper_face
    
def draw_3d_box(image, box_3d, color_front=cv_colors.RED.value, color_back=cv_colors.BLUE.value, color_edges=cv_colors.BLACK.value, thickness=2):
    """
    Draw the 3D bounding box on an image.

    Args:
        image: Input image to draw on.
        box_3d: Array of 8 points (8x2) representing the 3D box in image coordinates.
        color_front: Color for the front face of the box.
        color_back: Color for the back face of the box.
        color_edges: Color for the edges connecting the front and back faces.
        thickness: Thickness of the lines.

    Returns:
        The image with the 3D box drawn.
    """
    # Ensure the box has 8 points
    assert box_3d.shape == (8, 2), "Box must have 8 corner points."

    # Draw the front face (first 4 points)
    front_face = box_3d[:4].astype(int)
    for i in range(4):
        cv2.line(image, tuple(front_face[i]), tuple(front_face[(i + 1) % 4]), color_front, thickness)

    # Draw the back face (last 4 points)
    back_face = box_3d[4:].astype(int)
    for i in range(4):
        cv2.line(image, tuple(back_face[i]), tuple(back_face[(i + 1) % 4]), color_back, thickness)

    # Connect corresponding points between the front and back faces
    for i in range(4):
        cv2.line(image, tuple(front_face[i]), tuple(back_face[i]), color_edges, thickness)

    return image
    
def compute_3d_box(box_corners, vert_vp, hor_left_vp, hor_right_vp):
    """
    Compute the 8 corner points of a minimal 3D bounding box from a 2D bounding box.

    Args:
        box_corners: List of 4 points representing the 2D bounding box corners.
        vert_vp: Vertical vanishing point [x, y].
        hor_left_vp: Left horizontal vanishing point [x, y].
        hor_right_vp: Right horizontal vanishing point [x, y].

    Returns:
        np.array: Array of 8 points (8x2) representing the 3D box in image coordinates.
    """
    x_min, y_max = box_corners[0]
    x_max, y_min = box_corners[2]

    # Front face corners (directly from 2D bounding box)
    front_bottom_left = [x_min, y_max]
    front_top_left = [x_min, y_min]
    front_top_right = [x_max, y_min]
    front_bottom_right = [x_max, y_max]

    # Extend lines to compute the back face
    back_bottom_left = compute_intersection(
        [front_bottom_left, hor_left_vp],
        [front_bottom_left, vert_vp]
    )
    back_top_left = compute_intersection(
        [front_top_left, hor_left_vp],
        [front_top_left, vert_vp]
    )
    back_top_right = compute_intersection(
        [front_top_right, hor_right_vp],
        [front_top_right, vert_vp]
    )
    back_bottom_right = compute_intersection(
        [front_bottom_right, hor_right_vp],
        [front_bottom_right, vert_vp]
    )

    # Combine all points into an array
    box_3d = np.array([
        front_bottom_left,  # Front face
        front_top_left,
        front_top_right,
        front_bottom_right,
        back_bottom_left,   # Back face
        back_top_left,
        back_top_right,
        back_bottom_right
    ])

    return box_3d
    
def bev_to_plain_view(bev_mask, inv_mat, image_size):
    """
    Transform a BEV label mask to plain view using the inverse homography matrix.
    Args:
        bev_mask: Input label mask in BEV coordinates.
        inv_mat: Inverse homography matrix (3x3) for BEV to plain-view mapping.
        image_size: Tuple (height, width) for the output plain-view image.
    Returns:
        Transformed label mask in plain view.
    """
    h, w = image_size
    plain_view_mask = cv2.warpPerspective(bev_mask.astype(np.int32), inv_mat, (w, h), flags=cv2.INTER_NEAREST)
    return plain_view_mask

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
 

def compute_3d_box_from_plain_mask(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False):

    # Step 1: Extract non-zero points (boundary of the mask)
    points = np.argwhere(mask > 0)  # Extract all non-zero pixel coordinates (row, col)
    points = points[:, [1, 0]]  # Switch to (x, y) format for consistency

    if len(points) < 3:
        raise ValueError(f"Not enough points in mask for a valid polygon. Found {len(points)} points.")

    # Step 2: Compute Convex Hull for a clean boundary
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
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
    
    
    # corner_c2 = compute_intersection(line2, line6)
    # corner_b2 = compute_intersection(line1, line4)
    # corner_h2 = corner_h
    # corner_a2 = corner_a
    
    
    corner_c1 = compute_intersection(line2, line3)
    corner_a11 = get_intersect(hor_left_vp, corner_c1, vert_vp, corner_a)
    corner_b1 = compute_intersection(line5, line4)
    corner_h1 = compute_intersection(line2, line5)
    corner_a12 = get_intersect(hor_left_vp, corner_c1, hor_right_vp, corner_b1)
    corner_a1 = ((corner_a11[0] + corner_a12[0])/2, (corner_a11[1] + corner_a12[1])/2)
   
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


def compute_3d_box_from_plain_mask_adjust(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False):

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

    # epsilon = 1  # Small shift value
    # if corner_h is None:
       # # Slightly shift corner_c and corner_b if no intersection exists
        # #epsilon = 1e-3  # Small shift value
        # shifted_corner_c = (corner_c[0] + epsilon, corner_c[1] + epsilon)
        # shifted_corner_b = corner_b #(corner_b[0] - epsilon, corner_b[1] - epsilon)
        
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


    # if corner_h is None:
       # # Slightly shift corner_c and corner_b if no intersection exists
        # #epsilon = 1e-3  # Small shift value
        # shifted_corner_c = corner_c #(corner_c[0] + epsilon, corner_c[1] + epsilon)
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
    
    
    # corner_c2 = compute_intersection(line2, line6)
    # corner_b2 = compute_intersection(line1, line4)
    # corner_h2 = corner_h
    # corner_a2 = corner_a
    
    
    corner_c1 = compute_intersection(line2, line3)
    corner_a1 = get_intersect(hor_left_vp, corner_c1, vert_vp, corner_a)
    corner_b1 = compute_intersection(line5, line4)
    corner_h1 = compute_intersection(line2, line5)
    # corner_a12 = get_intersect(hor_left_vp, corner_c1, hor_right_vp, corner_b1)
    # corner_a1 = ((corner_a11[0] + corner_a12[0])/2, (corner_a11[1] + corner_a12[1])/2)
   
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
        # valid_corners = [corner for corner in corners if corner is not None]
        # valid_corners1 = [corner for corner in corners1 if corner is not None]
    
        # Zoom into the mask region
        rows, cols = np.where(mask)
        if rows.size > 0 and cols.size > 0:
            plt.xlim([cols.min() - 10, cols.max() + 10])
            plt.ylim([rows.max() + 10, rows.min() - 10])  # Invert y-axis for correct orientation
    
        # Show legend and final plot
        plt.legend()
        plt.show()
        
    adjust = True
    if adjust:
        #print(f"corner_h1: {corner_h1}, hor_right_vp: {hor_right_vp}, vert_vp: {vert_vp}, corner_h: {corner_h}")
        corner_h1 = get_intersect(corner_h1, hor_left_vp, vert_vp, corner_h)
        corner_c1 = get_intersect(corner_h1, hor_left_vp, vert_vp, corner_c)
        corner_b1 = get_intersect(corner_h1, hor_right_vp, vert_vp, corner_b)
        #corner_a1 = get_intersect(corner_c1, hor_right_vp, vert_vp, corner_a)
        corner_a1 = get_intersect(corner_c1, hor_right_vp, hor_left_vp, corner_b1)


        
        
    # print([corner_a, corner_b, corner_h, corner_c])
    # print([corner_a1, corner_b1, corner_h1, corner_c1])

    return np.array([corner_a, corner_b, corner_h, corner_c]), np.array([corner_a1, corner_b1, corner_h1, corner_c1])


def compute_3d_box_from_plain_mask_2(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False):
    """
    Compute a 3D box aligned to the given vanishing points from a 2D mask.
    """
    # Step 1: Extract boundary points of the mask
    points = np.argwhere(mask > 0)  # (row, col)
    points = points[:, [1, 0]]  # Switch to (x, y)

    if len(points) < 3:
        raise ValueError(f"Not enough points in mask for a valid polygon. Found {len(points)} points.")

    # Step 2: Compute Convex Hull for a clean boundary
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # Step 3: Calculate tangent points to vanishing points
    mask_center = np.mean(hull_points, axis=0)  # Centroid of the convex hull

    point_b, line2, point_a, line1 = find_tangent_points(mask_center, hull_points, hor_right_vp)
    point_d, line4, point_c, line3 = find_tangent_points(mask_center, hull_points, vert_vp)
    point_f, line6, point_e, line5 = find_tangent_points(mask_center, hull_points, hor_left_vp)

    # Step 4: Compute intersections for box corners
    def compute_intersection(line1, line2):
        """Compute the intersection of two lines in homogeneous coordinates."""
        inter = np.cross(line1, line2)
        if inter[2] != 0:
            return inter[:2] / inter[2]
        return None

    # Lower face corners
    corner_a = compute_intersection(line1, line6)
    corner_b = compute_intersection(line1, line4)
    corner_c = compute_intersection(line3, line6)
    corner_d = compute_intersection(line3, line4)

    # Upper face corners aligned with vertical vanishing point
    if corner_a is not None:
        corner_a1 = get_intersect(hor_left_vp, corner_a, vert_vp, corner_b)
    if corner_b is not None:
        corner_b1 = get_intersect(hor_right_vp, corner_b, vert_vp, corner_c)
    if corner_c is not None:
        corner_c1 = get_intersect(hor_left_vp, corner_c, vert_vp, corner_d)
    if corner_d is not None:
        corner_d1 = get_intersect(hor_right_vp, corner_d, vert_vp, corner_a)

    # Compile lower and upper corners
    lower_corners = [corner_a, corner_b, corner_d, corner_c, corner_a]
    upper_corners = [corner_a1, corner_b1, corner_d1, corner_c1, corner_a1]

    # Debugging and visualization
    if debug:
        plt.imshow(mask, cmap='gray')  # Show the mask in the background

        def plot_point(corner, color, label, marker='o'):
            if corner is not None and len(corner) == 2:
                plt.scatter(*corner, color=color, marker=marker, label=label)

        def draw_line_segment(vp, point, color):
            if vp is not None and point is not None:
                plt.plot([vp[0], point[0]], [vp[1], point[1]], color=color, linestyle="--")

        # Plot corners
        for corner, color, label in zip(lower_corners, ['red', 'blue', 'green', 'yellow'], ["A", "B", "C", "D"]):
            plot_point(corner, color, f"Lower {label}")
        for corner, color, label in zip(upper_corners, ['red', 'blue', 'green', 'yellow'], ["A1", "B1", "C1", "D1"]):
            plot_point(corner, color, f"Upper {label}")

        # Plot tangent lines
        draw_line_segment(hor_right_vp, point_a, 'red')
        draw_line_segment(hor_right_vp, point_b, 'red')
        draw_line_segment(vert_vp, point_c, 'blue')
        draw_line_segment(vert_vp, point_d, 'blue')
        draw_line_segment(hor_left_vp, point_e, 'green')
        draw_line_segment(hor_left_vp, point_f, 'green')

        # Filter valid corners for visualization
        valid_lower = [corner for corner in lower_corners if corner is not None]
        valid_upper = [corner for corner in upper_corners if corner is not None]

        # Zoom into the mask region
        rows, cols = np.where(mask)
        if rows.size > 0 and cols.size > 0:
            plt.xlim([cols.min() - 10, cols.max() + 10])
            plt.ylim([rows.max() + 10, rows.min() - 10])  # Invert y-axis for correct orientation

        # Show legend and final plot
        plt.legend()
        plt.show()

    # Return the corners
    return np.array(lower_corners), np.array(upper_corners)

def is_segment_occluded(camera_position, segment, other_segments):
    """
    Check if any line from extreme points of `segment` to `camera_position` 
    intersects with any `other_segments`. Returns the label of the occluding segment.
    """
    # Get extreme points (min/max X and Y) of the current segment
    segment_points = np.argwhere(segment["plain_mask"])
    min_x, min_y = np.min(segment_points, axis=0)
    max_x, max_y = np.max(segment_points, axis=0)

    # Define the extreme points
    extreme_points = [
        np.array([min_x, min_y]),  # Bottom-left
        np.array([max_x, min_y]),  # Bottom-right
        np.array([min_x, max_y]),  # Top-left
        np.array([max_x, max_y]),  # Top-right
    ]

    for i, other_segment in enumerate(other_segments):
        # Skip if the other segment has no mask
        if other_segment["plain_mask"].sum() == 0:
            continue

        # Get the bounding box of the other segment
        other_points = np.argwhere(other_segment["plain_mask"])
        other_min_x, other_min_y = np.min(other_points, axis=0)
        other_max_x, other_max_y = np.max(other_points, axis=0)

        # Iterate through all extreme points of the current segment
        for extreme_point in extreme_points:
            # Check if the ray from the extreme point to the camera intersects the bounding box
            if not bounding_box_intersects_ray(
                extreme_point, camera_position, (other_min_x, other_min_y, other_max_x, other_max_y)
            ):
                continue  # No intersection with the bounding box

            # Refine check: ray intersects the actual segment mask
            if segment_ray_intersects(extreme_point, camera_position, other_points):
                return i  # Return the label of the occluding segment

    return None  # No occlusion


def bounding_box_intersects_ray(point, camera_position, bbox):
    """
    Quick bounding box check for ray intersection.
    """
    x1, y1, x2, y2 = bbox
    ray_x_min = min(point[0], camera_position[0])
    ray_x_max = max(point[0], camera_position[0])
    ray_y_min = min(point[1], camera_position[1])
    ray_y_max = max(point[1], camera_position[1])

    # Check if bounding box overlaps with the ray's bounding box
    return not (x2 < ray_x_min or x1 > ray_x_max or y2 < ray_y_min or y1 > ray_y_max)


def segment_ray_intersects(point, camera_position, other_points):
    """
    Check if the ray from `point` to `camera_position` intersects the `other_points`.
    """
    ray_direction = camera_position - point
    ray_direction = ray_direction / np.linalg.norm(ray_direction)  # Normalize

    for other_point in other_points:
        to_other = other_point - point
        cross_product = np.cross(ray_direction, to_other)

        # Check if the other point is aligned with the ray
        if np.abs(cross_product) < 1e-3:  # Tolerance for alignment
            # Check if the other point lies between the camera and the segment point
            dot_product = np.dot(ray_direction, to_other)
            if dot_product > 0:  # Point is in front of the segment point
                return True  # Intersection found

    return False

def main(video_path, draw_boundaries=True, debug=False):

    ipm_matrix, inv_mat, target_shape = generate_perspective_matrix(ratio = 0.75)
    
    line1_bev = map_points_to_BEV(line1, ipm_matrix)
    line2_bev = map_points_to_BEV(line2, ipm_matrix)
    int_bev = get_intersect(line1_bev[0], line1_bev[1], line2_bev[0], line2_bev[1])
    print(int_bev)
    int_bev_iv = iv(int_bev)
    
    hor_van = get_intersect(line3[0], line3[1], line4[0], line4[1])
    ver_van = get_intersect(line1[0], line1[1], line2[0], line2[1])

    cap = cv2.VideoCapture(input_video_path)
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) // 2
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_position = (width/2, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Can't read video.")
        cap.release()
        return

    avg_frame = compute_average_frame(video_path, 'avg_frame.png')

    frame_count = 0

    while True:
        ret, next_frame = cap.read()
        frame_count += 1

        if not ret:
            break

        print(f"Processing {frame_count} frame")
        if frame_count < 75:# - 20:
            prev_frame = next_frame
            #prev_bev = next_bev
            continue
        if frame_count >  80:# + 20:
            prev_frame = next_frame
            #prev_bev = next_bev
            break
            
        height, width = next_frame.shape[:2]
        resized_frame = cv2.resize(next_frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA)

        # Save the resized frame
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, resized_frame)  # Save resized frame
        print(f"Saved resized frame {frame_count} to {frame_filename}")
                                      

        # Compute the difference and create the moving mask
        diff = cv2.absdiff(next_frame, avg_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        ratio = 0.1
        blurred_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
        _, moving_mask = cv2.threshold(blurred_diff, 15, 255, cv2.THRESH_BINARY)
    
        #moving_mask = cv2.dilate(moving_mask, None, iterations=1)
        moving_mask_bin = moving_mask // 255
        # Find connected components
        num_labels, labels = cv2.connectedComponents(moving_mask)
        
        # Initialize variables to track the largest component
        largest_area = 0
        largest_label = 0
        
        # Iterate through all components (excluding background, label 0)
        for label in range(1, num_labels):
            # Create a mask for the current component
            component_mask = (labels == label).astype(np.uint8)
            
            # Calculate the area of the component
            area = cv2.countNonZero(component_mask)
            
            # Update the largest component if this one is bigger
            if area > largest_area:
                largest_area = area
                largest_label = label
        
            # Create a binary mask for the largest component
            largest_component_mask = (labels == largest_label).astype(np.uint8)
            
            # Find the coordinates of all pixels in the largest component
            y_coords, x_coords = np.where(largest_component_mask > 0)
            
            # Check if the component has pixels
            if len(x_coords) > 0 and len(y_coords) > 0:
                # Combine x and y coordinates into a single array of points
                points = np.column_stack((x_coords, y_coords))
            
                # Compute the convex hull of the points
                #hull = ConvexHull(points)
            
                # Create a blank mask and draw the convex hull
                new_mask = np.zeros_like(moving_mask, dtype=np.uint8)
                hull_points = points#[hull.vertices]  # Get the points forming the convex hull
                cv2.fillPoly(new_mask, [hull_points], 255)
            
                # Optional: Dilate the mask for smoother edges
                new_mask = cv2.dilate(new_mask, None, iterations=1)
            else:
                # If no points are found, use the original mask as fallback
                new_mask = moving_mask

        moving_mask = new_mask
        # Display the moving mask
        #cv2.imshow('Moving Mask', resize_to_match_height(moving_mask, screen_height))

        region_size = 2*5*10  #200 #100 #15  #10#30
        ruler = 2*5*10  #150 # 100 #20 #14

        slic = cv2.ximgproc.createSuperpixelSLIC(next_frame,
                                                  algorithm=cv2.ximgproc.MSLIC,
                                                  region_size=region_size,
                                                  ruler=ruler)
        slic.iterate(slic_iterations)

        labels = slic.getLabels() + 1
        num_labels = np.max(labels) + 1
        
        labels[moving_mask == 0] = 0
        

        
        # If debug is enabled
        frame_output = next_frame.copy()
        contour_mask = slic.getLabelContourMask(False)
        frame_output[0 < contour_mask] = cv_colors.RED.value
        
        # Display the moving mask
        #cv2.imshow('Segmentation', resize_to_match_height(frame_output, screen_height))

        bounding_box_image = next_frame.copy()
        img_with_box  = next_frame.copy()

        neighbors_dict = find_neighbors_within_mask(labels)
        segments = []

        lower_faces = []  # Initialize as an empty list
        heights = []  # Similarly, initialize upper_faces if needed
        colors = []

        for label in range(1, num_labels):
            mask = (labels == label)

            if True: #moving_percentage >= 0.3:
                # Find the bounding box for the current segment
                ys, xs = np.where(mask)
                if xs.size > 0 and ys.size > 0:
                    x_min, x_max = np.min(xs), np.max(xs)
                    y_min, y_max = np.min(ys), np.max(ys)
                    top_point = (xs[np.argmin(ys)], y_min)  # Top-most point
                    bottom_point = (xs[np.argmax(ys)], y_max)  # Bottom-most point
                    left_point = (x_min, ys[np.argmin(xs)])  # Left-most point
                    right_point = (x_max, ys[np.argmax(xs)])  # Right-most point

                    plain_mask = mask #(plain_labels == label)
                    points = np.argwhere(plain_mask > 0)  # Extract all non-zero pixel coordinates (row, col)
                    points = points[:, [1, 0]]  # Switch to (x, y) format for consistency
                
                    if len(points) < 3:
                        continue
                        #raise ValueError(f"Not enough points in mask for a valid polygon. Found {len(points)} points.")
                    mask_center = np.mean(points, axis=0)

                    # Store segment data (bounding box, mask, etc.)
                    segments.append({
                        'label': label,
                        'top_point': top_point,
                        'bottom_point': bottom_point,
                        'left_point': left_point,
                        'right_point': right_point,
                        'plain_mask' : mask,
                        'mask_center': mask_center,
                        'lower_face' : None,
                        'z': 0,
                        'mat': ipm_matrix,
                        'level': 0,
                        
                        #'mask': mask,
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
        segments = sorted(segments, key=lambda s: np.linalg.norm(np.array(s['mask_center']) - camera_position))

        # Initialize ray-based colors
        segment_colors = {}  # Store colors for each segment by label
        camera_position = np.array(camera_position)
        
        # Process each segment and check rays for all points
        for i, segment in enumerate(segments):
            # Draw the segment with the assigned color
            lower_face, upper_face = None, None
            segment_mask = segment['plain_mask']
            try:
                # Compute 3D box
                lower_face, upper_face = compute_3d_box_from_plain_mask_new(
                    segment_mask, vert_vp, hor_left_vp, hor_right_vp, debug=False
                )
                #cv2_color = segment_colors[segment_label][0]
                # draw_cube(
                    # img_with_box,
                    # lower_face.astype("int"),
                    # upper_face.astype("int"),
                    # color=cv2_color,
                    # thickness=2,
                # )
                # print(f"OK segment ")
            except Exception as e:
                print(f"Error processing segment ")
                continue

            #continue

            if lower_face is not None and upper_face is not None:

                bottom_face = map_points_to_BEV(lower_face, ipm_matrix)
                top_face = map_points_to_BEV(upper_face, ipm_matrix)
                bev_points_iv = to_iv(bottom_face)
                top_iv = to_iv(top_face)
                print(bottom_face)
                # A-B = 0-1

                avg_upper_face = upper_face
                avg_lower_face = lower_face

                bev_length = np.linalg.norm(bottom_face[0] -
                                            bottom_face[-1])
                # TODO adjust according to view angle
                proj_length = np.linalg.norm(lower_face[0] -
                                            upper_face[-1])
                plain_height = np.linalg.norm(
                    avg_lower_face[0] -
                    avg_upper_face[1])
                scaling_factor = bev_length / proj_length

                height = plain_height * scaling_factor


                segment['height'] = height
                segment['lower_face'] = avg_lower_face
                segment['upper_face'] = avg_upper_face
                segment['bottom'] = bev_points_iv
                segment['top'] = top_iv

        #cv2.imshow('img_with_box ', img_with_box)


        for i, segment in enumerate(segments):
            


            z = 0
            level = 0
            height = segment['height']

            segment_label =  segment['label']
            avg_lower_face = segment['lower_face']
            avg_upper_face = segment['upper_face']
            avg_bootom_iv = segment['bottom']
            avg_top_iv = segment['top']
            superposed_mat = segment['mat']
            cv2_color, plt_color = generate_random_color()
            cv2_color = cv_colors.BLACK.value
            
            print(avg_bootom_iv)

            if avg_bootom_iv is not None:# None: continue
                #cv2_color = labels_to_color(segment_label)
                #cv2_color, plt_color = generate_random_color()
                # Loop through earlier segments (which are below in the image)
                for j, other_segment in enumerate(segments):#[:i]):
                    other_label = other_segment['label']
                    if other_label == segment_label:
                        continue

   
                    other_upper_face = other_segment['upper_face']
                    if avg_lower_face is not None and other_upper_face is not None:
                        if faces_overlap(avg_lower_face, other_upper_face):
                            cv2_color = other_segment['cv2_color']
                            plt_color = other_segment['plt_color']

                            z = other_segment['z'] + other_segment['height']
                            level = other_segment['level'] + 1
                            pts1 = np.float32(other_segment['upper_face'])
                            pts2 = np.float32(other_segment['lower_face'])
                            #pts2 = np.float32(to_iv(other_segment['bottom']))

                            # Generate the perspective transformation matrices
                            loc_persp = cv2.getPerspectiveTransform(pts1, pts2)
                            superposed_mat = np.dot(loc_persp, other_segment['mat'])

                            avg_bootom_iv = to_iv(map_points_to_BEV(avg_lower_face, superposed_mat))
                            #top_iv = to_iv(map_points_to_BEV(avg_upper_face, superposed_mat))

                            break

                    other_lower_face = other_segment['lower_face']
                    if avg_lower_face is not None and other_lower_face is not None:
                        if faces_overlap(avg_lower_face, other_lower_face):
                            cv2_color = other_segment['cv2_color']
                            plt_color = other_segment['plt_color']

                            z = other_segment['z']
                            level = other_segment['level']
                            superposed_mat = other_segment['mat']

                            avg_bootom_iv = to_iv(map_points_to_BEV(avg_lower_face, superposed_mat))
                            #top_iv = to_iv(map_points_to_BEV(avg_upper_face, superposed_mat))

                            break

                    # Check if other_segment is a neighbor of segment
                    # if other_label in neighbors_dict[segment_label]:
                        # relative_positions = neighbors_dict[segment_label][
                            # other_label]

                        # if ('left' in relative_positions
                                # or 'right' in relative_positions):
                                # # or 'bottom-left' in relative_positions
                                # # or 'bottom-right' in relative_positions):

                            # cv2_color = other_segment['cv2_color']
                            # plt_color = other_segment['plt_color']

                            # z = other_segment['z']
                            # level = other_segment['level']
                            # superposed_mat = other_segment['mat']

                            # avg_bootom_iv = to_iv(map_points_to_BEV(avg_lower_face, superposed_mat))
                            # break
    
                # segment['top'] = avg_top_iv
                
                cv2_color = list(cv_colors)[level % len(cv_colors)].value

                segment['bottom'] = avg_bootom_iv
                segment['cv2_color'] = cv2_color
                segment['plt_color'] = plt_color

                segment['level'] = level
                segment['height'] = height
                segment['z'] = z
                segment['mat'] = superposed_mat
                #Extend avg_upper_face by adding height to each point
                avg_bottom_3d = []
                for point in avg_bootom_iv:
                    x_2d, y_2d = point
                    avg_bottom_3d.append(
                        [x_2d, y_2d, z])  # Add height as the third coordinate
    
                avg_bottom_3d = np.array(
                    avg_bottom_3d)  # Convert to a NumPy array if needed
                #segment['bottom'] = avg_bottom
                
                plt_color = [c / 255.0 for c in cv2_color[::-1]]  # Normalize for plt
    
                #Collect lower and upper faces for 3D rendering
                if len(avg_bottom_3d) == 4:
                    lower_faces.append(
                        avg_bottom_3d
                    )  # Lower face remains 2D (or use zeros for Z if required)
                    heights.append(height)  # Now a 3D face
                    colors.append(plt_color)


            if avg_lower_face is not None and avg_upper_face is not None:
                draw_cube(
                bounding_box_image,
                avg_lower_face.astype("int"),
                avg_upper_face.astype("int"),
                color=cv2_color,  #cv_colors.ORANGE.value,
                thickness=2)



        resized_image = bounding_box_image #resize_to_match_height(bounding_box_image, screen_height)
        out.write(resized_image)
        prev_frame = next_frame

        cv2.imshow('Bounding Boxes', resized_image)
        #cv2.imshow('3D Boxes',img_with_box)

        if cv2.waitKey(30) & 0xFF == 27:  # Esc key to stop
            cv2.imwrite('BoundingBoxes.png', bounding_box_image)
            break

        # Call your function to draw cubes
        if debug and len(lower_faces) > 1:# and frame_count == 80:
            draw_cubes_in_3d(lower_faces, heights, colors)

    #cv2.imwrite('BoundingBoxes.png', segm_out)
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
