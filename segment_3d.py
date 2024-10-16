import cv2
import numpy as np
import os

from lifting import *
from enum import Enum

# Initialize video capture and output
input_video_path = '/home/dmytrozhuravlov/cv/data/bike.mp4'
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
use_slick = False
show_warp = False  #True
resize_shape = (1600, 900)
screen_width, screen_height = 1920 // 2, 1080 // 2
roi_corners = None
min_square = 10000


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


def generate_perspective_matrix(img, roi_corners):
    src = np.float32(roi_corners)
    top_padding, left_padding = 1000000, 0
    warped_size = (img.shape[1] + 2 * left_padding, img.shape[0] + top_padding)
    offset = int((warped_size[0] - 2 * left_padding) / 3.0)

    dst = np.float32([[offset + left_padding, warped_size[1]],
                      [offset + left_padding, top_padding],
                      [warped_size[0] - offset - left_padding, top_padding],
                      [warped_size[0] - offset - left_padding,
                       warped_size[1]]])

    persp = cv2.getPerspectiveTransform(src, dst)
    inv = cv2.getPerspectiveTransform(dst, src)
    return persp, inv, warped_size


def crop_warp(warped):
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


def compute_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15,
                                        3, 5, 1.2, 0)
    return flow


def mark_boundaries(image, labels, draw_boundaries=True):
    if not draw_boundaries:
        return image

    boundaries = np.copy(image)
    for y in range(1, labels.shape[0] - 1):
        for x in range(1, labels.shape[1] - 1):
            if (labels[y, x] != labels[y - 1, x]
                    or labels[y, x] != labels[y + 1, x]
                    or labels[y, x] != labels[y, x - 1]
                    or labels[y, x] != labels[y, x + 1]):
                boundaries[y, x] = (0, 0, 0)
    return boundaries


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


def map_box_to_BEV(box_corners, ipm_matrix):
    """Project bounding box corners onto the BEV using IPM."""
    box_corners = np.array(box_corners, dtype=np.float32)
    box_corners = np.array([box_corners
                            ])  # Reshape to (1, N, 2) for perspectiveTransform
    bev_points = cv2.perspectiveTransform(box_corners, ipm_matrix)
    return bev_points[0]  # Return as (N, 2)


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


def map_box_to_BEV_with_padding(box_corners, ipm_matrix, padding_size):
    """
    Maps bounding box corners to BEV coordinates and accounts for padding in the BEV image.
    """
    bev_points = map_box_to_BEV(box_corners, ipm_matrix)
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


def draw_cube(image,
              lower_face,
              upper_face,
              color=cv_colors.RED.value,
              thickness=1):
    # Draw the lower face (as a quadrilateral)
    cv2.polylines(image, [lower_face],
                  isClosed=True,
                  color=color,
                  thickness=thickness)
    # Draw the upper face (as a quadrilateral)
    cv2.polylines(image, [upper_face], isClosed=True, color=color, thickness=thickness)

    # Draw the vertical line between the lower and upper vertex
    cv2.line(image, lower_face[0], upper_face[2], color=color, thickness=thickness)
    cv2.line(image, lower_face[1], upper_face[1], color=color, thickness=thickness)
    cv2.line(image, lower_face[2], upper_face[0], color=color, thickness=thickness)
    cv2.line(image, lower_face[3], upper_face[3], color=color, thickness=thickness)


# def draw_cube(image, bottom, top, camera, color=(0, 0, 255), thickness=1):
# # Convert bottom and top faces from BEV to the plain view using camera transformations
# lower_face = camera.bev_2_plain(bottom).astype("int")  # Bottom face
# upper_face = camera.bev_2_plain(top).astype("int")     # Top face

# # Ensure that the faces have 4 vertices (since we're working with cubes or rectangular boxes)
# assert lower_face.shape[0] == 4 and upper_face.shape[0] == 4, "Faces must have 4 vertices"

# # Draw the lower face (as a quadrilateral)
# cv2.polylines(image, [lower_face], isClosed=True, color=color, thickness=thickness)

# # Draw the upper face (as a quadrilateral)
# cv2.polylines(image, [upper_face], isClosed=True, color=color, thickness=thickness)

# # Draw vertical lines connecting corresponding vertices of the lower and upper faces
# for i in range(4):
# pt_lower = tuple(lower_face[i])  # Vertex from the lower face
# pt_upper = tuple(upper_face[i])  # Corresponding vertex from the upper face

# # Draw the vertical line between the lower and upper vertex
# cv2.line(image, pt_lower, pt_upper, color=color, thickness=thickness)

# return image


def draw_bounding_box_for_blob(image, flow, component, flow_threshold=1.0):
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


def main(video_path, draw_boundaries=True, debug=False):
    cap = cv2.VideoCapture(input_video_path)
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Can't read video.")
        cap.release()
        return

    avg_frame = compute_average_frame(video_path, 'avg_frame.png')
    roi_corners = [[68, 1024], [836, 431], [1025, 431],
                   [prev_frame.shape[1], prev_frame.shape[0]]]

    # Compute min/max x, y values from roi_corners
    x_min_roi = min(roi[0] for roi in roi_corners)
    x_max_roi = max(roi[0] for roi in roi_corners)
    y_min_roi = min(roi[1] for roi in roi_corners)
    y_max_roi = max(roi[1] for roi in roi_corners)

    ipm_matrix, inv_mat, target_shape = generate_perspective_matrix(
        avg_frame, roi_corners)

    region_size = 300  #10#30
    ruler = 30

    #bev_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    #prev_boxes = []
    frame_count = 0

    while True:
        ret, next_frame = cap.read()
        frame_count += 1
        if not ret:
            break

        #if frame_count < 1100: continue

        # Perform background subtraction (difference from average frame)
        diff = cv2.absdiff(next_frame, avg_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to remove noise
        blurred_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)

        # Threshold the blurred difference image to create a binary mask for moving pixels
        _, moving_mask = cv2.threshold(blurred_diff, 60, 255,
                                       cv2.THRESH_BINARY)

        cv2.imshow('Moving Mask',
                   resize_to_match_height(moving_mask, screen_height))

        bev_image = cv2.warpPerspective(next_frame, ipm_matrix, target_shape)
        flow = compute_optical_flow(prev_frame, next_frame)

        # Mask out flow where there is no motion
        flow_masked = np.zeros_like(flow)
        flow_masked[moving_mask > 0] = flow[moving_mask > 0]

        num_labels, labels_im = cv2.connectedComponents(
            moving_mask.astype(np.uint8))
        #flow = flow_masked

        if not use_slick:
            for label in range(1, num_labels):  # Skip the background (label 0)
                component_mask = (labels_im == label)

                ys, xs = np.where(component_mask)
                x_min, x_max = np.min(xs), np.max(xs)
                y_min, y_max = np.min(ys), np.max(ys)
                square = (y_max - y_min) * (x_max - x_min)
                if square < min_square: continue

                bounding_box_image = draw_bounding_box_for_blob(
                    next_frame, flow, component_mask
                )  # Ensure draw_bounding_box_for_blob is implemented

                # Define the extreme points of the blob (top-most, bottom-most, left-most, right-most)
                top_point = (xs[np.argmin(ys)], y_min)  # Top-most point
                bottom_point = (xs[np.argmax(ys)], y_max)  # Bottom-most point
                left_point = (x_min, ys[np.argmin(xs)])  # Left-most point
                right_point = (x_max, ys[np.argmax(xs)])  # Right-most point

                extreme_points = [
                    top_point, bottom_point, left_point, right_point
                ]

                # Check if the bounding box is within the ROI
                if True:  #x_min >= x_min_roi and x_max <= x_max_roi and y_min >= y_min_roi and y_max <= y_max_roi:
                    #TODO crop to roi
                    box_corners = [[x_min, y_max], [x_min, y_min_roi],
                                   [x_max, y_min_roi], [x_max, y_max]]

                    # Extract the flow vectors inside the box
                    flow_in_box = flow[component_mask
                                       > 0]  # Flow vectors within the box
                    avg_flow_vector = np.mean(flow_in_box, axis=0)

                    if np.linalg.norm(avg_flow_vector) > 1.0:
                        scale = 1  #0
                        start_point_img = np.array([x_max, y_max],
                                                   dtype=np.float32)
                        end_point_img = np.array([
                            x_max + scale * avg_flow_vector[0],
                            y_max + scale * avg_flow_vector[1]
                        ],
                                                 dtype=np.float32)
                        cv2.arrowedLine(bounding_box_image,
                                        start_point_img.astype(int),
                                        end_point_img.astype(int),
                                        cv_colors.RED.value, 2)

                        start_point_bev = map_box_to_BEV(
                            [start_point_img], ipm_matrix)[
                                0]  # Ensure map_box_to_BEV is implemented
                        end_point_bev = map_box_to_BEV([end_point_img],
                                                       ipm_matrix)[0]

                        avg_flow_vector_bev = end_point_bev - start_point_bev
                        avg_flow_angle_bev = np.arctan2(
                            avg_flow_vector_bev[1], avg_flow_vector_bev[0])

                        avg_flow_angle = np.arctan2(avg_flow_vector[1],
                                                    avg_flow_vector[0])
                        print(
                            f"Rectangle: Average flow angle (degrees): {np.degrees(avg_flow_angle)}"
                        )
                        print(
                            f"Quadrangle: Average flow angle (degrees): {np.degrees(avg_flow_angle_bev)}"
                        )

                        # Compute previous positions of extreme points using the flow
                        # TODO compute directly
                        prev_extreme_points = []
                        for (x, y) in extreme_points:
                            flow_at_point = flow[int(y), int(
                                x)]  # Get the flow vector at the extreme point
                            prev_x = x - flow_at_point[
                                0]  # Backtrack using the flow
                            prev_y = y - flow_at_point[1]
                            prev_extreme_points.append((prev_x, prev_y))

                        prev_x_min, prev_x_max = np.min([
                            p[0] for p in prev_extreme_points
                        ]), np.max([p[0] for p in prev_extreme_points])
                        prev_y_min, prev_y_max = np.min([
                            p[1] for p in prev_extreme_points
                        ]), np.max([p[1] for p in prev_extreme_points])

                        prev_box_corners = [(prev_x_min, prev_y_max),
                                            (prev_x_min, prev_y_min),
                                            (prev_x_max, prev_y_min),
                                            (prev_x_max, prev_y_max)]
                        cv2.rectangle(bounding_box_image,
                                      (int(prev_x_min), int(prev_y_min)),
                                      (int(prev_x_max), int(prev_y_max)),
                                      cv_colors.PURPLE.value, 2)

                        # Compute angles
                        bev_points = map_box_to_BEV(box_corners, ipm_matrix)
                        bev_points1 = map_box_to_BEV(prev_box_corners,
                                                     ipm_matrix)

                        bev_points_iv = to_iv(bev_points)
                        bev_points1_iv = to_iv(bev_points1)

                        d, alpha, beta = compute_angles(
                            bev_points_iv,
                            debug)  # Ensure compute_angles is implemented
                        d1, alpha1, beta1 = compute_angles(
                            bev_points1_iv, debug
                        )  # Ensure compute_upper_angles is implemented

                        angle_radians = move_to_second_quadrant(
                            -avg_flow_angle_bev
                        )  # Ensure move_to_second_quadrant is implemented
                        x, y = calculate_solution(np.pi - angle_radians, alpha,
                                                  beta, d,
                                                  np.pi - angle_radians,
                                                  alpha1, beta1, d1, debug)

                        if x is not None and y is not None:
                            bottom_iv = compute_bottom_rectangle(
                                bev_points_iv, np.pi + angle_radians, x, y,
                                debug
                            )  # Ensure compute_bottom_rectangle is implemented
                            bottom = to_iv(bottom_iv)
                            center = bottom[0]
                            bottom = bottom[1:]
                            lower_face = map_box_to_BEV(bottom, inv_mat)
                            #lower_face_iv = to_iv(lower_face)
                            draw_bottom(bounding_box_image,
                                        lower_face.astype("int"),
                                        color=cv_colors.MINT.value,
                                        thickness=3)
                            #continue
                            d, alpha, beta = compute_upper_angles(
                                bev_points_iv,
                                debug)  # Ensure compute_angles is implemented
                            d1, alpha1, beta1 = compute_upper_angles(
                                bev_points1_iv, debug
                            )  # Ensure compute_upper_angles is implemented
                            # TODO use upper angles
                            x_up, y_up = calculate_solution(
                                -np.pi / 2 - angle_radians, alpha1, beta1, d1,
                                -np.pi / 2 - angle_radians, alpha1, beta1, d1,
                                debug)
                            if x_up is not None and y_up is not None:

                                top_iv = compute_top_rectangle(
                                    bev_points_iv, -np.pi / 2 - angle_radians,
                                    x_up, y_up, debug
                                )  # Ensure compute_top_rectangle is implemented
                                if bottom is not None and top_iv is not None:
                                    top = to_iv(top_iv)
                                    center_top = top[0]
                                    #bottom = camera.convert_bev_to_world(bottom)
                                    #det.centers = [[bottom[0], det.angle_radians]]

                                    top = top[1:]

                                    upper_face = map_box_to_BEV(top, inv_mat)

                                    draw_cube(bounding_box_image,
                                              lower_face.astype("int"),
                                              upper_face.astype("int"),
                                              color=cv_colors.ORANGE.value,
                                              thickness=3)

            resized_image = resize_to_match_height(
                bounding_box_image,
                screen_height)  # Ensure resize_to_match_height is implemented
            if show_warp:
                bev_image_resized = resize_to_match_height(
                    crop_warp(bev_image),
                    screen_height)  # Ensure crop_warp is implemented
                combined_image = np.hstack((resized_image, bev_image_resized))
                resized_image = combined_image

            prev_frame = next_frame
            cv2.imshow('Bounding Boxes', resized_image)

            if cv2.waitKey(30) & 0xFF == 27:  # Esc key to stop
                break
            continue

        slic = cv2.ximgproc.createSuperpixelSLIC(next_frame,
                                                 algorithm=cv2.ximgproc.MSLIC,
                                                 region_size=region_size,
                                                 ruler=ruler)
        slic.iterate(10)
        labels = slic.getLabels() + 1

        colored_segments = color_segments(next_frame, labels)
        boundaries_image = mark_boundaries(colored_segments, labels,
                                           draw_boundaries)

        flow_vectors_image = draw_flow_vectors(boundaries_image,
                                               labels,
                                               flow,
                                               flow_threshold=2.0)

        bounding_box_image = draw_bounding_boxes(flow_vectors_image,
                                                 labels,
                                                 flow,
                                                 flow_threshold=2.0)

        # Get current bounding boxes
        #current_boxes = []

        # Project bounding boxes to BEV and draw in BEV image
        num_labels = np.max(labels) + 1
        for label in range(1, num_labels):
            mask = labels == label
            if np.mean(np.linalg.norm(flow[mask], axis=1)) > 2.0:
                ys, xs = np.where(mask)
                x_min, x_max = np.min(xs), np.max(xs)
                y_min, y_max = np.min(ys), np.max(ys)

                # Define the extreme points of the blob (top-most, bottom-most, left-most, right-most)
                top_point = (xs[np.argmin(ys)], y_min)  # Top-most point
                bottom_point = (xs[np.argmax(ys)], y_max)  # Bottom-most point
                left_point = (x_min, ys[np.argmin(xs)])  # Left-most point
                right_point = (x_max, ys[np.argmax(xs)])  # Right-most point

                extreme_points = [
                    top_point, bottom_point, left_point, right_point
                ]
                # Check if the bounding box is within the ROI
                if x_min >= x_min_roi and x_max <= x_max_roi and y_min >= y_min_roi and y_max <= y_max_roi:
                    # TODO return np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]], dtype="float64")
                    box_corners = [
                        [x_min, y_max], [x_min, y_min], [x_max, y_min],
                        [x_max, y_max]
                    ]  #[(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

                    # Extract the flow vectors inside the box
                    flow_in_box = flow[mask]  # Flow vectors within the box
                    # Compute the average flow vector in the box
                    avg_flow_vector = np.mean(flow_in_box, axis=0)

                    if np.linalg.norm(avg_flow_vector) > 2.0:

                        # Define the start and end points of the flow vector in image coordinates
                        start_point_img = np.array(
                            [x_max, y_max], dtype=np.float32
                        )  # Assuming flow vector starts at origin within the box
                        end_point_img = np.array([
                            x_max + avg_flow_vector[0],
                            y_max + avg_flow_vector[1]
                        ],
                                                 dtype=np.float32)

                        # Project the start and end points of the flow vector to BEV
                        start_point_bev = map_box_to_BEV([start_point_img],
                                                         ipm_matrix)[0]
                        end_point_bev = map_box_to_BEV([end_point_img],
                                                       ipm_matrix)[0]

                        # Compute the BEV flow vector
                        avg_flow_vector_bev = end_point_bev - start_point_bev

                        # Compute the angle/orientation of the box based on the BEV flow vector
                        avg_flow_angle = np.arctan2(
                            avg_flow_vector_bev[1],
                            avg_flow_vector_bev[0])  # Angle in radians

                        # Compute the angle/orientation of the box based on average flow vector
                        #avg_flow_vector = map_box_to_BEV(avg_flow_vector, ipm_matrix)
                        # avg_flow_angle = np.arctan2(avg_flow_vector[1], avg_flow_vector[0])  # Angle in radians
                        avg_flow_angle_degrees = np.degrees(
                            avg_flow_angle)  # Convert to degrees

                        # Output the orientation
                        print(
                            f"Box {label}: Average flow angle (radians): {avg_flow_angle}"
                        )
                        print(
                            f"Box {label}: Average flow angle (degrees): {avg_flow_angle_degrees}"
                        )

                        # Find the predicted box position in the previous frame using the flow field
                        # prev_box_corners = []
                        # for (x, y) in box_corners:
                        # flow_at_point = flow[int(y), int(x)]  # Get the flow vector at (x, y)
                        # prev_x = x - flow_at_point[0]  # Backtrack using the flow
                        # prev_y = y - flow_at_point[1]
                        # prev_box_corners.append((prev_x, prev_y))

                        prev_extreme_points = []
                        for (x, y) in extreme_points:
                            flow_at_point = flow[int(y), int(
                                x)]  # Get the flow vector at the extreme point
                            prev_x = x - flow_at_point[
                                0]  # Backtrack using the flow
                            prev_y = y - flow_at_point[1]
                            prev_extreme_points.append((prev_x, prev_y))

                        prev_x_min, prev_x_max = np.min([
                            p[0] for p in prev_extreme_points
                        ]), np.max([p[0] for p in prev_extreme_points])
                        prev_y_min, prev_y_max = np.min([
                            p[1] for p in prev_extreme_points
                        ]), np.max([p[1] for p in prev_extreme_points])
                        prev_box_corners = [(prev_x_min, prev_y_max),
                                            (prev_x_min, prev_y_min),
                                            (prev_x_max, prev_y_min),
                                            (prev_x_max, prev_y_max)]

                        bev_points = map_box_to_BEV(box_corners, ipm_matrix)
                        bev_points1 = map_box_to_BEV(prev_box_corners,
                                                     ipm_matrix)

                        d, alpha, beta = compute_angles(bev_points, debug)
                        d1, alpha1, beta1 = compute_upper_angles(
                            bev_points1, debug)

                        angle_radians = move_to_second_quadrant(avg_flow_angle)

                        x, y = calculate_solution(np.pi - angle_radians, alpha,
                                                  beta, d,
                                                  np.pi - angle_radians,
                                                  alpha1, beta1, d1, debug)
                        if x is not None and y is not None:
                            x_up, y_up = calculate_solution(
                                -np.pi / 2 + angle_radians, alpha1, beta1, d1,
                                -np.pi / 2 + angle_radians, alpha1, beta1, d1,
                                debug)
                            if x_up is not None and y_up is not None:
                                if x_up < y: x_up = y
                                if y_up < x: y_up = x
                                bottom = compute_bottom_rectangle(
                                    bev_points, -angle_radians, x, y, debug)
                                top = compute_top_rectangle(
                                    bev_points, -np.pi / 2 - angle_radians,
                                    x_up, y_up, debug)
                                if bottom is not None and top is not None:
                                    center = bottom[0]
                                    center_top = top[0]
                                    #bottom = camera.convert_bev_to_world(bottom)
                                    #det.centers = [[bottom[0], det.angle_radians]]
                                    bottom = bottom[1:]
                                    top = top[1:]

                                    lower_face = map_box_to_BEV(
                                        bottom, inv_mat)
                                    upper_face = map_box_to_BEV(top, inv_mat)

                                    # Check if all points in lower_face and upper_face are within the bounding box
                                    # if np.all((lower_face[:, 0] >= x_min) & (lower_face[:, 0] <= x_max) &
                                    # (lower_face[:, 1] >= y_min) & (lower_face[:, 1] <= y_max)) and \
                                    # np.all((upper_face[:, 0] >= x_min) & (upper_face[:, 0] <= x_max) &
                                    # (upper_face[:, 1] >= y_min) & (upper_face[:, 1] <= y_max)):

                                    draw_cube(boundaries_image,
                                              lower_face.astype("int"),
                                              upper_face.astype("int"),
                                              color=cv_colors.ORANGE.value,
                                              thickness=1)
                                    #break

                        bev_image = draw_quadrangles_in_BEV(
                            bev_image, bev_points)
                        # Optionally, draw connections between matched boxes (visualizing tracking)
                        draw_tracking_line(boundaries_image,
                                           prev_box_corners,
                                           box_corners,
                                           color=(0, 0, 255))

        resized_image = resize_to_match_height(bounding_box_image,
                                               screen_height * 2)
        # Resize BEV image to match the height of bounding_box_image
        if show_warp:
            bev_image_resized = resize_to_match_height(
                crop_warp(bev_image),
                screen_height * 2)  #bounding_box_image.shape[0])
            combined_image = np.hstack((resized_image, bev_image_resized))
            resized_image = combined_image  #cv2.resize(combined_image, (screen_width, screen_height))

        prev_frame = next_frame
        cv2.imshow('Bounding Boxes', resized_image)

        if cv2.waitKey(30) & 0xFF == 27:  # Esc key to stop
            break

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
    video_path = '/home/dzhura/ComputerVision/data/bike.mp4'
    main(video_path, draw_boundaries=True, debug=True)
