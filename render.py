import cv2
import numpy as np
import os

from lifting import *
from enum import Enum

# Initialize video capture and output
input_video_path = '/home/dzhura/ComputerVision/data/bike.mp4'
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
show_warp = False
resize_shape = (1600, 900)
screen_width, screen_height = 1920 // 2, 1080 // 2
roi_corners = None


def unwarp(img, roi_corners):
    src = np.float32(roi_corners)
    warped_size = (img.shape[1], img.shape[0])
    offset = int(warped_size[0] / 3.0)

    dst = np.float32([[offset, warped_size[1]],
                      [offset, 0],
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
    top_padding, left_padding = 0, 0
    warped_size = (img.shape[1] + 2 * left_padding, img.shape[0] + top_padding)
    offset = int((warped_size[0] - 2 * left_padding) / 3.0)

    dst = np.float32([[offset + left_padding, warped_size[1]],
                      [offset + left_padding, top_padding],
                      [warped_size[0] - offset - left_padding, top_padding],
                      [warped_size[0] - offset - left_padding, warped_size[1]]])

    persp = cv2.getPerspectiveTransform(src, dst)
    inv = cv2.getPerspectiveTransform(dst, src)
    return persp, inv, warped_size


def crop_warp(warped):
    top_padding, left_padding = 0, 0
    warped_size = (warped.shape[1], warped.shape[0])
    offset = 0

    return warped[top_padding: warped_size[1], left_padding + offset: warped_size[0] - left_padding - offset]


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

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def mark_boundaries(image, labels, draw_boundaries=True):
    if not draw_boundaries:
        return image

    boundaries = np.copy(image)
    for y in range(1, labels.shape[0] - 1):
        for x in range(1, labels.shape[1] - 1):
            if (labels[y, x] != labels[y - 1, x] or
                    labels[y, x] != labels[y + 1, x] or
                    labels[y, x] != labels[y, x - 1] or
                    labels[y, x] != labels[y, x + 1]):
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


def draw_bounding_boxes(image, labels, flow, flow_threshold=2.0):
    """Draw bounding boxes over segments where flow > threshold."""
    num_labels = np.max(labels) + 1
    for label in range(1, num_labels):
        mask = labels == label
        if np.mean(np.linalg.norm(flow[mask], axis=1)) > flow_threshold:
            ys, xs = np.where(mask)
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
    return image


def map_box_to_BEV(box_corners, ipm_matrix):
    """Project bounding box corners onto the BEV using IPM."""
    box_corners = np.array(box_corners, dtype=np.float32)
    box_corners = np.array([box_corners])  # Reshape to (1, N, 2) for perspectiveTransform
    bev_points = cv2.perspectiveTransform(box_corners, ipm_matrix)
    return bev_points[0]  # Return as (N, 2)


def draw_quadrangles_in_BEV(bev_image, bev_points):
    """Draw quadrangles (projected bounding boxes) in the BEV."""
    bev_points = np.int32(bev_points)
    cv2.polylines(bev_image, [bev_points], isClosed=True, color=(0, 255, 255), thickness=2)
    return bev_image


def add_padding(image, padding_size):
    """
    Adds padding to an image.
    """
    height, width = image.shape[:2]
    padded_image = np.zeros((height + 2 * padding_size, width + 2 * padding_size, 3), dtype=np.uint8)
    padded_image[padding_size:padding_size+height, padding_size:padding_size+width] = image
    return padded_image


def map_box_to_BEV_with_padding(box_corners, ipm_matrix, padding_size):
    """
    Maps bounding box corners to BEV coordinates and accounts for padding in the BEV image.
    """
    bev_points = map_box_to_BEV(box_corners, ipm_matrix)
    # Adjust points for padding
    bev_points[:, 0] += padding_size
    bev_points[:, 1] += padding_size
    return bev_points
