import cv2
import numpy as np
import os

from lifting import *
from enum import Enum

from render import *

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
    WHITE = (255,255,255)
    BLACK = (0, 0, 0)

debug = True
use_slick = False
show_warp = False
resize_shape = (1600, 900)
screen_width, screen_height = 1920 // 2, 1080 // 2
roi_corners = None

def main():
    # Define paths for input and output videos
    input_video_path = '/home/dzhura/ComputerVision/data/bike.mp4'
    output_video_path = 'output_with_cubes.mp4'
    avg_frame_path = 'avg_frame.jpg'  # Path to save/load the average frame

    # Initialize video capture and video writer
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, output_size)

    # Select ROI corners and compute perspective matrix
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the first frame from the video.")
        return

    roi_corners = select_points(first_frame)  # Select points manually for ROI
    persp_matrix, inv_persp_matrix, warped_size = generate_perspective_matrix(first_frame, roi_corners)

    # Compute average frame for background subtraction
    avg_frame = compute_average_frame(input_video_path, avg_frame_path)
    
    prev_frame = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame if needed
        frame = cv2.resize(frame, resize_shape)
        
        # Perform background subtraction
        diff = cv2.absdiff(frame, avg_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

        # Extract the ROI and warp it
        warped_frame = unwarp(frame, roi_corners)
        
        if prev_frame is not None:
            # Compute optical flow
            flow = compute_optical_flow(prev_frame, frame)
            
            # Mark segment boundaries and color segments
            labels = np.zeros_like(gray_diff)
            labeled_img = color_segments(frame, labels)
            boundary_img = mark_boundaries(frame, labels)

            # Draw flow vectors and bounding boxes
            frame_with_flow = draw_flow_vectors(frame, labels, flow)
            frame_with_boxes = draw_bounding_boxes(frame_with_flow, labels, flow)
            
            # BEV visualization
            box_corners = [(0, 0), (100, 0), (100, 100), (0, 100)]  # Placeholder bounding box corners
            bev_points = map_box_to_BEV_with_padding(box_corners, inv_persp_matrix, padding_size=50)
            bev_frame = draw_quadrangles_in_BEV(warped_frame, bev_points)

            # Show and save the result
            cv2.imshow("Frame with Cubes", frame_with_boxes)
            cv2.imshow("BEV", bev_frame)
            out.write(frame_with_boxes)  # Save output frame

        prev_frame = frame.copy()
        frame_count += 1

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
