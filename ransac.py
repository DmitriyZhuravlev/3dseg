import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

# Find the intersection of two lines defined by two pairs of points
def get_intersect(a1, a2, b1, b2):
    s = np.vstack([a1, a2, b1, b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1))))    # h for homogeneous
    l1 = np.cross(h[0], h[1])              # get first line
    l2 = np.cross(h[2], h[3])              # get second line
    x, y, z = np.cross(l1, l2)             # point of intersection
    if z == 0:                             # lines are parallel
        return None
    return (x / z, y / z)

def compute_intersection(line1, line2):
    """Compute the intersection of two lines in ax + by + c = 0 form."""
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) < 1e-10:
        return None  # Lines are parallel
    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant
    return (x, y)

def compute_3d_bounding_box(blob_contour, vanishing_points):
    """Estimate the 3D bounding box."""
    hull = cv2.convexHull(blob_contour)
    hull_points = hull[:, 0, :]
    tangents = {}
    for vp_key, vp in vanishing_points.items():
        tangents[vp_key] = []
        for pt in hull_points:
            a = vp[1] - pt[1]
            b = pt[0] - vp[0]
            c = -(a * pt[0] + b * pt[1])
            tangents[vp_key].append((a, b, c))
    A = compute_intersection(tangents['VP1'][0], tangents['VP2'][0])
    B = compute_intersection(tangents['VP2'][1], tangents['VP3'][0])
    C = compute_intersection(tangents['VP3'][1], tangents['VP1'][1])
    D = compute_intersection(tangents['VP1'][1], tangents['VP2'][1])
    E = compute_intersection(tangents['VP2'][0], tangents['VP3'][1])
    F = compute_intersection(tangents['VP3'][0], tangents['VP1'][0])
    bounding_box = [A, B, C, D, E, F]
    return bounding_box

def ransac_vanishing_points(lines):
    """Estimate vanishing points using RANSAC."""
    vanishing_points = {}
    for axis in ['VP1', 'VP2', 'VP3']:
        x_coords, y_coords = [], []
        for line in lines[axis]:
            x1, y1, x2, y2 = line
            x_coords.append(x1 - x2)
            y_coords.append(y1 - y2)
        model = RANSACRegressor()
        xy = np.column_stack((x_coords, y_coords))
        model.fit(xy[:, 0].reshape(-1, 1), xy[:, 1])
        slope = model.estimator_.coef_[0]
        intercept = model.estimator_.intercept_
        vanishing_points[axis] = (slope, intercept)
    return vanishing_points

def detect_blobs_optical_flow(prev_frame, curr_frame):
    """Detect blobs using optical flow."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    _, binary_mag = cv2.threshold(magnitude, 0.2, 1, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mag.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def process_video(video_path):
    """Process video to estimate 3D bounding boxes."""
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video.")
        return

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Blob detection
        contours = detect_blobs_optical_flow(prev_frame, curr_frame)

        # Extract vanishing points (dummy lines here, replace with real)
        lines = {'VP1': [(90, 524, 1133, 705)], 'VP2': [(150, 150, 250, 250)], 'VP3': [(300, 300, 400, 400)]}
        vanishing_points = ransac_vanishing_points(lines)

        # Bounding box calculation
        for contour in contours:
            bbox = compute_3d_bounding_box(contour, vanishing_points)
            for point in bbox:
                if point:
                    cv2.circle(curr_frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

        # Display the result
        cv2.imshow("Frame", curr_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update previous frame
        prev_frame = curr_frame.copy()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video("/home/dzhura/ComputerVision/data/4kStreetViewCctv.mp4")
