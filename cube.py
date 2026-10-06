import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
#from shapely.geometry import Polygon
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import scale
from lifting import *


def compute_other_3d_faces(nb_cube_3d):
    """
    Compute other 3D faces based on left and right.
    This function reconstructs the missing faces of the 3D cube.
    """
    nb_cube_3d["top"] = np.copy(nb_cube_3d["upper"])
    nb_cube_3d["top"][:, 1] -= (nb_cube_3d["lower"][:, 1] - nb_cube_3d["upper"][:, 1])

    nb_cube_3d["bottom"] = np.copy(nb_cube_3d["lower"])
    nb_cube_3d["bottom"][:, 1] += (nb_cube_3d["lower"][:, 1] - nb_cube_3d["upper"][:, 1])

    return nb_cube_3d


def dist(points1, points2):
    return np.linalg.norm(np.array(points1[0]) - np.array(points2[0]))


def faces_overlap(face1, face2):

    # Define polygons for each face based on the corner points
    p1 = Polygon([face1[0], face1[1], face1[2], face1[3]])
    p2 = Polygon([face2[0], face2[1], face2[2], face2[3]])

    # Check if the polygons intersect (overlap)
    return p1.intersects(p2)
    
def faces_overlap_area(face1, face2):

    p1 = Polygon(face1)
    p2 = Polygon(face2)

    # Compute intersection area
    if p1.intersects(p2):
        return p1.intersection(p2).area
    
    return 0.0  # No overlap
    
def expand_polygon(polygon, scale_factor=1.2):
    """
    Expands a polygon by a given scale factor relative to its centroid.
    
    :param polygon: List of (x, y) points defining the polygon
    :param scale_factor: Factor to scale the polygon (default: 1.1 for 10% expansion)
    :return: Expanded polygon as a Shapely Polygon object
    """
    poly = Polygon(polygon)
    centroid = poly.centroid
    expanded_poly = scale(poly, xfact=scale_factor, yfact=scale_factor, origin=centroid)
    return expanded_poly

def faces_iou(face1, face2, scale_factor=1.5):
    """
    Compute the Intersection over Union (IoU) of two polygonal faces, 
    after expanding them by a relative margin.
    
    :param face1: List of (x, y) points defining the first face
    :param face2: List of (x, y) points defining the second face
    :param scale_factor: Factor to expand the faces before computing IoU (default: 1.0)
    :return: IoU value (0.0 to 1.0)
    """
    poly1 = Polygon(face1)
    poly2 = Polygon(face2)

    if scale_factor != 1.0:
        p1 = expand_polygon(face1, scale_factor)
        p2 = expand_polygon(face2, scale_factor)
    else:
        p1 = poly1
        p2 = poly2

    if not p1.is_valid or not p2.is_valid:
        return 0.0  # Invalid polygons

    if not p1.intersects(p2):
        return 0.0  # No overlap
        
    return 1.0

    # intersection_area = p1.intersection(p2).area
    # union_area = p1.union(p2).area

    # return intersection_area / union_area if union_area != 0 else 0.0
# def faces_iou(face1, face2):
    # """
    # Compute the Intersection over Union (IoU) of two polygonal faces.
    
    # :param face1: List of (x, y) points defining the first face
    # :param face2: List of (x, y) points defining the second face
    # :return: IoU value (0.0 to 1.0)
    # """
    # p1 = Polygon(face1)
    # p2 = Polygon(face2)

    # if not p1.intersects(p2):
        # return 0.0  # No overlap

    # # Compute intersection and union areas
    # intersection_area = p1.intersection(p2).area
    # union_area = p1.union(p2).area

    # # Compute IoU
    # iou = intersection_area / union_area if union_area != 0 else 0.0
    # return iou
    
def find_largest_intersection(face, neighbor_faces):
    largest_area = 0
    largest_overlap = None
    
    for neighbor_face in neighbor_faces:
        area = faces_overlap_area(face, neighbor_face)
        if area > largest_area:
            largest_area = area
            largest_overlap = overlap

    return largest_overlap

def process_cube_intersections(marked_image, pr_boxes, neighbors_dict, inv_ipm_matrix, top_inv_ipm_matrix):

    for segment_index, (lower_face, upper_face) in enumerate(pr_boxes):
        neighbors = neighbors_dict.get(segment_index, [])

        for neighbor_index in neighbors:
            neighbor_lower, neighbor_upper = pr_boxes[neighbor_index]

            # Get overlapping faces (largest intersection)
            face_intersections = {}
            for face_name in ["front", "right", "back", "left"]:
                main_face = get_projected_cube_faces(lower_face, upper_face)[face_name]
                neighbor_faces = [
                    get_projected_cube_faces(neighbor_lower, neighbor_upper)[face_name]
                ]
                
                largest_overlap = find_largest_intersection(main_face, neighbor_faces)
                if largest_overlap is not None:
                    face_intersections[face_name] = largest_overlap

            # Convert intersection faces to 3D using inv_ipm_matrix
            for face_name, overlap in face_intersections.items():
                pr_overlap = map_points_to_BEV(overlap, inv_ipm_matrix)
                height = calc_height(lower_face, upper_face, pr_overlap, inv_ipm_matrix, top_inv_ipm_matrix, object_height)
                
                # Draw the adjusted intersection area on the image
                overlay = marked_image.copy()
                cv2.fillPoly(overlay, [np.array(overlap, dtype=np.int32)], (255, 255, 255))
                marked_image = cv2.addWeighted(overlay, 0.5, marked_image, 0.5, 0)

    return marked_image

def get_projected_cube_faces(lower_face, upper_face):

    faces = {
        "lower": np.array(lower_face),
        "upper": np.array(upper_face),
        "front": np.array([lower_face[0], upper_face[0], upper_face[1], lower_face[1]]),
        "right": np.array([lower_face[1], upper_face[1], upper_face[2], lower_face[2]]),
        "back":  np.array([lower_face[2], upper_face[2], upper_face[3], lower_face[3]]),
        "left":  np.array([lower_face[0], upper_face[0], upper_face[3], lower_face[3]])
    }

    return faces
    
def draw_projected_cube(faces):
    """
    Draws the projected cube using Matplotlib, adjusting opacity based on face orientation.

    Args:
        faces (dict): A dictionary containing the projected faces of the cube.
    """
    fig, ax = plt.subplots()
    
    # Define opacity levels (lower values = more transparent)
    face_opacity = {
        "lower": 0.7,
        "upper": 0.3,
        "front": 0.8,
        "right": 0.6,
        "back": 0.2,
        "left": 0.5
    }

    # Define face colors
    face_colors = {
        "lower": "gray",
        "upper": "blue",
        "front": "red",
        "right": "green",
        "back": "purple",
        "left": "orange"
    }

    # Draw faces in the correct order (back -> left -> right -> front -> lower -> upper)
    draw_order = ["back", "left", "right", "front", "lower", "upper"]

    for face in draw_order:
        polygon = patches.Polygon(faces[face], closed=True, 
                                  facecolor=face_colors[face], 
                                  edgecolor="black",
                                  alpha=face_opacity[face])
        ax.add_patch(polygon)

    # Adjust axes
    ax.set_xlim(min(p[0] for f in faces.values() for p in f) - 10,
                max(p[0] for f in faces.values() for p in f) + 10)
    ax.set_ylim(min(p[1] for f in faces.values() for p in f) - 10,
                max(p[1] for f in faces.values() for p in f) + 10)
    
    ax.set_aspect("equal")
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.show()

def draw_projected_cube_on_image(image, faces):
    """
    Draws the projected cube onto an existing image with adjusted face opacity.
    
    Args:
        image (numpy array): The base image (marked_image).
        faces (dict): A dictionary containing the projected faces of the cube.
    
    Returns:
        numpy array: Image with the projected cube drawn on it.
    """
    overlay = image.copy()

    # Define opacity levels (lower values = more transparent)
    face_opacity = {
        "lower":  0.5, #0.7,
        "upper":  0.5, #0.3,
        "front":  0.5, #0.8,
        "right":  0.5, #0.6,
        "back":  0.9, #0.2,
        "left": 0.9
    }

    # Define face colors in BGR (OpenCV format)
    face_colors = {
        "lower": (128, 128, 128),  # Gray
        "upper": (255, 0, 0),      # Blue
        "front": (0, 0, 255),      # Red
        "right": (0, 255, 0),      # Green
        "back": (128, 0, 128),     # Purple
        "left": (0, 165, 255)      # Orange
    }

    # Draw faces in correct order (back -> left -> right -> lower -> upper -> front)
    draw_order = ["back", "left", "right", "lower", "upper", "front"]

    for face in draw_order:
        points = np.array(faces[face], dtype=np.int32)

        # Create a separate transparent layer for blending
        temp_overlay = image.copy()

        # Fill the face on temp overlay
        cv2.fillPoly(temp_overlay, [points], face_colors[face])

        # Blend face with the main overlay
        cv2.addWeighted(temp_overlay, face_opacity[face], overlay, 1 - face_opacity[face], 0, overlay)

        # Draw face edges
        cv2.polylines(overlay, [points], isClosed=True, color=(0, 0, 0), thickness=2)

    return overlay
    
def draw_cube(image,
              lower_face,
              upper_face,
              color, #=cv_colors.RED.value,
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

    # for i, point in enumerate(lower_face):
        # # Define the color for the circle
        # color_i=list(cv_colors)[i % (len(cv_colors)-1)].value

        # # Draw the circle
        # cv2.circle(
            # image,
            # center=(int(point[0]), int(point[1])),  # Convert to integer coordinates
            # radius=15,  # Circle radius
            # color=color_i,
            # thickness=-1  # Filled circle
        # )    

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
    
def get_projected_box(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False):
    #mask = (labels == label)
    lower_face, upper_face = None, None
    extrem = None
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
            return None, None, None
        
        # Calculate mask center
        mask_center = np.mean(points, axis=0)
        # lower_face, upper_face, extrem = compute_3d_box_from_plain_mask_new(
                # mask, vert_vp, hor_left_vp, hor_right_vp, debug)
        # Compute 3D box from plain mask
        try:
            lower_face, upper_face, extrem = compute_3d_box_from_plain_mask_new(
                mask, vert_vp, hor_left_vp, hor_right_vp, debug)
        except Exception as e:
            print(f"Error processing segment with label")# {label}: {e}")
            return None, None, None
        
    return lower_face, upper_face, extrem
    
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

    extrem = [point_a, point_b, point_c, point_d, point_e, point_f]
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

    return np.array([corner_a, corner_b, corner_h, corner_c]), np.array([corner_a1, corner_b1, corner_h1, corner_c1]), np.array(extrem)
