#import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
#from shapely.geometry import Polygon
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import scale
from lifting import *

from scipy.spatial.distance import cdist
from skimage.measure import EllipseModel
from scipy.spatial import ConvexHull

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

# def faces_iou(face1, face2, scale_factor=1.5):
    # """
    # Compute the Intersection over Union (IoU) of two polygonal faces, 
    # after expanding them by a relative margin.
    
    # :param face1: List of (x, y) points defining the first face
    # :param face2: List of (x, y) points defining the second face
    # :param scale_factor: Factor to expand the faces before computing IoU (default: 1.0)
    # :return: IoU value (0.0 to 1.0)
    # """
    # poly1 = Polygon(face1)
    # poly2 = Polygon(face2)

    # if scale_factor != 1.0:
        # p1 = expand_polygon(face1, scale_factor)
        # p2 = expand_polygon(face2, scale_factor)
    # else:
        # p1 = poly1
        # p2 = poly2

    # if not p1.is_valid or not p2.is_valid:
        # return 0.0  # Invalid polygons

    # if not p1.intersects(p2):
        # return 0.0  # No overlap
        
    # return 1.0

def faces_intersection(face1, face2):
    """
    face1, face2 are np.array of shape (4,2) (2D quadrilaterals)
    """
    from shapely.geometry import Polygon

    poly1 = Polygon(face1)
    poly2 = Polygon(face2)

    if not poly1.is_valid or not poly2.is_valid:
        return None

    return poly1.intersection(poly2)

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

# face_pairs = {
    # "left": ("left", "right"),
    # "right": ("right", "left"),
    # "top": ("upper", "lower"),
    # "bottom": ("lower", "upper"),
    # "forward": ("forward", "backward"),
    # "backward": ("backward","forward" )
    
# }

def fix_cube(lower_face, upper_face, y_width):
    """
    Adjusts only the Y-direction (index 1) width of the cube by modifying face[3] and face[2],
    keeping face[0] and face[1] unchanged.

    Parameters:
        lower_face (ndarray): (4, 3) array for bottom face.
        upper_face (ndarray): (4, 3) array for top face.
        y_width (float): Target offset in Y-direction.

    Returns:
        tuple: (fixed_lower_face, fixed_upper_face)
    """
    lower_face = np.array(lower_face, dtype=float)
    upper_face = np.array(upper_face, dtype=float)

    fixed_lower = lower_face.copy()
    fixed_upper = upper_face.copy()

    # Modify only the Y (index 1) component of face[3] and face[2]
    fixed_lower[3][1] = fixed_lower[0][1] + y_width
    fixed_lower[2][1] = fixed_lower[1][1] + y_width

    fixed_upper[3][1] = fixed_upper[0][1] + y_width
    fixed_upper[2][1] = fixed_upper[1][1] + y_width

    return fixed_lower, fixed_upper

def get_projected_cube_faces(lower_face, upper_face):

    faces = {
        "bottom": np.array(lower_face),
        "top": np.array(upper_face),
        "forward": np.array([lower_face[3], upper_face[3], upper_face[2], lower_face[2]]),
        "backward": np.array([lower_face[0], upper_face[0], upper_face[1], lower_face[1]]),
        "right": np.array([lower_face[1], upper_face[1], upper_face[2], lower_face[2]]),
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

def compute_3d_box_projection_last(mask, mask_center, vert_vp, hor_left_vp, hor_right_vp, debug=False):
    def compute_intersection(line1, line2):
        """Compute the intersection point of two lines in homogeneous coordinates."""
        inter = np.cross(line1, line2)
        if abs(inter[2]) > 1e-6:
            return inter[:2] / inter[2]
        return None

    def safe_intersect(p1, p2, p3, p4):
        """Get intersection point of two lines defined by (p1, p2) and (p3, p4)."""
        line1 = LineString([p1, p2])
        line2 = LineString([p3, p4])
        if line1.intersects(line2):
            pt = line1.intersection(line2)
            if pt.geom_type == "Point":
                return (pt.x, pt.y)
        return None

    def try_intersection_with_eps(p1, vp1, p2, vp2, eps=1e-1):
        """Try intersections with small perturbations in both directions."""
        for sign in [1, -1]:
            offset = eps * sign
            shifted_p1 = (p1[0] + offset, p1[1] + offset)
            shifted_p2 = (p2[0] - offset, p2[1] - offset)
            pt = safe_intersect(shifted_p1, vp1, shifted_p2, vp2)
            if pt:
                return pt
        return None

    # Step 1: Extract boundary points from the mask
    points = np.argwhere(mask > 0)
    points = points[:, [1, 0]]  # Convert to (x, y)

    if len(points) < 3:
        raise ValueError(f"Not enough points in mask for a valid polygon. Found {len(points)} points.")

    hull_points = points
    mask_center = mask_center[[1, 0]]  # Convert to (x, y)

    # Step 2: Compute tangents from vanishing points
    #from your_module import find_tangent_points, get_intersect  # Replace with actual module
    hor_right_pos, hor_right_pos_line, hor_right_neg, hor_right_neg_line = find_tangent_points(mask_center, hull_points, hor_right_vp)
    vert_pos, vert_pos_line, vert_neg, vert_neg_line = find_tangent_points(mask_center, hull_points, vert_vp)
    hor_left_pos, hor_left_pos_line, hor_left_neg, hor_left_neg_line = find_tangent_points(mask_center, hull_points, hor_left_vp)

    extrem = [hor_right_neg, hor_right_pos, vert_neg, vert_pos, hor_left_neg, hor_left_pos]

    # Step 3: Compute corners (projected 3D bounding box base)
    corner_a = compute_intersection(hor_right_neg_line, hor_left_pos_line)
    corner_c = compute_intersection(hor_left_pos_line, vert_neg_line)
    corner_b = compute_intersection(vert_pos_line, hor_right_neg_line)
    corner_r = compute_intersection(hor_left_neg_line, vert_pos_line)

    # Step 4: Compute corner_h (opposite of corner_a)
    corner_h = safe_intersect(corner_c, hor_right_vp, corner_b, hor_left_vp)
    if corner_h is None:
        corner_h = try_intersection_with_eps(corner_c, hor_right_vp, corner_b, hor_left_vp)
        if debug and corner_h:
            print("corner_h found with epsilon perturbation.")

    # Step 5: Upper face corners
    corner_c1 = compute_intersection(hor_right_pos_line, vert_neg_line)
    corner_b1 = compute_intersection(hor_left_neg_line, vert_pos_line)

    # corner_a1 (top opposite of corner_h1)
    corner_a11 = get_intersect(hor_left_vp, corner_c1, vert_vp, corner_a)
    corner_a12 = get_intersect(hor_right_vp, corner_b1, vert_vp, corner_a)
    corner_a1 = corner_a11 if corner_a11[1] < corner_a12[1] else corner_a12

    # Refine corner_c1, corner_b1 from corner_a1
    corner_c1 = get_intersect(hor_left_vp, corner_a1, vert_vp, corner_c)
    corner_b1 = get_intersect(hor_right_vp, corner_a1, vert_vp, corner_b)

    # Compute center projections for top face
    center = get_intersect(corner_c, corner_b, corner_a, corner_h)
    center1 = get_intersect(corner_c1, corner_b1, vert_vp, center)
    corner_h1 = get_intersect(corner_a1, center1, vert_vp, corner_h)

    if debug:
        print("Base corners: A, B, H, C")
        print("A:", corner_a, "B:", corner_b, "H:", corner_h, "C:", corner_c)
        print("Top corners: A1, B1, H1, C1")
        print("A1:", corner_a1, "B1:", corner_b1, "H1:", corner_h1, "C1:", corner_c1)

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
        plot_point(corner_r, 'orange', "corner r")

        # Plot tangent lines
        draw_line_segment(hor_right_vp, hor_right_neg, 'red')
        draw_line_segment(hor_right_vp, hor_right_pos, 'red')
        draw_line_segment(vert_vp, vert_neg, 'blue')
        draw_line_segment(vert_vp, vert_pos, 'blue')
        draw_line_segment(hor_left_vp, hor_left_neg, 'green')
        draw_line_segment(hor_left_vp, hor_left_pos, 'green')
    
        # Zoom into the mask region
        rows, cols = np.where(mask)
        pad = 100
        if rows.size > 0 and cols.size > 0:
            plt.xlim([cols.min() - pad, cols.max() + pad])
            plt.ylim([rows.max() + pad, rows.min() - pad])  # Invert y-axis for correct orientation
    
        # Show legend and final plot
        plt.legend()
        plt.show()

    base = np.array([corner_a, corner_b, corner_h, corner_c])
    top = np.array([corner_a1, corner_b1, corner_h1, corner_c1])
    return base, top, np.array(extrem)
    
def get_projected_box(mask, mask_center, vert_vp, hor_left_vp, hor_right_vp, internal = True, debug=False):
    #mask = (labels == label)
    lower_face, upper_face = None, None
    extrem = None
    ys, xs = np.where(mask)
    
    if xs.size > 0 and ys.size > 0:
        # Compute boundary points
        # x_min, x_max = np.min(xs), np.max(xs)
        # y_min, y_max = np.min(ys), np.max(ys)
        # top_point = (xs[np.argmin(ys)], y_min)
        # bottom_point = (xs[np.argmax(ys)], y_max)
        # left_point = (x_min, ys[np.argmin(xs)])
        # right_point = (x_max, ys[np.argmax(xs)])
        
        # Collect all points in the mask
        points = np.argwhere(mask > 0)#[:, [1, 0]]  # Switch to (x, y) format
        
        if len(points) < 3:
            # Skip segments with fewer than 3 points
            return None, None, None

        try:
            lower_face, upper_face, extrem = compute_3d_box_projection_last( #compute_3d_box_projection(
                mask, mask_center, vert_vp, hor_left_vp, hor_right_vp, debug)
        except Exception as e:
            print(f"Error processing segment with label")# {label}: {e}")
            return None, None, None

    return lower_face, upper_face, extrem
    
def get_internal_box(mask, mask_center, vert_vp, hor_left_vp, hor_right_vp, debug=False):
    #mask = (labels == label)
    lower_face, upper_face = None, None
    extrem = None
    ys, xs = np.where(mask)
    
    if xs.size > 0 and ys.size > 0:
        points = np.argwhere(mask > 0)#[:, [1, 0]]  # Switch to (x, y) format
        
        if len(points) < 3:
            # Skip segments with fewer than 3 points
            return None, None, None

        #try:
        if True:
            lower_face, upper_face, extrem = compute_3d_box_projection_common( #compute_3d_box_projection_top(
                mask, mask_center, vert_vp, hor_left_vp, hor_right_vp, debug)
        #except Exception as e:
        else:
            print(f"Error processing segment with label")# {label}: {e}")
            return None, None, None

    return lower_face, upper_face, extrem
    
def compute_3d_box_projection_common(mask, mask_center, vert_vp, hor_left_vp, hor_right_vp, debug=False):
    def compute_intersection(line1, line2):
        inter = np.cross(line1, line2)
        if abs(inter[2]) > 1e-6:
            return inter[:2] / inter[2]
        return None

    def safe_intersect(p1, p2, p3, p4):
        line1 = LineString([p1, p2])
        line2 = LineString([p3, p4])
        if line1.intersects(line2):
            pt = line1.intersection(line2)
            if pt.geom_type == "Point":
                return (pt.x, pt.y)
        return None

    def try_intersection_with_eps(p1, vp1, p2, vp2, eps=1e-1):
        for sign in [1, -1]:
            offset = eps * sign
            shifted_p1 = (p1[0] + offset, p1[1] + offset)
            shifted_p2 = (p2[0] - offset, p2[1] - offset)
            pt = safe_intersect(shifted_p1, vp1, shifted_p2, vp2)
            if pt:
                return pt
        return None

    # Step 1: Extract boundary points from the mask
    points = np.argwhere(mask > 0)
    points = points[:, [1, 0]]  # Convert to (x, y)

    if len(points) < 3:
        raise ValueError(f"Not enough points in mask for a valid polygon. Found {len(points)} points.")

    hull_points = points
    mask_center = mask_center[[1, 0]]  # Convert to (x, y)

    # Step 2: Compute tangents
    hor_right_pos, hor_right_pos_line, hor_right_neg, hor_right_neg_line = find_tangent_points(mask_center, hull_points, hor_right_vp)
    vert_pos, vert_pos_line, vert_neg, vert_neg_line = find_tangent_points(mask_center, hull_points, vert_vp)
    hor_left_pos, hor_left_pos_line, hor_left_neg, hor_left_neg_line = find_tangent_points(mask_center, hull_points, hor_left_vp)

    extrem = [hor_right_neg, hor_right_pos, vert_neg, vert_pos, hor_left_neg, hor_left_pos]

    corner_a = compute_intersection(hor_left_pos_line, hor_right_neg_line)
    corner_b = compute_intersection(hor_right_neg_line, vert_pos_line)
    corner_b1 = compute_intersection(hor_left_neg_line, vert_pos_line)
    corner_a1 = get_intersect(corner_a, vert_vp, corner_b1, hor_right_vp)
    corner_h1 = compute_intersection(hor_left_neg_line, hor_right_pos_line)
    corner_c1 = get_intersect(corner_h1, hor_right_vp, corner_a1, hor_left_vp)
    corner_c = get_intersect(corner_c1, vert_vp, corner_a, hor_left_vp)
    corner_h = get_intersect(corner_h1, vert_vp, corner_c, hor_right_vp)


    if debug:
        print("Top corners: A1, B1, H1, C1")
        print("A1:", corner_a1, "B1:", corner_b1, "H1:", corner_h1, "C1:", corner_c1)
        print("Bottom corners: A, B, H, C")
        print("A:", corner_a, "B:", corner_b, "H:", corner_h, "C:", corner_c)

        plt.imshow(mask, cmap='gray')
        def plot_point(corner, color, label):
            if corner is not None and len(corner) == 2:
                plt.scatter(*corner, color=color, label=label)

        plot_point(corner_a, 'red', "corner a")
        plot_point(corner_b, 'blue', "corner b")
        plot_point(corner_c, 'green', "corner c")
        plot_point(corner_h, 'yellow', "corner h")
        # plot_point(corner_a1, 'cyan', "corner a1")
        # plot_point(corner_b1, 'magenta', "corner b1")
        # plot_point(corner_c1, 'orange', "corner c1")
        # plot_point(corner_h1, 'purple', "corner h1")

        rows, cols = np.where(mask)
        pad = 100
        if rows.size > 0 and cols.size > 0:
            plt.xlim([cols.min() - pad, cols.max() + pad])
            plt.ylim([rows.max() + pad, rows.min() - pad])

        plt.legend()
        plt.show()

    base = np.array([corner_a, corner_b, corner_h, corner_c])
    top = np.array([corner_a1, corner_b1, corner_h1, corner_c1])

    #return base, top, np.array(extrem)
    
    extrem_orig = extrem.copy()
    
    ########################################################################
    # Step 1.5: Add computed 3D box corner points to the point set
    additional_points = np.array([
        corner_a, corner_b, corner_h, corner_c,
        corner_a1, corner_b1, corner_h1, corner_c1
    ], dtype=np.float32)
    
    # Keep only valid (non-None) points and round to integers
    additional_points = np.array([
        [int(round(p[0])), int(round(p[1]))]
        for p in additional_points if p is not None
    ])
    
    # Append to the points array
    points = np.vstack([points, additional_points])
    ###########################################################
    hull_points = points
    #mask_center = mask_center[[1, 0]]  # Convert to (x, y)

    # Step 2: Compute tangents from vanishing points
    #from your_module import find_tangent_points, get_intersect  # Replace with actual module
    hor_right_pos, hor_right_pos_line, hor_right_neg, hor_right_neg_line = find_tangent_points(mask_center, hull_points, hor_right_vp)
    vert_pos, vert_pos_line, vert_neg, vert_neg_line = find_tangent_points(mask_center, hull_points, vert_vp)
    hor_left_pos, hor_left_pos_line, hor_left_neg, hor_left_neg_line = find_tangent_points(mask_center, hull_points, hor_left_vp)

    extrem = [hor_right_neg, hor_right_pos, vert_neg, vert_pos, hor_left_neg, hor_left_pos]

    # Step 3: Compute corners (projected 3D bounding box base)
    corner_a = compute_intersection(hor_right_neg_line, hor_left_pos_line)
    corner_c = compute_intersection(hor_left_pos_line, vert_neg_line)
    corner_b = compute_intersection(vert_pos_line, hor_right_neg_line)
    corner_r = compute_intersection(hor_left_neg_line, vert_pos_line)

    # Step 4: Compute corner_h (opposite of corner_a)
    corner_h = safe_intersect(corner_c, hor_right_vp, corner_b, hor_left_vp)
    if corner_h is None:
        corner_h = try_intersection_with_eps(corner_c, hor_right_vp, corner_b, hor_left_vp)
        if debug and corner_h:
            print("corner_h found with epsilon perturbation.")

    # Step 5: Upper face corners
    corner_c1 = compute_intersection(hor_right_pos_line, vert_neg_line)
    corner_b1 = compute_intersection(hor_left_neg_line, vert_pos_line)

    # corner_a1 (top opposite of corner_h1)
    corner_a11 = get_intersect(hor_left_vp, corner_c1, vert_vp, corner_a)
    corner_a12 = get_intersect(hor_right_vp, corner_b1, vert_vp, corner_a)
    corner_a1 = corner_a11 if corner_a11[1] < corner_a12[1] else corner_a12

    # Refine corner_c1, corner_b1 from corner_a1
    corner_c1 = get_intersect(hor_left_vp, corner_a1, vert_vp, corner_c)
    corner_b1 = get_intersect(hor_right_vp, corner_a1, vert_vp, corner_b)

    # Compute center projections for top face
    center = get_intersect(corner_c, corner_b, corner_a, corner_h)
    center1 = get_intersect(corner_c1, corner_b1, vert_vp, center)
    corner_h1 = get_intersect(corner_a1, center1, vert_vp, corner_h)

    if debug:
        print("Base corners: A, B, H, C")
        print("A:", corner_a, "B:", corner_b, "H:", corner_h, "C:", corner_c)
        print("Top corners: A1, B1, H1, C1")
        print("A1:", corner_a1, "B1:", corner_b1, "H1:", corner_h1, "C1:", corner_c1)

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
        plot_point(corner_r, 'orange', "corner r")

        # Plot tangent lines
        draw_line_segment(hor_right_vp, hor_right_neg, 'red')
        draw_line_segment(hor_right_vp, hor_right_pos, 'red')
        draw_line_segment(vert_vp, vert_neg, 'blue')
        draw_line_segment(vert_vp, vert_pos, 'blue')
        draw_line_segment(hor_left_vp, hor_left_neg, 'green')
        draw_line_segment(hor_left_vp, hor_left_pos, 'green')
    
        # Zoom into the mask region
        rows, cols = np.where(mask)
        pad = 100
        if rows.size > 0 and cols.size > 0:
            plt.xlim([cols.min() - pad, cols.max() + pad])
            plt.ylim([rows.max() + pad, rows.min() - pad])  # Invert y-axis for correct orientation
    
        # Show legend and final plot
        plt.legend()
        plt.show()

    base = np.array([corner_a, corner_b, corner_h, corner_c])
    top = np.array([corner_a1, corner_b1, corner_h1, corner_c1])

    return base, top, np.array(extrem_orig)
    
def compute_3d_box_projection_top(mask, mask_center, vert_vp, hor_left_vp, hor_right_vp, debug=False):
    def compute_intersection(line1, line2):
        inter = np.cross(line1, line2)
        if abs(inter[2]) > 1e-6:
            return inter[:2] / inter[2]
        return None

    def safe_intersect(p1, p2, p3, p4):
        line1 = LineString([p1, p2])
        line2 = LineString([p3, p4])
        if line1.intersects(line2):
            pt = line1.intersection(line2)
            if pt.geom_type == "Point":
                return (pt.x, pt.y)
        return None

    def try_intersection_with_eps(p1, vp1, p2, vp2, eps=1e-1):
        for sign in [1, -1]:
            offset = eps * sign
            shifted_p1 = (p1[0] + offset, p1[1] + offset)
            shifted_p2 = (p2[0] - offset, p2[1] - offset)
            pt = safe_intersect(shifted_p1, vp1, shifted_p2, vp2)
            if pt:
                return pt
        return None

    # Step 1: Extract boundary points from the mask
    points = np.argwhere(mask > 0)
    points = points[:, [1, 0]]  # Convert to (x, y)

    if len(points) < 3:
        raise ValueError(f"Not enough points in mask for a valid polygon. Found {len(points)} points.")

    hull_points = points
    mask_center = mask_center[[1, 0]]  # Convert to (x, y)

    # Step 2: Compute tangents
    hor_right_pos, hor_right_pos_line, hor_right_neg, hor_right_neg_line = find_tangent_points(mask_center, hull_points, hor_right_vp)
    vert_pos, vert_pos_line, vert_neg, vert_neg_line = find_tangent_points(mask_center, hull_points, vert_vp)
    hor_left_pos, hor_left_pos_line, hor_left_neg, hor_left_neg_line = find_tangent_points(mask_center, hull_points, hor_left_vp)

    extrem = [hor_right_neg, hor_right_pos, vert_neg, vert_pos, hor_left_neg, hor_left_pos]
    extrem_orig = extrem.copy()

    # Step 3: Compute top face first
    corner_c1 = compute_intersection(hor_right_pos_line, vert_neg_line)
    corner_b1 = compute_intersection(hor_left_neg_line, vert_pos_line)

    corner_a11 = get_intersect(hor_left_vp, corner_c1, vert_vp, mask_center)
    corner_a12 = get_intersect(hor_right_vp, corner_b1, vert_vp, mask_center)
    corner_a1 = corner_a11 if corner_a11[1] < corner_a12[1] else corner_a12

    corner_c1 = get_intersect(hor_left_vp, corner_a1, vert_vp, corner_c1)
    corner_b1 = get_intersect(hor_right_vp, corner_a1, vert_vp, corner_b1)

    center1 = get_intersect(corner_c1, corner_b1, hor_left_vp, corner_a1)
    corner_h1 = get_intersect(corner_a1, center1, hor_right_vp, corner_c1)

    # Step 4: Project bottom face from top
    corner_a = get_intersect(vert_vp, corner_a1, hor_left_vp, corner_c1)
    corner_c = get_intersect(vert_vp, corner_c1, hor_right_vp, corner_h1)
    corner_b = get_intersect(vert_vp, corner_b1, hor_right_vp, corner_a1)
    corner_h = get_intersect(vert_vp, corner_h1, hor_left_vp, corner_b1)

    if debug:
        print("Top corners: A1, B1, H1, C1")
        print("A1:", corner_a1, "B1:", corner_b1, "H1:", corner_h1, "C1:", corner_c1)
        print("Bottom corners: A, B, H, C")
        print("A:", corner_a, "B:", corner_b, "H:", corner_h, "C:", corner_c)

        plt.imshow(mask, cmap='gray')
        def plot_point(corner, color, label):
            if corner is not None and len(corner) == 2:
                plt.scatter(*corner, color=color, label=label)

        plot_point(corner_a, 'red', "corner a")
        plot_point(corner_b, 'blue', "corner b")
        plot_point(corner_c, 'green', "corner c")
        plot_point(corner_h, 'yellow', "corner h")
        plot_point(corner_a1, 'cyan', "corner a1")
        plot_point(corner_b1, 'magenta', "corner b1")
        plot_point(corner_c1, 'orange', "corner c1")
        plot_point(corner_h1, 'purple', "corner h1")

        rows, cols = np.where(mask)
        pad = 100
        if rows.size > 0 and cols.size > 0:
            plt.xlim([cols.min() - pad, cols.max() + pad])
            plt.ylim([rows.max() + pad, rows.min() - pad])

        plt.legend()
        plt.show()

    base = np.array([corner_a, corner_b, corner_h, corner_c])
    top = np.array([corner_a1, corner_b1, corner_h1, corner_c1])

    return base, top, np.array(extrem)
    ########################################################################
    # Step 1.5: Add computed 3D box corner points to the point set
    additional_points = np.array([
        corner_a, corner_b, corner_h, corner_c,
        corner_a1, corner_b1, corner_h1, corner_c1
    ], dtype=np.float32)
    
    # Keep only valid (non-None) points and round to integers
    additional_points = np.array([
        [int(round(p[0])), int(round(p[1]))]
        for p in additional_points if p is not None
    ])
    
    # Append to the points array
    points = np.vstack([points, additional_points])
    ###########################################################
    hull_points = points
    #mask_center = mask_center[[1, 0]]  # Convert to (x, y)

    # Step 2: Compute tangents from vanishing points
    #from your_module import find_tangent_points, get_intersect  # Replace with actual module
    hor_right_pos, hor_right_pos_line, hor_right_neg, hor_right_neg_line = find_tangent_points(mask_center, hull_points, hor_right_vp)
    vert_pos, vert_pos_line, vert_neg, vert_neg_line = find_tangent_points(mask_center, hull_points, vert_vp)
    hor_left_pos, hor_left_pos_line, hor_left_neg, hor_left_neg_line = find_tangent_points(mask_center, hull_points, hor_left_vp)

    extrem = [hor_right_neg, hor_right_pos, vert_neg, vert_pos, hor_left_neg, hor_left_pos]

    # Step 3: Compute corners (projected 3D bounding box base)
    corner_a = compute_intersection(hor_right_neg_line, hor_left_pos_line)
    corner_c = compute_intersection(hor_left_pos_line, vert_neg_line)
    corner_b = compute_intersection(vert_pos_line, hor_right_neg_line)
    corner_r = compute_intersection(hor_left_neg_line, vert_pos_line)

    # Step 4: Compute corner_h (opposite of corner_a)
    corner_h = safe_intersect(corner_c, hor_right_vp, corner_b, hor_left_vp)
    if corner_h is None:
        corner_h = try_intersection_with_eps(corner_c, hor_right_vp, corner_b, hor_left_vp)
        if debug and corner_h:
            print("corner_h found with epsilon perturbation.")

    # Step 5: Upper face corners
    corner_c1 = compute_intersection(hor_right_pos_line, vert_neg_line)
    corner_b1 = compute_intersection(hor_left_neg_line, vert_pos_line)

    # corner_a1 (top opposite of corner_h1)
    corner_a11 = get_intersect(hor_left_vp, corner_c1, vert_vp, corner_a)
    corner_a12 = get_intersect(hor_right_vp, corner_b1, vert_vp, corner_a)
    corner_a1 = corner_a11 if corner_a11[1] < corner_a12[1] else corner_a12

    # Refine corner_c1, corner_b1 from corner_a1
    corner_c1 = get_intersect(hor_left_vp, corner_a1, vert_vp, corner_c)
    corner_b1 = get_intersect(hor_right_vp, corner_a1, vert_vp, corner_b)

    # Compute center projections for top face
    center = get_intersect(corner_c, corner_b, corner_a, corner_h)
    center1 = get_intersect(corner_c1, corner_b1, vert_vp, center)
    corner_h1 = get_intersect(corner_a1, center1, vert_vp, corner_h)

    if debug:
        print("Base corners: A, B, H, C")
        print("A:", corner_a, "B:", corner_b, "H:", corner_h, "C:", corner_c)
        print("Top corners: A1, B1, H1, C1")
        print("A1:", corner_a1, "B1:", corner_b1, "H1:", corner_h1, "C1:", corner_c1)

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
        plot_point(corner_r, 'orange', "corner r")

        # Plot tangent lines
        draw_line_segment(hor_right_vp, hor_right_neg, 'red')
        draw_line_segment(hor_right_vp, hor_right_pos, 'red')
        draw_line_segment(vert_vp, vert_neg, 'blue')
        draw_line_segment(vert_vp, vert_pos, 'blue')
        draw_line_segment(hor_left_vp, hor_left_neg, 'green')
        draw_line_segment(hor_left_vp, hor_left_pos, 'green')
    
        # Zoom into the mask region
        rows, cols = np.where(mask)
        pad = 100
        if rows.size > 0 and cols.size > 0:
            plt.xlim([cols.min() - pad, cols.max() + pad])
            plt.ylim([rows.max() + pad, rows.min() - pad])  # Invert y-axis for correct orientation
    
        # Show legend and final plot
        plt.legend()
        plt.show()

    base = np.array([corner_a, corner_b, corner_h, corner_c])
    top = np.array([corner_a1, corner_b1, corner_h1, corner_c1])

    return base, top, np.array(extrem_orig)
    
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
    
def line_to_points(line, width=1000):
    a, b, c = line
    points = []

    # Fix x = 0 and x = width to get two points
    if b != 0:
        y0 = (-c - a * 0) / b
        y1 = (-c - a * width) / b
        points.append((0, y0))
        points.append((width, y1))
    else:
        # Vertical line (b == 0), fix y instead
        x = -c / a
        points.append((x, 0))
        points.append((x, width))
    
    return points
    
def get_second_intersection(mask, corner_r, hor_right_vp):
    # 1. Define the ray line
    ray_line = LineString([corner_r, hor_right_vp])

    # 2. Find all contour pixels (the outer boundary)
    from skimage import measure
    contours = measure.find_contours(mask, level=0.5)
    
    all_intersections = []
    for contour in contours:
        contour_line = LineString(contour[:, ::-1])  # Convert to (x, y) format
        inter = ray_line.intersection(contour_line)
        if inter.is_empty:
            continue
        if isinstance(inter, Point):
            all_intersections.append((inter.x, inter.y))
        elif hasattr(inter, '__iter__'):  # Could be MultiPoint
            for p in inter:
                if isinstance(p, Point):
                    all_intersections.append((p.x, p.y))

    # 3. Sort intersections by distance to corner_r
    if len(all_intersections) >= 2:
        all_intersections.sort(key=lambda pt: np.linalg.norm(np.array(pt) - np.array(corner_r)))
        return all_intersections[1]  # Return the second closest point
    else:
        return None

def compute_3d_box_from_plain_mask(mask, mask_center, vert_vp, hor_left_vp, hor_right_vp, debug=False):

    # Step 1: Extract non-zero points (boundary of the mask)
    points = np.argwhere(mask > 0)  # Extract all non-zero pixel coordinates (row, col)
    points = points[:, [1, 0]]  # Switch to (x, y) format for consistency

    if len(points) < 3:
        raise ValueError(f"Not enough points in mask for a valid polygon. Found {len(points)} points.")

    hull_points = points #points[hull.vertices]
    #polygon = Polygon(hull_points)  # Create a polygon using the convex hull

    # Step 3: Calculate the reference angle (line from VP to mask center)
    mask_center = mask_center[[1, 0]] # np.mean(hull_points, axis=0)  # Centroid of the convex hull
    # TODO left, right ?
    #pos_point, pos_line, neg_point, neg_line
    hor_right_pos, hor_right_pos_line, hor_right_neg, hor_right_neg_line = find_tangent_points(mask_center, hull_points, hor_right_vp)
    vert_pos, vert_pos_line, vert_neg, vert_neg_line  = find_tangent_points(mask_center, hull_points, vert_vp)
    # TODO check orientation
    hor_left_pos, hor_left_pos_line, hor_left_neg, hor_left_neg_line  = find_tangent_points(mask_center, hull_points, hor_left_vp)

    extrem = [hor_right_neg, hor_right_pos, vert_neg, vert_pos, hor_left_neg, hor_left_pos]
    # Step 2: Compute intersections for corners
    def compute_intersection(hor_right_neg_line, hor_right_pos_line):
        """Compute the intersection of two lines in homogeneous coordinates."""
        inter = np.cross(hor_right_neg_line, hor_right_pos_line)
        if inter[2] != 0:
            return inter[:2] / inter[2]
        return None  # Parallel lines

    corner_a = compute_intersection(hor_right_neg_line, hor_left_pos_line)
    corner_c = compute_intersection(hor_left_pos_line, vert_neg_line)
    corner_b = compute_intersection(vert_pos_line, hor_right_neg_line)
    
    corner_r = compute_intersection(hor_left_neg_line, vert_pos_line)
    
    p = line_to_points(hor_left_pos_line)
    #p = line_to_points(vert_neg_line)
    
    corner_t = get_intersect(np.array(p[0]), np.array(p[1]), np.array(hor_right_vp), np.array(corner_r))
    u0 = corner_t
    u1 = corner_r
    u2 = compute_intersection(hor_right_pos_line, hor_left_neg_line)
    u3 = compute_intersection(hor_right_pos_line, hor_left_pos_line)

    p = line_to_points(hor_right_neg_line)
    l0 = get_intersect(np.array(u0), np.array(vert_vp), np.array(p[0]), np.array(p[1]))
    p = line_to_points(hor_right_neg_line)
    l1 = get_intersect(np.array(u1), np.array(vert_vp), np.array(p[0]), np.array(p[1]))
    l2 = get_intersect(l1, np.array(hor_left_vp), np.array(vert_vp), u2)
    l3 = get_intersect(l2, np.array(hor_right_vp), np.array(vert_vp), u3)
    
    return np.array([l0, l1, l2, l3]), np.array([u0, u1, u2, u3]), np.array(extrem)


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
    
    
    corner_c1 = compute_intersection(hor_right_pos_line, vert_neg_line)
    corner_b1 = compute_intersection(hor_left_neg_line, vert_pos_line)

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
  
      
        plot_point(u0, 'red', "u0")
        plot_point(l0, 'red', "u0")
        plot_point(u1, 'blue', "u1")
        plot_point(l1, 'blue', "u1")
        plot_point(u2, 'green', "u2")
        plot_point(l2, 'green', "u2")
        plot_point(u3, 'yellow', "u3")
        plot_point(l3, 'yellow', "u3")
        # plot_point(corner_r, 'orange', "corner r")
        # plot_point(corner_t, 'orange', "corner t")
        
        # plot_point(corner_a1, 'red', "corner a1")
        # plot_point(corner_b1, 'blue', "corner b1")
        # plot_point(corner_c1, 'green', "corner c1")
        # plot_point(corner_h1, 'yellow', "corner h1")
        #extrem = [hor_right_neg, hor_right_pos, vert_neg, vert_pos, hor_left_neg, hor_left_pos]
        
        # plot_point(hor_right_neg, 'red', "hor_right_neg")
        # plot_point(hor_right_pos, 'blue', "hor_right_pos")
        # plot_point(vert_neg, 'green', "vert_neg")
        # plot_point(vert_pos, 'yellow', "vert_pos")
        # plot_point(hor_left_neg, 'teal', "hor_left_neg")
        # plot_point(hor_left_pos, 'orange', "hor_left_pos")
    
        # Plot tangent lines
        draw_line_segment(corner_t, corner_r, 'red')
        draw_line_segment(hor_right_vp, hor_right_neg, 'red')
        draw_line_segment(hor_right_vp, hor_right_pos, 'red')
        draw_line_segment(vert_vp, vert_neg, 'blue')
        draw_line_segment(vert_vp, vert_pos, 'blue')
        draw_line_segment(hor_left_vp, hor_left_neg, 'green')
        draw_line_segment(hor_left_vp, hor_left_pos, 'green')
        # plot_point(corner_a, 'teal', "corner_a")
        # plot_point(corner_c, 'green', "corner_c")
        # plot_point(corner_b, 'yellow', "corner_b")


    
        # Filter valid corners for visualization
        valid_corners = [corner for corner in corners if corner is not None]
        valid_corners1 = [corner for corner in corners1 if corner is not None]
    
        # Zoom into the mask region
        rows, cols = np.where(mask)
        pad = 500
        if rows.size > 0 and cols.size > 0:
            plt.xlim([cols.min() - pad, cols.max() + pad])
            plt.ylim([rows.max() + pad, rows.min() - pad])  # Invert y-axis for correct orientation
    
        # Show legend and final plot
        plt.legend()
        plt.show()
        
    # print([corner_a, corner_b, corner_h, corner_c])
    # print([corner_a1, corner_b1, corner_h1, corner_c1])

    return np.array([corner_a, corner_b, corner_h, corner_c]), np.array([corner_a1, corner_b1, corner_h1, corner_c1]), np.array(extrem)


def compute_3d_box_ellipse(mask, mask_center, vert_vp, hor_left_vp, hor_right_vp, debug=False):
    def compute_intersection(line1, line2):
        """Compute the intersection point of two lines in homogeneous coordinates."""
        inter = np.cross(line1, line2)
        if abs(inter[2]) > 1e-6:
            return inter[:2] / inter[2]
        return None

    def safe_intersect(p1, p2, p3, p4):
        """Get intersection point of two lines defined by (p1, p2) and (p3, p4)."""
        line1 = LineString([p1, p2])
        line2 = LineString([p3, p4])
        if line1.intersects(line2):
            pt = line1.intersection(line2)
            if pt.geom_type == "Point":
                return (pt.x, pt.y)
        return None

    def try_intersection_with_eps(p1, vp1, p2, vp2, eps=1e-1):
        """Try intersections with small perturbations in both directions."""
        for sign in [1, -1]:
            offset = eps * sign
            shifted_p1 = (p1[0] + offset, p1[1] + offset)
            shifted_p2 = (p2[0] - offset, p2[1] - offset)
            pt = safe_intersect(shifted_p1, vp1, shifted_p2, vp2)
            if pt:
                return pt
        return None

    # Step 1: Extract boundary points from the mask
    points = np.argwhere(mask > 0)
    points = points[:, [1, 0]]  # Convert to (x, y)

    if len(points) < 5:  # EllipseModel requires at least 5 points
        raise ValueError(f"Not enough points in mask to fit an ellipse. Found {len(points)}.")
    
    # # Step 2: Fit an ellipse to the points
    # ellipse_model = EllipseModel()
    # success = ellipse_model.estimate(points)
    
    # if not success:
        # raise RuntimeError("Ellipse fitting failed.")
    
    # xc, yc, a, b, theta = ellipse_model.params  # center (xc, yc), axes a, b, and rotation angle theta
    
    # # Step 3: Sample boundary points along the ellipse
    # num_samples = len(points)
    # angles = np.linspace(0, 2 * np.pi, num_samples)
    # cos_t, sin_t = np.cos(angles), np.sin(angles)
    
    # # Ellipse equation with rotation
    # ellipse_pts = np.column_stack((
        # xc + a * cos_t * np.cos(theta) - b * sin_t * np.sin(theta),
        # yc + a * cos_t * np.sin(theta) + b * sin_t * np.cos(theta)
    # ))
    # Compute the convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    #hull_points = ellipse_pts.astype(np.float32)
    mask_center = mask_center[[1, 0]]  # Ensure center is in (x, y) order

    # Step 2: Compute tangents from vanishing points
    #from your_module import find_tangent_points, get_intersect  # Replace with actual module
    hor_right_pos, hor_right_pos_line, hor_right_neg, hor_right_neg_line = find_tangent_points(mask_center, hull_points, hor_right_vp)
    vert_pos, vert_pos_line, vert_neg, vert_neg_line = find_tangent_points(mask_center, hull_points, vert_vp)
    hor_left_pos, hor_left_pos_line, hor_left_neg, hor_left_neg_line = find_tangent_points(mask_center, hull_points, hor_left_vp)

    extrem = [hor_right_neg, hor_right_pos, vert_neg, vert_pos, hor_left_neg, hor_left_pos]

    # Step 3: Compute corners (projected 3D bounding box base)
    corner_a = compute_intersection(hor_right_neg_line, hor_left_pos_line)
    corner_c = compute_intersection(hor_left_pos_line, vert_neg_line)
    corner_b = compute_intersection(vert_pos_line, hor_right_neg_line)
    corner_r = compute_intersection(hor_left_neg_line, vert_pos_line)

    # Step 4: Compute corner_h (opposite of corner_a)
    corner_h = safe_intersect(corner_c, hor_right_vp, corner_b, hor_left_vp)
    if corner_h is None:
        corner_h = try_intersection_with_eps(corner_c, hor_right_vp, corner_b, hor_left_vp)
        if debug and corner_h:
            print("corner_h found with epsilon perturbation.")

    # Step 5: Upper face corners
    corner_c1 = compute_intersection(hor_right_pos_line, vert_neg_line)
    corner_b1 = compute_intersection(hor_left_neg_line, vert_pos_line)

    # corner_a1 (top opposite of corner_h1)
    corner_a11 = get_intersect(hor_left_vp, corner_c1, vert_vp, corner_a)
    corner_a12 = get_intersect(hor_right_vp, corner_b1, vert_vp, corner_a)
    corner_a1 = corner_a11 if corner_a11[1] < corner_a12[1] else corner_a12

    # Refine corner_c1, corner_b1 from corner_a1
    corner_c1 = get_intersect(hor_left_vp, corner_a1, vert_vp, corner_c)
    corner_b1 = get_intersect(hor_right_vp, corner_a1, vert_vp, corner_b)

    # Compute center projections for top face
    center = get_intersect(corner_c, corner_b, corner_a, corner_h)
    center1 = get_intersect(corner_c1, corner_b1, vert_vp, center)
    corner_h1 = get_intersect(corner_a1, center1, vert_vp, corner_h)

    if debug:
        print("Base corners: A, B, H, C")
        print("A:", corner_a, "B:", corner_b, "H:", corner_h, "C:", corner_c)
        print("Top corners: A1, B1, H1, C1")
        print("A1:", corner_a1, "B1:", corner_b1, "H1:", corner_h1, "C1:", corner_c1)

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
        plot_point(corner_r, 'orange', "corner r")

        # Plot tangent lines
        draw_line_segment(hor_right_vp, hor_right_neg, 'red')
        draw_line_segment(hor_right_vp, hor_right_pos, 'red')
        draw_line_segment(vert_vp, vert_neg, 'blue')
        draw_line_segment(vert_vp, vert_pos, 'blue')
        draw_line_segment(hor_left_vp, hor_left_neg, 'green')
        draw_line_segment(hor_left_vp, hor_left_pos, 'green')
    
        # Zoom into the mask region
        rows, cols = np.where(mask)
        pad = 100
        if rows.size > 0 and cols.size > 0:
            plt.xlim([cols.min() - pad, cols.max() + pad])
            plt.ylim([rows.max() + pad, rows.min() - pad])  # Invert y-axis for correct orientation
    
        # Show legend and final plot
        plt.legend()
        plt.show()

    base = np.array([corner_a, corner_b, corner_h, corner_c])
    top = np.array([corner_a1, corner_b1, corner_h1, corner_c1])
    return base, top, np.array(extrem)
    
def compute_3d_box_projection(mask, mask_center, vert_vp, hor_left_vp, hor_right_vp, debug=False):
    def compute_intersection(line1, line2):
        """Compute the intersection point of two lines in homogeneous coordinates."""
        inter = np.cross(line1, line2)
        if abs(inter[2]) > 1e-6:
            return inter[:2] / inter[2]
        return None

    def safe_intersect(p1, p2, p3, p4):
        """Get intersection point of two lines defined by (p1, p2) and (p3, p4)."""
        line1 = LineString([p1, p2])
        line2 = LineString([p3, p4])
        if line1.intersects(line2):
            pt = line1.intersection(line2)
            if pt.geom_type == "Point":
                return (pt.x, pt.y)
        return None

    def try_intersection_with_eps(p1, vp1, p2, vp2, eps=1e-1):
        """Try intersections with small perturbations in both directions."""
        for sign in [1, -1]:
            offset = eps * sign
            shifted_p1 = (p1[0] + offset, p1[1] + offset)
            shifted_p2 = (p2[0] - offset, p2[1] - offset)
            pt = safe_intersect(shifted_p1, vp1, shifted_p2, vp2)
            if pt:
                return pt
        return None

    # Step 1: Extract boundary points from the mask
    points = np.argwhere(mask > 0)
    points = points[:, [1, 0]]  # Convert to (x, y)

    if len(points) < 3:
        raise ValueError(f"Not enough points in mask for a valid polygon. Found {len(points)} points.")

    hull_points = points
    mask_center = mask_center[[1, 0]]  # Convert to (x, y)

    # Step 2: Compute tangents from vanishing points
    #from your_module import find_tangent_points, get_intersect  # Replace with actual module
    hor_right_pos, hor_right_pos_line, hor_right_neg, hor_right_neg_line = find_tangent_points(mask_center, hull_points, hor_right_vp)
    vert_pos, vert_pos_line, vert_neg, vert_neg_line = find_tangent_points(mask_center, hull_points, vert_vp)
    hor_left_pos, hor_left_pos_line, hor_left_neg, hor_left_neg_line = find_tangent_points(mask_center, hull_points, hor_left_vp)

    extrem = [hor_right_neg, hor_right_pos, vert_neg, vert_pos, hor_left_neg, hor_left_pos]

    # Step 3: Compute corners (projected 3D bounding box base)
    corner_a = compute_intersection(hor_right_neg_line, hor_left_pos_line)
    corner_c = compute_intersection(hor_left_pos_line, vert_neg_line)
    corner_b = compute_intersection(vert_pos_line, hor_right_neg_line)
    corner_r = compute_intersection(hor_left_neg_line, vert_pos_line)

    # Step 4: Compute corner_h (opposite of corner_a)
    corner_h = safe_intersect(corner_c, hor_right_vp, corner_b, hor_left_vp)
    if corner_h is None:
        corner_h = try_intersection_with_eps(corner_c, hor_right_vp, corner_b, hor_left_vp)
        if debug and corner_h:
            print("corner_h found with epsilon perturbation.")

    # Step 5: Upper face corners
    corner_c1 = compute_intersection(hor_right_pos_line, vert_neg_line)
    corner_b1 = compute_intersection(hor_left_neg_line, vert_pos_line)

    # corner_a1 (top opposite of corner_h1)
    corner_a11 = get_intersect(hor_left_vp, corner_c1, vert_vp, corner_a)
    corner_a12 = get_intersect(hor_right_vp, corner_b1, vert_vp, corner_a)
    corner_a1 = corner_a11 if corner_a11[1] < corner_a12[1] else corner_a12

    # Refine corner_c1, corner_b1 from corner_a1
    corner_c1 = get_intersect(hor_left_vp, corner_a1, vert_vp, corner_c)
    corner_b1 = get_intersect(hor_right_vp, corner_a1, vert_vp, corner_b)

    # Compute center projections for top face
    center = get_intersect(corner_c, corner_b, corner_a, corner_h)
    center1 = get_intersect(corner_c1, corner_b1, vert_vp, center)
    corner_h1 = get_intersect(corner_a1, center1, vert_vp, corner_h)

    if debug:
        print("Base corners: A, B, H, C")
        print("A:", corner_a, "B:", corner_b, "H:", corner_h, "C:", corner_c)
        print("Top corners: A1, B1, H1, C1")
        print("A1:", corner_a1, "B1:", corner_b1, "H1:", corner_h1, "C1:", corner_c1)

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
        plot_point(corner_r, 'orange', "corner r")

        # Plot tangent lines
        draw_line_segment(hor_right_vp, hor_right_neg, 'red')
        draw_line_segment(hor_right_vp, hor_right_pos, 'red')
        draw_line_segment(vert_vp, vert_neg, 'blue')
        draw_line_segment(vert_vp, vert_pos, 'blue')
        draw_line_segment(hor_left_vp, hor_left_neg, 'green')
        draw_line_segment(hor_left_vp, hor_left_pos, 'green')
    
        # Zoom into the mask region
        rows, cols = np.where(mask)
        pad = 100
        if rows.size > 0 and cols.size > 0:
            plt.xlim([cols.min() - pad, cols.max() + pad])
            plt.ylim([rows.max() + pad, rows.min() - pad])  # Invert y-axis for correct orientation
    
        # Show legend and final plot
        plt.legend()
        plt.show()

    base = np.array([corner_a, corner_b, corner_h, corner_c])
    top = np.array([corner_a1, corner_b1, corner_h1, corner_c1])
    return base, top, np.array(extrem)

def compute_3d_box_from_plain_mask_orig(mask, mask_center, vert_vp, hor_left_vp, hor_right_vp, debug=False):

    # Step 1: Extract non-zero points (boundary of the mask)
    points = np.argwhere(mask > 0)  # Extract all non-zero pixel coordinates (row, col)
    points = points[:, [1, 0]]  # Switch to (x, y) format for consistency

    if len(points) < 3:
        raise ValueError(f"Not enough points in mask for a valid polygon. Found {len(points)} points.")

    hull_points = points #points[hull.vertices]
    #polygon = Polygon(hull_points)  # Create a polygon using the convex hull

    # Step 3: Calculate the reference angle (line from VP to mask center)
    mask_center = mask_center[[1, 0]] # np.mean(hull_points, axis=0)  # Centroid of the convex hull
    # TODO left, right ?
    #pos_point, pos_line, neg_point, neg_line
    hor_right_pos, hor_right_pos_line, hor_right_neg, hor_right_neg_line = find_tangent_points(mask_center, hull_points, hor_right_vp)
    vert_pos, vert_pos_line, vert_neg, vert_neg_line  = find_tangent_points(mask_center, hull_points, vert_vp)
    # TODO check orientation
    hor_left_pos, hor_left_pos_line, hor_left_neg, hor_left_neg_line  = find_tangent_points(mask_center, hull_points, hor_left_vp)

    extrem = [hor_right_neg, hor_right_pos, vert_neg, vert_pos, hor_left_neg, hor_left_pos]
    # Step 2: Compute intersections for corners
    def compute_intersection(hor_right_neg_line, hor_right_pos_line):
        """Compute the intersection of two lines in homogeneous coordinates."""
        inter = np.cross(hor_right_neg_line, hor_right_pos_line)
        if inter[2] != 0:
            return inter[:2] / inter[2]
        return None  # Parallel lines

    corner_a = compute_intersection(hor_right_neg_line, hor_left_pos_line)
    corner_c = compute_intersection(hor_left_pos_line, vert_neg_line)
    corner_b = compute_intersection(vert_pos_line, hor_right_neg_line)
    
    corner_r = compute_intersection(hor_left_neg_line, vert_pos_line)

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

    epsilon = -1e-1  # Small shift value
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
    
    
    corner_c1 = compute_intersection(hor_right_pos_line, vert_neg_line)
    corner_b1 = compute_intersection(hor_left_neg_line, vert_pos_line)

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
        plot_point(corner_r, 'orange', "corner r")
        
        # plot_point(corner_a1, 'red', "corner a1")
        # plot_point(corner_b1, 'blue', "corner b1")
        # plot_point(corner_c1, 'green', "corner c1")
        # plot_point(corner_h1, 'yellow', "corner h1")
        #extrem = [hor_right_neg, hor_right_pos, vert_neg, vert_pos, hor_left_neg, hor_left_pos]
        
        # plot_point(hor_right_neg, 'red', "hor_right_neg")
        # plot_point(hor_right_pos, 'blue', "hor_right_pos")
        # plot_point(vert_neg, 'green', "vert_neg")
        # plot_point(vert_pos, 'yellow', "vert_pos")
        # plot_point(hor_left_neg, 'teal', "hor_left_neg")
        # plot_point(hor_left_pos, 'orange', "hor_left_pos")
    
        # Plot tangent lines
        draw_line_segment(hor_right_vp, hor_right_neg, 'red')
        draw_line_segment(hor_right_vp, hor_right_pos, 'red')
        draw_line_segment(vert_vp, vert_neg, 'blue')
        draw_line_segment(vert_vp, vert_pos, 'blue')
        draw_line_segment(hor_left_vp, hor_left_neg, 'green')
        draw_line_segment(hor_left_vp, hor_left_pos, 'green')
        # plot_point(corner_a, 'teal', "corner_a")
        # plot_point(corner_c, 'green', "corner_c")
        # plot_point(corner_b, 'yellow', "corner_b")


    
        # Filter valid corners for visualization
        valid_corners = [corner for corner in corners if corner is not None]
        valid_corners1 = [corner for corner in corners1 if corner is not None]
    
        # Zoom into the mask region
        rows, cols = np.where(mask)
        pad = 100
        if rows.size > 0 and cols.size > 0:
            plt.xlim([cols.min() - pad, cols.max() + pad])
            plt.ylim([rows.max() + pad, rows.min() - pad])  # Invert y-axis for correct orientation
    
        # Show legend and final plot
        plt.legend()
        plt.show()
        
    # print([corner_a, corner_b, corner_h, corner_c])
    # print([corner_a1, corner_b1, corner_h1, corner_c1])

    return np.array([corner_a, corner_b, corner_h, corner_c]), np.array([corner_a1, corner_b1, corner_h1, corner_c1]), np.array(extrem)



def compute_3d_box_from_plain_mask_old(mask, vert_vp, hor_left_vp, hor_right_vp, debug=False):

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
    point_b, hor_right_pos_line, point_a, hor_right_neg_line = find_tangent_points(mask_center, hull_points, hor_right_vp)
    vert_pos, vert_pos_line, vert_neg, vert_neg_line  = find_tangent_points(mask_center, hull_points, vert_vp)
    # TODO check orientation
    hor_left_pos, hor_left_pos_line, hor_left_neg, hor_left_neg_line  = find_tangent_points(mask_center, hull_points, hor_left_vp)

    extrem = [point_a, point_b, vert_neg, vert_pos, hor_left_neg, hor_left_pos]
    # Step 2: Compute intersections for corners
    def compute_intersection(hor_right_neg_line, hor_right_pos_line):
        """Compute the intersection of two lines in homogeneous coordinates."""
        inter = np.cross(hor_right_neg_line, hor_right_pos_line)
        if inter[2] != 0:
            return inter[:2] / inter[2]
        return None  # Parallel lines

    corner_a = compute_intersection(hor_right_neg_line, hor_left_pos_line)
    corner_c = compute_intersection(hor_left_pos_line, vert_neg_line)
    corner_b = compute_intersection(vert_pos_line, hor_right_neg_line)
    
    
    
    
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
    
    
    corner_c1 = compute_intersection(hor_right_pos_line, vert_neg_line)
    corner_b1 = compute_intersection(hor_left_neg_line, vert_pos_line)

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
        draw_line_segment(vert_vp, vert_neg, 'blue')
        draw_line_segment(vert_vp, vert_pos, 'blue')
        draw_line_segment(hor_left_vp, hor_left_neg, 'green')
        draw_line_segment(hor_left_vp, hor_left_pos, 'green')
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


def calc_height(lower_face, upper_face, bottom_face, inv_ipm_matrix, top_inv_ipm_matrix, object_height):
    proj_height = np.linalg.norm(lower_face[0] - upper_face[0])  # height
    proj_bottom = map_points_to_BEV([bottom_face[0]], inv_ipm_matrix)
    proj_top = map_points_to_BEV([proj_bottom[0]], top_inv_ipm_matrix)

    #dist = np.linalg.norm(camera_position_bev - bottom_face[0])
    full_height = np.linalg.norm(proj_bottom[0] - proj_top[0]) # object_height

    height = object_height * proj_height / full_height

    return height


def clip_cube_to_bounds(avg_bottom, avg_top, original_cube):
    """
    Clips avg_bottom and avg_top points to stay within min and max bounds
    defined by the original cube_3d's bottom[0] and top[2] points.

    Parameters:
        avg_bottom (np.ndarray): (4, 3) array for bottom face of averaged cube.
        avg_top (np.ndarray): (4, 3) array for top face of averaged cube.
        original_cube (dict): Original cube_3d with "bottom" and "top" keys.

    Returns:
        dict: Clipped cube_3d faces using get_projected_cube_faces.
    """
    min_bounds = original_cube["bottom"][0]
    max_bounds = original_cube["top"][2]

    clipped_bottom = np.clip(avg_bottom, min_bounds, max_bounds)
    clipped_top = np.clip(avg_top, min_bounds, max_bounds)

    return get_projected_cube_faces(clipped_bottom, clipped_top)
