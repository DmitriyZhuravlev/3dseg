import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu, threshold_local
from skimage.segmentation import slic, mark_boundaries
from scipy.spatial import Delaunay
from matplotlib.lines import Line2D
from skimage import img_as_float
from skimage.transform import ProjectiveTransform
from skimage.measure import find_contours

from skimage.segmentation import slic
from skimage.transform import ProjectiveTransform, warp
from skimage.util import img_as_float

from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.filters import rank
from skimage.util import img_as_ubyte

from skimage import io, color, segmentation
from scipy import ndimage as ndi

from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_local, threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects, binary_closing, disk, opening
import numpy as np


import sys
import os


from graph import * #build_graph, dfs, bfs, dijkstra, default_weight
from plot import * #plot_mask, plot_centers_and_edges, plot_graph_with_avg_weights
from segment import *



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


# Optional: Your own weight function
def weight_by_euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def dist0(points1, points2):
    return np.linalg.norm(np.array(points1[0]) - np.array(points2[0]))


# def compute_foreground_mask(img, ref, method="otsu", threshold_value=0.5):
    # # Convert to grayscale
    # gray_img = rgb2gray(img)
    # gray_ref = rgb2gray(ref)

    # # Absolute difference
    # abs_diff = np.abs(gray_img - gray_ref)
    # abs_diff_blur = gaussian(abs_diff, sigma=35.0)

    # # Thresholding
    # if method == "adaptive":
        # block_size = 35
        # local_thresh = threshold_local(abs_diff_blur, block_size)
        # object_mask = abs_diff_blur > local_thresh
    # elif method == "otsu":
        # thresh_val = threshold_otsu(abs_diff_blur)
        # object_mask = abs_diff_blur > thresh_val
    # else:
        # object_mask = abs_diff_blur > threshold_value

    # return object_mask

def compute_foreground_mask(img, ref, method="otsu", threshold_value=0.5,
                            blur_sigma=15.0, morph_radius=5,
                            min_hole_area=1000, min_object_size=500,
                            shadow_thresh_ratio=0.5):
    """
    Computes a clean foreground mask by comparing img and ref.
    Removes shadows and fills holes using intensity-based segmentation.
    """
    # Convert to grayscale
    gray_img = rgb2gray(img)
    gray_ref = rgb2gray(ref)

    # Absolute difference
    abs_diff = np.abs(gray_img - gray_ref)
    abs_diff_blur = gaussian(abs_diff, sigma=blur_sigma)

    # Initial thresholding
    if method == "adaptive":
        local_thresh = threshold_local(abs_diff_blur, block_size=35)
        object_mask = abs_diff_blur > local_thresh
    elif method == "otsu":
        thresh_val = threshold_otsu(abs_diff_blur)
        object_mask = abs_diff_blur > thresh_val
    else:
        object_mask = abs_diff_blur > threshold_value

    # Morphological filtering
    selem = disk(morph_radius)
    object_mask = opening(object_mask, selem)                 # remove fine structures (shadows)
    object_mask = binary_closing(object_mask, selem)          # fill gaps/holes
    object_mask = remove_small_holes(object_mask, min_hole_area)

    # Shadow suppression via segmentation inside mask
    masked_diff = abs_diff_blur * object_mask
    nonzero_vals = masked_diff[masked_diff > 0]

    if len(nonzero_vals) > 0:
        mean_val = np.mean(nonzero_vals)
        shadow_mask = (masked_diff < shadow_thresh_ratio * mean_val) & object_mask
        object_mask = object_mask & ~shadow_mask  # remove shadow pixels

    return object_mask

def load_images():
    if len(sys.argv) > 2 and os.path.exists(sys.argv[1]) and os.path.exists(sys.argv[2]):
        img = imread(sys.argv[1])
        ref = imread(sys.argv[2])
        print(f"Loaded image: {sys.argv[1]}")
        print(f"Loaded reference (background): {sys.argv[2]}")
    else:
        print("No valid image path provided. Using default  image.")
        img = imread("/home/dzhura/ComputerVision/data/img/test/out/pig.JPG").astype(np.float32) / 255.
        ref = imread("/home/dzhura/ComputerVision/data/img/reference.JPG").astype(np.float32) / 255.

    return img_as_float(img), img_as_float(ref)


cube = False #True
pig = True #False

if pig:
    g_xmin = xmin = 0
    g_xmax = xmax = 2100
    g_ymin = ymin = 0
    g_object_width = g_ymax = ymax = 700
    g_zmin = zmin = 0
    g_zmax = zmax = 700
elif cube:
    g_xmin = xmin = 0
    g_xmax = xmax = 21
    g_ymin = ymin = 0
    g_object_width = g_ymax = ymax = 21
    g_zmin = zmin = 0
    g_zmax = zmax = 21
    
object_length = xmax - xmin
object_width = ymax - ymin
object_height = zmax - zmin


debug = True



def segment_mask(img, foreground_mask, info, curr_level, max_level = 2, n_segments = 3, debug = False, internal = False, clip = False):
    if curr_level > max_level:
        return [info]
    
    main_info = info.copy()
    cube_2d = info.cube_2d
    cube_3d = info.cube_3d
    extrem = info.extrem
    # Run SLIC only on masked foreground
    segments = slic(img, n_segments=n_segments, compactness=100, start_label=1, mask=foreground_mask)
    segment_ids = np.unique(segments)
    


    centers = []
    valid_ids = []
    
    for i in segment_ids:
        coords = np.column_stack(np.nonzero(segments == i))
        mean_yx = np.mean(coords, axis=0)
        y, x = map(int, mean_yx)
    
        # Check if center lies in the foreground mask
        if 0 <= y < foreground_mask.shape[0] and 0 <= x < foreground_mask.shape[1]:
            if foreground_mask[y, x]:  # Only include if inside the mask
                centers.append(mean_yx)
                valid_ids.append(i)

    if debug:
        plot_centers_and_edges(img, segments, np.array(centers))

    segment_infos = build_segment_info_list(segments, foreground_mask, vert_vp, hor_left_vp, hor_right_vp, debug=debug)
    
    # # Build graph
    # graph = build_segment_info_graph(
        # segment_infos, 
        # foreground_mask, 
        # vert_vp, 
        # hor_left_vp, 
        # hor_right_vp, 
        # weight_func=weight_by_euclidean,
        # debug=debug
    # )
    
    # if graph is None:
        # return  [info]

    # if debug:
        # draw_segment_graph(segment_infos, segments, background_image=img)
        # #cube_img = draw_all_segment_cubes_matplotlib(segment_infos, background_image=img)

    hor_right_neg, hor_right_pos, vert_neg, vert_pos, hor_left_neg, hor_left_pos = extrem
    initial_internal_segments = []
    initial_external_segments = []
    used_labels = set()
    
    if not internal:
        #vert_neg
        segment_label = find_segment_for_point(segments, vert_neg)
        used_labels.add(segment_label)
        mask = (segments == segment_label)  # Get all pixels belonging to the segment
    
        # if debug:
            # highlite_segment(img, mask, segments, segment_label)
    
        segment_info = get_segment_by_id(segment_infos, segment_label)
        if segment_info is not None:
            center_yx = segment_info.center[[1, 0]] #np.array(segment_info.center[1], segment_info.center[0])
            lower_face, upper_face, extrem = get_projected_box(mask.astype(np.uint8), center_yx, vert_vp, hor_left_vp, hor_right_vp, internal = False, debug=False)
            segment_info.cube_2d = get_projected_cube_faces(lower_face, upper_face)
            segment_info.extrem = extrem.copy()

            nb_cube_2d = segment_info.cube_2d
            #left
            face1 = "left"
            face2 = "right"
            
            pts1 = np.float32(cube_2d[face1])
            arr = cube_3d[face1].copy()
            arr = arr[:, 1:]
            arr[:, [0, 1]] = arr[:, [1, 0]]  # swap y-z -> z-y
            pts2 = np.float32(arr)
            ipm = get_projective_transform(pts1, pts2)
        
            dist_2d = dist0(cube_2d[face1], cube_2d[face2])
            dist_3d = dist0(cube_3d[face1], cube_3d[face2])
            nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
            if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
            nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
        
            mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
            mapped = mapped[:, [1, 0]]  # swap z-y
        
        
            l_3d = np.column_stack((np.full(mapped.shape[0], cube_3d[face1][0, 0]), mapped))
            r_3d = l_3d.copy()
            r_3d[:, 0] += nb_dist_3d
            
            upper_3d = np.array([l_3d[1], r_3d[1], r_3d[2], l_3d[2]])
            lower_3d = np.array([l_3d[0], r_3d[0], r_3d[3], l_3d[3]])
            
            initial_external_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))
            
            #forfard
            face1 = "forward"
            face2 = "backward"
            
            pts1 = np.float32(cube_2d[face1])
            arr = cube_3d[face1].copy()
            arr = arr[:, [0, 2]]  # remove Y 
            arr[:, [0, 1]] = arr[:, [1, 0]]  # swap x-z -> z-x
            pts2 = np.float32(arr)
            ipm = get_projective_transform(pts1, pts2)
            
            dist_2d = dist0(cube_2d[face1], cube_2d[face2])
            dist_3d = dist0(cube_3d[face1], cube_3d[face2])
            nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
            if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
            nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
        
            mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
            mapped = mapped[:, [1, 0]]  # swap z-x
            
            f_3d = np.column_stack((
                mapped[:, 0],                                   # x
                np.full(mapped.shape[0], cube_3d[face1][0, 1]), # FIX y
                mapped[:, 1]                                    # Z
            ))
            b_3d = f_3d.copy()
            b_3d[:, 1] -= nb_dist_3d
        
        
            upper_3d = np.array([b_3d[1], b_3d[2], f_3d[2], f_3d[1]])
            lower_3d = np.array([b_3d[0], b_3d[3], f_3d[3], f_3d[0]])
        
            
            initial_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))

    
    # hor_right_neg
    segment_label = find_segment_for_point(segments, hor_right_neg)
    if segment_label not in used_labels:
        used_labels.add(segment_label)
        mask = (segments == segment_label)  # Get all pixels belonging to the segment
    
        # if debug:
            # highlite_segment(img, mask, segments, segment_label)
    
        segment_info = get_segment_by_id(segment_infos, segment_label)
        if segment_info is not None:
            center_yx = segment_info.center[[1, 0]] #np.array(segment_info.center[1], segment_info.center[0])
            lower_face, upper_face, extrem = get_projected_box(mask.astype(np.uint8), center_yx, vert_vp, hor_left_vp, hor_right_vp, internal = False, debug=False)
            segment_info.cube_2d = get_projected_cube_faces(lower_face, upper_face)
            segment_info.extrem = extrem.copy()
            nb_cube_2d = segment_info.cube_2d
        
            # bottom
            face1 = "bottom"
            face2 = "top"
        
            pts1 = np.float32(cube_2d[face1])
            arr = cube_3d[face1].copy()[:, :2]
            pts2 = np.float32(arr)
            ipm = get_projective_transform(pts1, pts2)
        
            dist_2d = dist0(cube_2d[face1], cube_2d[face2])
            dist_3d = dist0(cube_3d[face1], cube_3d[face2])
            nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
            if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
            
            print(f"dist_2d: {dist_2d}")
            print(f"dist_3d: {dist_3d}")
            print(f"nb_dist_2d: {nb_dist_2d}")
            
            nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
        
            mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
        
            lower_3d = np.column_stack((mapped.copy(), np.full(mapped.shape[0], cube_3d[face1][0, 2])))
            upper_3d = lower_3d.copy()
            upper_3d[:, 2] += nb_dist_3d
        
            
            initial_internal_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))


        # backward
        face1 = "backward"
        face2 = "forward"
    
    
        pts1 = np.float32(cube_2d[face1])
        arr = cube_3d[face1].copy()
        arr = arr[:, [0, 2]]  # remove Y 
        arr[:, [0, 1]] = arr[:, [1, 0]]  # swap x-z -> z-x
        pts2 = np.float32(arr)
        ipm = get_projective_transform(pts1, pts2)
        
        dist_2d = dist0(cube_2d[face1], cube_2d[face2])
        dist_3d = dist0(cube_3d[face1], cube_3d[face2])
        nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
        if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
        nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
    
        mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
        mapped = mapped[:, [1, 0]]  # swap z-x
        
        b_3d = np.column_stack((
            mapped[:, 0],                                   # x
            np.full(mapped.shape[0], cube_3d[face1][0, 1]), # FIX y
            mapped[:, 1]                                    # Z
        ))
        f_3d = b_3d.copy()
        f_3d[:, 1] += nb_dist_3d
    
    
        upper_3d = np.array([b_3d[1], b_3d[2], f_3d[2], f_3d[1]])
        lower_3d = np.array([b_3d[0], b_3d[3], f_3d[3], f_3d[0]])
    
        
        initial_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))

    # hor_right_pos
    segment_label = find_segment_for_point(segments, hor_right_pos)
    if segment_label not in used_labels:
        used_labels.add(segment_label)
        mask = (segments == segment_label)  # Get all pixels belonging to the segment
    
        if debug:
            highlite_segment(img, mask, segments, segment_label)
    
        segment_info = get_segment_by_id(segment_infos, segment_label)
        if segment_info is not None:
            center_yx = segment_info.center[[1, 0]] #np.array(segment_info.center[1], segment_info.center[0])
            lower_face, upper_face, extrem = get_projected_box(mask.astype(np.uint8), center_yx, vert_vp, hor_left_vp, hor_right_vp, internal = False, debug=False)
            segment_info.cube_2d = get_projected_cube_faces(lower_face, upper_face)
            segment_info.extrem = extrem.copy()
            
            nb_cube_2d = segment_info.cube_2d
        
            #top
            face1 = "top"
            face2 = "bottom"
        
            pts1 = np.float32(cube_2d[face1])
            arr = cube_3d[face1].copy()[:, :2]
            pts2 = np.float32(arr)
            ipm = get_projective_transform(pts1, pts2)
        
            dist_2d = dist0(cube_2d[face1], cube_2d[face2])
            dist_3d = dist0(cube_3d[face1], cube_3d[face2])
            nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
            if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
            nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
        
            mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
        
            upper_3d = np.column_stack((mapped.copy(), np.full(mapped.shape[0], cube_3d[face1][0, 2])))
            lower_3d = upper_3d.copy()
            lower_3d[:, 2] -= nb_dist_3d
            
            initial_internal_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))
        

        #forward
        face1 = "forward"
        face2 = "backward"
    
        pts1 = np.float32(cube_2d[face1])
        arr = cube_3d[face1].copy()
        arr = arr[:, [0, 2]]  # remove Y 
        arr[:, [0, 1]] = arr[:, [1, 0]]  # swap x-z -> z-x
        pts2 = np.float32(arr)
        ipm = get_projective_transform(pts1, pts2)
        
        dist_2d = dist0(cube_2d[face1], cube_2d[face2])
        dist_3d = dist0(cube_3d[face1], cube_3d[face2])
        nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
        if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
        nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
    
        mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
        mapped = mapped[:, [1, 0]]  # swap z-x
        
        f_3d = np.column_stack((
            mapped[:, 0],                                   # x
            np.full(mapped.shape[0], cube_3d[face1][0, 1]), # FIX y
            mapped[:, 1]                                    # Z
        ))
        b_3d = f_3d.copy()
        b_3d[:, 1] -= nb_dist_3d
    
    
        upper_3d = np.array([b_3d[1], b_3d[2], f_3d[2], f_3d[1]])
        lower_3d = np.array([b_3d[0], b_3d[3], f_3d[3], f_3d[0]])
    
        
        initial_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))




    # vert_pos
    segment_label = find_segment_for_point(segments, vert_pos)
    if segment_label not in used_labels:
        used_labels.add(segment_label)
        mask = (segments == segment_label)  # Get all pixels belonging to the segment
    
        if debug:
            highlite_segment(img, mask, segments, segment_label)
    
        segment_info = get_segment_by_id(segment_infos, segment_label)
        if segment_info is not None:
            center_yx = segment_info.center[[1, 0]] #np.array(segment_info.center[1], segment_info.center[0])
            lower_face, upper_face, extrem = get_projected_box(mask.astype(np.uint8), center_yx, vert_vp, hor_left_vp, hor_right_vp, internal = False, debug=False)
            segment_info.cube_2d = get_projected_cube_faces(lower_face, upper_face)
            segment_info.extrem = extrem.copy()
            nb_cube_2d = segment_info.cube_2d
            #left
            face1 = "right"
            face2 = "left"
        
            pts1 = np.float32(cube_2d[face1])
            arr = cube_3d[face1].copy()
            arr = arr[:, 1:]
            arr[:, [0, 1]] = arr[:, [1, 0]]  # swap y-z -> z-y
            pts2 = np.float32(arr)
            ipm = get_projective_transform(pts1, pts2)
        
            dist_2d = dist0(cube_2d[face1], cube_2d[face2])
            dist_3d = dist0(cube_3d[face1], cube_3d[face2])
            nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
            #if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
            nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
        
            mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
            mapped = mapped[:, [1, 0]]  # swap z-y
        
        
            r_3d = np.column_stack((np.full(mapped.shape[0], cube_3d[face1][0, 0]), mapped))
            l_3d = r_3d.copy()
            l_3d[:, 0] -= nb_dist_3d
            
            upper_3d = np.array([l_3d[1], r_3d[1], r_3d[2], l_3d[2]])
            lower_3d = np.array([l_3d[0], r_3d[0], r_3d[3], l_3d[3]])
            
            initial_internal_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))


    if not internal:
        # hor_left_neg
        segment_label = find_segment_for_point(segments, hor_left_neg)
        mask = (segments == segment_label)  # Get all pixels belonging to the segment
    
        if debug:
            highlite_segment(img, mask, segments, segment_label)
    
        segment_info = get_segment_by_id(segment_infos, segment_label)
        if segment_info is not None:
            center_yx = segment_info.center[[1, 0]] #np.array(segment_info.center[1], segment_info.center[0])
            lower_face, upper_face, extrem = get_projected_box(mask.astype(np.uint8), center_yx, vert_vp, hor_left_vp, hor_right_vp, internal = False, debug=False)
            segment_info.cube_2d = get_projected_cube_faces(lower_face, upper_face)
            segment_info.extrem = extrem.copy()
            nb_cube_2d = segment_info.cube_2d
            #left
            face1 = "right"
            face2 = "left"
        
            pts1 = np.float32(cube_2d[face1])
            arr = cube_3d[face1].copy()
            arr = arr[:, 1:]
            arr[:, [0, 1]] = arr[:, [1, 0]]  # swap y-z -> z-y
            pts2 = np.float32(arr)
            ipm = get_projective_transform(pts1, pts2)
        
            dist_2d = dist0(cube_2d[face1], cube_2d[face2])
            dist_3d = dist0(cube_3d[face1], cube_3d[face2])
            nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
            if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
            nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
        
            mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
            mapped = mapped[:, [1, 0]]  # swap z-y
        
        
            r_3d = np.column_stack((np.full(mapped.shape[0], cube_3d[face1][0, 0]), mapped))
            l_3d = r_3d.copy()
            l_3d[:, 0] -= nb_dist_3d
            
            upper_3d = np.array([l_3d[1], r_3d[1], r_3d[2], l_3d[2]])
            lower_3d = np.array([l_3d[0], r_3d[0], r_3d[3], l_3d[3]])
            
            initial_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))

    if not internal:
        # hor_left_pos
        segment_label = find_segment_for_point(segments, hor_left_pos)
        mask = (segments == segment_label)  # Get all pixels belonging to the segment
    
        if debug:
            highlite_segment(img, mask, segments, segment_label)
    
        segment_info = get_segment_by_id(segment_infos, segment_label)
        if segment_info is not None:
            center_yx = segment_info.center[[1, 0]] #np.array(segment_info.center[1], segment_info.center[0])
            lower_face, upper_face, extrem = get_projected_box(mask.astype(np.uint8), center_yx, vert_vp, hor_left_vp, hor_right_vp, internal = False, debug=False)
            segment_info.cube_2d = get_projected_cube_faces(lower_face, upper_face)
            segment_info.extrem = extrem.copy()
            nb_cube_2d = segment_info.cube_2d
            #left
            face1 = "left"
            face2 = "right"
            
            pts1 = np.float32(cube_2d[face1])
            arr = cube_3d[face1].copy()
            arr = arr[:, 1:]
            arr[:, [0, 1]] = arr[:, [1, 0]]  # swap y-z -> z-y
            pts2 = np.float32(arr)
            ipm = get_projective_transform(pts1, pts2)
        
            dist_2d = dist0(cube_2d[face1], cube_2d[face2])
            dist_3d = dist0(cube_3d[face1], cube_3d[face2])
            nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
            if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
            nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
        
            mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
            mapped = mapped[:, [1, 0]]  # swap z-y
        
        
            l_3d = np.column_stack((np.full(mapped.shape[0], cube_3d[face1][0, 0]), mapped))
            r_3d = l_3d.copy()
            r_3d[:, 0] += nb_dist_3d
            
            upper_3d = np.array([l_3d[1], r_3d[1], r_3d[2], l_3d[2]])
            lower_3d = np.array([l_3d[0], r_3d[0], r_3d[3], l_3d[3]])
            
            initial_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))


     # Dictionary to collect lower and upper cubes per segment
    segment_bottoms = defaultdict(list)
    segment_tops = defaultdict(list)

    lower = info.cube_3d["bottom"]
    upper = info.cube_3d["top"]
    height = (upper - lower)[0, 2]


    avg_bottoms = []
    avg_heights = []
    seg_infos = []


    for segment_label, lower_3d, upper_3d in initial_segments:
        # Run Dijkstra from start
        segment_info = get_segment_by_id(segment_infos, segment_label) #next(iter(graph))  # e.g., first segment in graph
        segment_info.cube_3d = get_projected_cube_faces(lower_3d, upper_3d)
        
        loc_infos = segment_mask(img, (segments == segment_label), segment_info, curr_level+1, n_segments = n_segments, max_level = max_level, internal = True, debug=debug)
        if loc_infos is not None:
            seg_infos += loc_infos.copy()

    # for segment_label, lower_3d, upper_3d in initial_external_segments:
        # # Run Dijkstra from start
        # segment_info = get_segment_by_id(segment_infos, segment_label) #next(iter(graph))  # e.g., first segment in graph
        # segment_info.cube_3d = get_projected_cube_faces(lower_3d, upper_3d)
        
        # loc_infos = segment_mask(img, (segments == segment_label), segment_info, curr_level+1, n_segments = n_segments, max_level = max_level, internal = False, debug=debug)
        # if loc_infos is not None:
            # seg_infos += loc_infos.copy()

    
        # for info in segment_infos:
            # if info.cube_3d is None: continue
            # segment_bottoms[info.segment_id].append(info.cube_3d["bottom"].copy())
            # segment_tops[info.segment_id].append(info.cube_3d["top"].copy())
            
            # lower = info.cube_3d["bottom"]
            # upper = info.cube_3d["top"]
            
            # height = (upper - lower)[0, 2]
            
            # bottoms.append(lower.copy())
            # heights.append(height)
            
    if debug:       
        #draw_cubes_with_bounding_image(cube_img, bottoms, heights, alpha = 0.15)
        draw_segment_cubes_2d_3d(segment_infos, background_image=img, previous_segment_infos=[main_info], avg_paths=None, alpha=0.25)
    

    # # Step 3: Compute average cube per segment
    # avg_bottoms = []
    # avg_heights = []
    # seg_infos = []
    
    # for label in segment_bottoms:
        # bottoms = np.array(segment_bottoms[label])  # shape: (N, 4, 3)
        # tops = np.array(segment_tops[label])
        
        # avg_bottom = np.mean(bottoms, axis=0)
        # avg_top = np.mean(tops, axis=0)
        
        # height = (avg_top - avg_bottom)[0, 2]  # Use Z-axis height
        # # print(f"height: {height}")
        # # print("avg_bottom:\n", avg_bottom)
        # # print("avg_top:\n", avg_top)
        
        # # avg_bottoms.append(avg_bottom.copy())
        # # avg_heights.append(height)
        
        # segment_info = get_segment_by_id(segment_infos, label)

        
        # # if is_internal_segment(segments, label):
            # # y_width = 0.10
            # # avg_bottom, avg_top = fix_cube(avg_bottom, avg_top, y_width)

        # if clip:
            # segment_info.cube_3d = clip_cube_to_bounds(avg_bottom, avg_top, cube_3d)
        # #get_projected_cube_faces(avg_bottom, avg_top)
        # if label in 
        # loc_infos = segment_mask(img, (segments == label), segment_info, curr_level+1, n_segments = n_segments, max_level = max_level, debug=debug)
        # if loc_infos is not None:
            # seg_infos += loc_infos.copy()
        # # loc_infos = segment_mask(img, (segments == label), segment_info, curr_level+1, n_segments = 2, max_level = max_level, debug=debug)
        # # if loc_infos is not None:
            # # seg_infos += loc_infos.copy()
    
    # if debug:
        # draw_cubes_with_bounding_image(img, avg_bottoms, avg_heights, alpha=0.15)
    
    return seg_infos
    
def compute_ipm_transform(vert_vp, hor_left_vp, hor_right_vp, image_shape, margin=50):
    """
    Compute a projective transform (IPM) using 3 vanishing points.
    Assumes vanishing points define a rectangle in the BEV.

    Returns:
        - ProjectiveTransform object (inverse IPM)
        - Output shape of the warped image
    """
    h, w = image_shape[:2]
    # Choose a center point (approximate road intersection)
    center = np.array([w // 2, h // 2])

    # Define source quadrilateral (on the image)
    src = np.array([
        get_intersect(hor_left_vp, center, vert_vp, center),
        get_intersect(hor_right_vp, center, vert_vp, center),
        get_intersect(hor_left_vp, center, vert_vp, [center[0], h]),
        get_intersect(hor_right_vp, center, vert_vp, [center[0], h]),
    ], dtype=np.float32)

    if None in src:
        raise ValueError("Could not compute all IPM corner points from vanishing points.")

    # Define destination rectangle (warped IPM view)
    width = 300
    height = 400
    dst = np.array([
        [margin, margin],
        [margin + width, margin],
        [margin, margin + height],
        [margin + width, margin + height]
    ], dtype=np.float32)

    transform = ProjectiveTransform()
    if not transform.estimate(src, dst):
        raise RuntimeError("Failed to estimate projective transform.")

    return transform, (margin + width + margin, margin + height + margin)


def slic_with_ipm(img, ipm_transform, warped_img, ipm_shape, 
                  n_segments=2, compactness=50, sigma=1, mask=None, debug=True):

    warped_mask = None
    if mask is not None:
        warped_mask = warp(mask.astype(np.float32), ipm_transform, output_shape=ipm_shape) > 0.5

        if debug:
            plt.figure(figsize=(10, 5))
            plt.title("Warped Mask")
            plt.imshow(warped_mask, cmap='gray')
            plt.axis("off")
            plt.show()

        if np.sum(warped_mask) == 0:
            raise ValueError("Warped mask is empty — no foreground region to segment.")

    # Apply SLIC to warped image
    ipm_segments = slic(warped_img, n_segments=n_segments,
                        compactness=compactness, sigma=sigma,
                        start_label=1, mask=warped_mask)

    # Warp segmentation labels back to original space
    segments_original_space = warp(ipm_segments, ipm_transform.inverse,
                                   output_shape=img.shape[:2], order=0, preserve_range=True)

    return segments_original_space.astype(np.int32)
    
def segment_mask_v3(img, foreground_mask, info, curr_level, max_level = 2, debug = False, internal = False, clip = False, ipm_transform = None, warped_img = None, ipm_shape = None):
    if curr_level > max_level:
        return [info]
    
    main_info = info.copy()
    cube_2d = info.cube_2d
    cube_3d = info.cube_3d
    extrem = info.extrem

    # Step 1: Get center of foreground
    ys, xs = np.nonzero(foreground_mask)
    center_y = np.mean(ys)
    center_x = np.mean(xs)
    center_xy = (center_x, center_y)
    # print(center_xy)

    gradient = rank.gradient(img, disk(1))
    # Step 2: Convert extrem (list of coordinates) into marker image
    markers = np.zeros_like(img, dtype=np.int32)
    # Move each extrem point 1/10 of the way toward the center

    extrem_to_segment = {}  # Dictionary to store mapping
    
    for i, (x, y) in enumerate(extrem):  # extrem assumed as (x, y)
        if i == 0:
            shift_factor = 0.2
        else:
            shift_factor = 0.07
        max_attempts = 5  # Try decreasing shift up to 5 times
    
        for attempt in range(max_attempts):
            # Compute shifted point
            new_x = int(x + shift_factor * (center_xy[0] - x))
            new_y = int(y + shift_factor * (center_xy[1] - y))
    
            # Bounds and mask check
            if (0 <= new_y < markers.shape[0] and
                0 <= new_x < markers.shape[1] and
                foreground_mask[new_y, new_x]):
    
                marker_id = i + 1
                markers[new_y, new_x] = marker_id  # Unique label
                extrem_to_segment[(x, y)] = marker_id  # Store the mapping
                break  # Successfully placed marker
    
            shift_factor *= 0.5  # Reduce shift if point not valid

    segments = segmentation.watershed(gradient, markers=markers, mask=foreground_mask)

    # segmented = np.zeros_like(dist, dtype=np.uint8)
    # for i in range(1, marker_id):
        # segmented[labels == i] = int(255 * i / marker_id)

    if debug:
        import matplotlib.patches as patches
    
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img, cmap='gray')
        ax.imshow(segments, cmap='nipy_spectral', alpha=0.4)

        circ = patches.Circle(center_xy, radius=15, color='blue', fill=True)
        ax.add_patch(circ)
    
        # Draw marker circles
        ys, xs = np.nonzero(markers)
        for y, x in zip(ys, xs):
            circ = patches.Circle((x, y), radius=5, color='red', fill=True)
            ax.add_patch(circ)
    
        ax.set_title("Segmentation with Marker Circles")
        ax.axis('off')
        plt.tight_layout()
        plt.show()


    #segments = slic(img, n_segments=2, compactness=2*30, sigma = 1, start_label=1, mask=foreground_mask)
    segment_ids = np.unique(segments)
    
    num_segments = len(np.unique(segments)) - (1 if 0 in segments else 0)
    print("Number of segments:", num_segments)


    centers = []
    valid_ids = []
    
    for i in segment_ids:
        coords = np.column_stack(np.nonzero(segments == i))
        mean_yx = np.mean(coords, axis=0)
        y, x = map(int, mean_yx)
    
        # Check if center lies in the foreground mask
        if 0 <= y < foreground_mask.shape[0] and 0 <= x < foreground_mask.shape[1]:
            if foreground_mask[y, x]:  # Only include if inside the mask
                centers.append(mean_yx)
                valid_ids.append(i)

    if debug:
        plot_centers_and_edges(img, segments, np.array(centers))

    segment_infos = build_segment_info_list(segments, foreground_mask, vert_vp, hor_left_vp, hor_right_vp, debug=debug)


    hor_right_neg, hor_right_pos, vert_neg, vert_pos, hor_left_neg, hor_left_pos = extrem
    initial_segments = []

    if not internal:
        #vert_neg
        segment_label = find_segment_for_point(segments, vert_neg)
        #used_labels.add(segment_label)
        mask = (segments == segment_label)  # Get all pixels belonging to the segment
    
        # if debug:
            # highlite_segment(img, mask, segments, segment_label)
    
        segment_info = get_segment_by_id(segment_infos, segment_label)
        if segment_info is not None:
            if segment_info.cube_2d is not None:

                nb_cube_2d = segment_info.cube_2d
                #left
                face1 = "left"
                face2 = "right"
                
                pts1 = np.float32(cube_2d[face1])
                arr = cube_3d[face1].copy()
                arr = arr[:, 1:]
                arr[:, [0, 1]] = arr[:, [1, 0]]  # swap y-z -> z-y
                pts2 = np.float32(arr)
                ipm = get_projective_transform(pts1, pts2)
            
                dist_2d = dist0(cube_2d[face1], cube_2d[face2])
                dist_3d = dist0(cube_3d[face1], cube_3d[face2])
                nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
                if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
                nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
            
                mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
                mapped = mapped[:, [1, 0]]  # swap z-y
            
            
                l_3d = np.column_stack((np.full(mapped.shape[0], cube_3d[face1][0, 0]), mapped))
                r_3d = l_3d.copy()
                r_3d[:, 0] += nb_dist_3d
                
                upper_3d = np.array([l_3d[1], r_3d[1], r_3d[2], l_3d[2]])
                lower_3d = np.array([l_3d[0], r_3d[0], r_3d[3], l_3d[3]])
                
                initial_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))#, False))
                
                #forfard
                face1 = "forward"
                face2 = "backward"
                
                pts1 = np.float32(cube_2d[face1])
                arr = cube_3d[face1].copy()
                arr = arr[:, [0, 2]]  # remove Y 
                arr[:, [0, 1]] = arr[:, [1, 0]]  # swap x-z -> z-x
                pts2 = np.float32(arr)
                ipm = get_projective_transform(pts1, pts2)
                
                dist_2d = dist0(cube_2d[face1], cube_2d[face2])
                dist_3d = dist0(cube_3d[face1], cube_3d[face2])
                nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
                if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
                nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
            
                mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
                mapped = mapped[:, [1, 0]]  # swap z-x
                
                f_3d = np.column_stack((
                    mapped[:, 0],                                   # x
                    np.full(mapped.shape[0], cube_3d[face1][0, 1]), # FIX y
                    mapped[:, 1]                                    # Z
                ))
                b_3d = f_3d.copy()
                b_3d[:, 1] -= nb_dist_3d
            
            
                upper_3d = np.array([b_3d[1], b_3d[2], f_3d[2], f_3d[1]])
                lower_3d = np.array([b_3d[0], b_3d[3], f_3d[3], f_3d[0]])
            
                
                initial_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))

    
    # hor_right_neg
    segment_label = find_segment_for_point(segments, hor_right_neg)
    # if segment_label not in used_labels:
        # used_labels.add(segment_label)
    mask = (segments == segment_label)  # Get all pixels belonging to the segment
    
        # if debug:
            # highlite_segment(img, mask, segments, segment_label)
    
    segment_info = get_segment_by_id(segment_infos, segment_label)
    if segment_info is not None:
        if segment_info.cube_2d is not None:
            nb_cube_2d = segment_info.cube_2d
        
            # bottom
            face1 = "bottom"
            face2 = "top"
        
            pts1 = np.float32(cube_2d[face1])
            arr = cube_3d[face1].copy()[:, :2]
            pts2 = np.float32(arr)
            ipm = get_projective_transform(pts1, pts2)
        
            dist_2d = dist0(cube_2d[face1], cube_2d[face2])
            dist_3d = dist0(cube_3d[face1], cube_3d[face2])
            nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
            if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
            
            print(f"dist_2d: {dist_2d}")
            print(f"dist_3d: {dist_3d}")
            print(f"nb_dist_2d: {nb_dist_2d}")
            
            nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
        
            mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
        
            lower_3d = np.column_stack((mapped.copy(), np.full(mapped.shape[0], cube_3d[face1][0, 2])))
            upper_3d = lower_3d.copy()
            upper_3d[:, 2] += nb_dist_3d
        
            
            initial_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))
    
    
            # backward
            face1 = "backward"
            face2 = "forward"
        
        
            pts1 = np.float32(cube_2d[face1])
            arr = cube_3d[face1].copy()
            arr = arr[:, [0, 2]]  # remove Y 
            arr[:, [0, 1]] = arr[:, [1, 0]]  # swap x-z -> z-x
            pts2 = np.float32(arr)
            ipm = get_projective_transform(pts1, pts2)
            
            dist_2d = dist0(cube_2d[face1], cube_2d[face2])
            dist_3d = dist0(cube_3d[face1], cube_3d[face2])
            nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
            if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
            nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
        
            mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
            mapped = mapped[:, [1, 0]]  # swap z-x
            
            b_3d = np.column_stack((
                mapped[:, 0],                                   # x
                np.full(mapped.shape[0], cube_3d[face1][0, 1]), # FIX y
                mapped[:, 1]                                    # Z
            ))
            f_3d = b_3d.copy()
            f_3d[:, 1] += nb_dist_3d
        
        
            upper_3d = np.array([b_3d[1], b_3d[2], f_3d[2], f_3d[1]])
            lower_3d = np.array([b_3d[0], b_3d[3], f_3d[3], f_3d[0]])
        
            
            initial_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))

    # hor_right_pos
    segment_label = find_segment_for_point(segments, hor_right_pos)
    # if segment_label not in used_labels:
        # used_labels.add(segment_label)
    mask = (segments == segment_label)  # Get all pixels belonging to the segment

    if debug:
        highlite_segment(img, mask, segments, segment_label)

    segment_info = get_segment_by_id(segment_infos, segment_label)
    if segment_info is not None:
        if segment_info.cube_2d is not None:
            nb_cube_2d = segment_info.cube_2d
        
            #top
            face1 = "top"
            face2 = "bottom"
        
            pts1 = np.float32(cube_2d[face1])
            arr = cube_3d[face1].copy()[:, :2]
            pts2 = np.float32(arr)
            ipm = get_projective_transform(pts1, pts2)
        
            dist_2d = dist0(cube_2d[face1], cube_2d[face2])
            dist_3d = dist0(cube_3d[face1], cube_3d[face2])
            nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
            if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
            nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
        
            mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
        
            upper_3d = np.column_stack((mapped.copy(), np.full(mapped.shape[0], cube_3d[face1][0, 2])))
            lower_3d = upper_3d.copy()
            lower_3d[:, 2] -= nb_dist_3d
            
            initial_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))
        
    
            #forward
            face1 = "forward"
            face2 = "backward"
        
            pts1 = np.float32(cube_2d[face1])
            arr = cube_3d[face1].copy()
            arr = arr[:, [0, 2]]  # remove Y 
            arr[:, [0, 1]] = arr[:, [1, 0]]  # swap x-z -> z-x
            pts2 = np.float32(arr)
            ipm = get_projective_transform(pts1, pts2)
            
            dist_2d = dist0(cube_2d[face1], cube_2d[face2])
            dist_3d = dist0(cube_3d[face1], cube_3d[face2])
            nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
            if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
            nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
        
            mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
            mapped = mapped[:, [1, 0]]  # swap z-x
            
            f_3d = np.column_stack((
                mapped[:, 0],                                   # x
                np.full(mapped.shape[0], cube_3d[face1][0, 1]), # FIX y
                mapped[:, 1]                                    # Z
            ))
            b_3d = f_3d.copy()
            b_3d[:, 1] -= nb_dist_3d
        
        
            upper_3d = np.array([b_3d[1], b_3d[2], f_3d[2], f_3d[1]])
            lower_3d = np.array([b_3d[0], b_3d[3], f_3d[3], f_3d[0]])
        
            
            initial_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))




    # vert_pos
    segment_label = find_segment_for_point(segments, vert_pos)
    # if segment_label not in used_labels:
        # used_labels.add(segment_label)
    mask = (segments == segment_label)  # Get all pixels belonging to the segment

    # if debug:
        # highlite_segment(img, mask, segments, segment_label)

    segment_info = get_segment_by_id(segment_infos, segment_label)
    if segment_info is not None:
        if segment_info.cube_2d is not None:
            nb_cube_2d = segment_info.cube_2d
            #left
            face1 = "right"
            face2 = "left"
        
            pts1 = np.float32(cube_2d[face1])
            arr = cube_3d[face1].copy()
            arr = arr[:, 1:]
            arr[:, [0, 1]] = arr[:, [1, 0]]  # swap y-z -> z-y
            pts2 = np.float32(arr)
            ipm = get_projective_transform(pts1, pts2)
        
            dist_2d = dist0(cube_2d[face1], cube_2d[face2])
            dist_3d = dist0(cube_3d[face1], cube_3d[face2])
            nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
            #if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
            nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
        
            mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
            mapped = mapped[:, [1, 0]]  # swap z-y
        
        
            r_3d = np.column_stack((np.full(mapped.shape[0], cube_3d[face1][0, 0]), mapped))
            l_3d = r_3d.copy()
            l_3d[:, 0] -= nb_dist_3d
            
            upper_3d = np.array([l_3d[1], r_3d[1], r_3d[2], l_3d[2]])
            lower_3d = np.array([l_3d[0], r_3d[0], r_3d[3], l_3d[3]])
            
            initial_segments.append((segment_label, lower_3d.copy(), upper_3d.copy())) #, True))


    if not internal:
        # hor_left_neg
        segment_label = find_segment_for_point(segments, hor_left_neg)
        mask = (segments == segment_label)  # Get all pixels belonging to the segment
    
        if debug:
            highlite_segment(img, mask, segments, segment_label)
    
        segment_info = get_segment_by_id(segment_infos, segment_label)
        if segment_info is not None:
            if segment_info.cube_2d is not None:
                nb_cube_2d = segment_info.cube_2d
                #left
                face1 = "right"
                face2 = "left"
            
                pts1 = np.float32(cube_2d[face1])
                arr = cube_3d[face1].copy()
                arr = arr[:, 1:]
                arr[:, [0, 1]] = arr[:, [1, 0]]  # swap y-z -> z-y
                pts2 = np.float32(arr)
                ipm = get_projective_transform(pts1, pts2)
            
                dist_2d = dist0(cube_2d[face1], cube_2d[face2])
                dist_3d = dist0(cube_3d[face1], cube_3d[face2])
                nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
                if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
                nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
            
                mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
                mapped = mapped[:, [1, 0]]  # swap z-y
            
            
                r_3d = np.column_stack((np.full(mapped.shape[0], cube_3d[face1][0, 0]), mapped))
                l_3d = r_3d.copy()
                l_3d[:, 0] -= nb_dist_3d
                
                upper_3d = np.array([l_3d[1], r_3d[1], r_3d[2], l_3d[2]])
                lower_3d = np.array([l_3d[0], r_3d[0], r_3d[3], l_3d[3]])
                
                initial_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))

    if not internal:
        # hor_left_pos
        segment_label = find_segment_for_point(segments, hor_left_pos)
        mask = (segments == segment_label)  # Get all pixels belonging to the segment
    
        if debug:
            highlite_segment(img, mask, segments, segment_label)
    
        segment_info = get_segment_by_id(segment_infos, segment_label)
        if segment_info is not None:
            if segment_info.cube_2d is not None:
                nb_cube_2d = segment_info.cube_2d
                #left
                face1 = "left"
                face2 = "right"
                
                pts1 = np.float32(cube_2d[face1])
                arr = cube_3d[face1].copy()
                arr = arr[:, 1:]
                arr[:, [0, 1]] = arr[:, [1, 0]]  # swap y-z -> z-y
                pts2 = np.float32(arr)
                ipm = get_projective_transform(pts1, pts2)
            
                dist_2d = dist0(cube_2d[face1], cube_2d[face2])
                dist_3d = dist0(cube_3d[face1], cube_3d[face2])
                nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
                if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
                nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
            
                mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
                mapped = mapped[:, [1, 0]]  # swap z-y
            
            
                l_3d = np.column_stack((np.full(mapped.shape[0], cube_3d[face1][0, 0]), mapped))
                r_3d = l_3d.copy()
                r_3d[:, 0] += nb_dist_3d
                
                upper_3d = np.array([l_3d[1], r_3d[1], r_3d[2], l_3d[2]])
                lower_3d = np.array([l_3d[0], r_3d[0], r_3d[3], l_3d[3]])
                
                initial_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))

    # segment_data = defaultdict(list)

    # for segment_label, lower_3d, upper_3d in initial_segments:
        # # Run Dijkstra from start
        # start_segment = get_segment_by_id(segment_infos, segment_label) #next(iter(graph))  # e.g., first segment in graph
        # start_segment.cube_3d = get_projected_cube_faces(lower_3d, upper_3d)
        # distances, prev = dijkstra_segment_graph(start_segment)

        # for info in segment_infos:
            # segment_data[info.segment_id].append((info.cube_3d["bottom"].copy(), info.cube_3d["top"].copy()))

    # # Compute averages
    # averages = {}
    # for segment_id, cubes in segment_data.items():
        # lowers = np.array([item[0] for item in cubes])  # shape: (N, 4, 3)
        # uppers = np.array([item[1] for item in cubes])  # shape: (N, 4, 3)
        
        # avg_lower = np.mean(lowers, axis=0)  # shape: (4, 3)
        # avg_upper = np.mean(uppers, axis=0)  # shape: (4, 3)
    
        # averages[segment_id] = (avg_lower, avg_upper)
        # #print(f"Segment {segment_id} - Avg Lower:\n{avg_lower}\nAvg Upper:\n{avg_upper}")
        # #Compute bounding boxes (min/max) that contain all cubes per segment
        # #bounding_boxes = {}
        
    # # for segment_id, cubes in segment_data.items():
        # # lowers = np.array([item[0] for item in cubes])  # shape: (N, 4, 3)
        # # uppers = np.array([item[1] for item in cubes])  # shape: (N, 4, 3)
        # # internal = cubes[0][2]

        # # all_points = np.concatenate([lowers, uppers], axis=1).reshape(-1, 3)  # shape: (N*8, 3)
        
        # # min_point = np.min(all_points, axis=0)  # shape: (3,)
        # # max_point = np.max(all_points, axis=0)  # shape: (3,)
    
        # # # Create lower and upper faces of the box
        # # lower_face = np.array([
            # # [min_point[0], min_point[1], min_point[2]],
            # # [max_point[0], min_point[1], min_point[2]],
            # # [max_point[0], max_point[1], min_point[2]],
            # # [min_point[0], max_point[1], min_point[2]],
        # # ])
    
        # # upper_face = np.array([
            # # [min_point[0], min_point[1], max_point[2]],
            # # [max_point[0], min_point[1], max_point[2]],
            # # [max_point[0], max_point[1], max_point[2]],
            # # [min_point[0], max_point[1], max_point[2]],
        # # ])
    
        # # #bounding_boxes[segment_id] = (lower_face, upper_face)
        # # averages[segment_id] = (lower_face, upper_face, internal)
    # for info in segment_infos:
        # bottom, top = averages[info.segment_id]
        # info.cube_3d = get_projected_cube_faces(bottom, top)

    actual_initial_segments = []
    for segment_label, lower_3d, upper_3d in initial_segments:
        seg = get_segment_by_id(segment_infos, segment_label)
        seg.cube_3d = get_projected_cube_faces(lower_3d, upper_3d)
        actual_initial_segments.append(seg)

    if debug:       
        #draw_cubes_with_bounding_image(cube_img, bottoms, heights, alpha = 0.15)
        draw_segment_cubes_2d_3d(segment_infos, background_image=img, previous_segment_infos=[main_info], avg_paths=None, alpha=0.25, segments = segments, extrem = extrem)
    
    return actual_initial_segments

def faces_touch(face1, face2, threshold=1e-3):
    """Check if two cube faces are touching within a distance threshold."""
    dists = cdist(face1, face2)
    return np.any(dists < threshold)

def face_center(face):
    return np.mean(face, axis=0)

def extend_cube_to_face(info, target_face):
    """Extend info's cube to touch the given target face."""
    cube = info.cube_3d
    best_face_name = None
    min_dist = float("inf")
    for face_name, face_pts in cube.items():
        d = np.linalg.norm(face_center(face_pts) - face_center(target_face))
        if d < min_dist:
            min_dist = d
            best_face_name = face_name

    # Calculate shift vector
    face_pts = cube[best_face_name]
    shift_vector = face_center(target_face) - face_center(face_pts)

    # Apply shift to all cube faces
    new_cube = {}
    for k, v in cube.items():
        new_cube[k] = v + shift_vector
    info.cube_3d = new_cube




def faces_touch(face1, face2, threshold=1e-3):
    """Check if two cube faces are touching within a distance threshold."""
    return np.any(cdist(face1, face2) < threshold)

def face_center(face):
    return np.mean(face, axis=0)

def extend_faces_to_touch(lower_face, upper_face, target_face):
    """Shift the cube defined by lower_face and upper_face so a face touches target_face."""
    cube_faces = get_projected_cube_faces(lower_face, upper_face)
    best_face = None
    min_dist = float('inf')

    for face_name, face_pts in cube_faces.items():
        dist = np.linalg.norm(face_center(face_pts) - face_center(target_face))
        if dist < min_dist:
            min_dist = dist
            best_face = face_name

    shift = face_center(target_face) - face_center(cube_faces[best_face])
    new_lower = lower_face + shift
    new_upper = upper_face + shift
    return new_lower, new_upper

def ensure_segment_cubes_touch(main_lower, main_upper, averages):
    main_faces = get_projected_cube_faces(main_lower, main_upper)

    #for label, (lower_face, upper_face) in enumerate(segment_list):
    for label, (lower_face, upper_face) in averages.items():
        if np.allclose(lower_face, main_lower) and np.allclose(upper_face, main_upper):
            continue  # skip main cube itself

        cube_faces = get_projected_cube_faces(lower_face, upper_face)
        touching = False

        for f_main in main_faces.values():
            for f_seg in cube_faces.values():
                if faces_touch(f_main, f_seg):
                    touching = True
                    break
            if touching:
                break

        if not touching:
            # Extend current segment to touch first available face of main
            target_face = list(main_faces.values())[0]
            new_lower, new_upper = extend_faces_to_touch(lower_face, upper_face, target_face)
            averages[label] = (new_lower, new_upper)

def extend_averages_to_touch(averages, debug=False):
    # Unpack items
    items = list(averages.items())
    if len(items) != 2:
        return  # Or raise ValueError("Expected exactly two segments in averages.")

    (key1, (lower1, upper1, internal1)), (key2, (lower2, upper2, internal2)) = items

    # Compute bounding boxes
    min1 = np.min(np.vstack((lower1, upper1)), axis=0)
    max1 = np.max(np.vstack((lower1, upper1)), axis=0)
    min2 = np.min(np.vstack((lower2, upper2)), axis=0)
    max2 = np.max(np.vstack((lower2, upper2)), axis=0)

    # Initialize new min/max to current ones
    new_min1, new_max1 = min1.copy(), max1.copy()
    new_min2, new_max2 = min2.copy(), max2.copy()

    for axis in range(3):  # 0: x, 1: y, 2: z
        if max1[axis] < min2[axis]:  # Segment 1 is before segment 2
            mid = (max1[axis] + min2[axis]) / 2
            new_max1[axis] = mid
            new_min2[axis] = mid
            if debug:
                print(f"Axis {axis}: Segment1 is before Segment2. Adjusting to touch at {mid}.")
        elif max2[axis] < min1[axis]:  # Segment 2 is before segment 1
            mid = (max2[axis] + min1[axis]) / 2
            new_max2[axis] = mid
            new_min1[axis] = mid
            if debug:
                print(f"Axis {axis}: Segment2 is before Segment1. Adjusting to touch at {mid}.")
        else:
            if debug:
                print(f"Axis {axis}: Segments already touching or overlapping. No adjustment needed.")

    def make_faces(min_corner, max_corner):
        x1, y1, z1 = min_corner
        x2, y2, z2 = max_corner
        bottom = np.array([[x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1]])
        top    = np.array([[x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2]])
        return bottom, top

    # Update the averages dictionary in-place with adjusted face geometry
    bottom1, top1 = make_faces(new_min1, new_max1)
    bottom2, top2 = make_faces(new_min2, new_max2)

    averages[key1] = (bottom1, top1, internal1)
    averages[key2] = (bottom2, top2, internal2)

def segment_mask_v2(img, foreground_mask, info, curr_level, max_level = 2, debug = False, internal = False, clip = True):
    if curr_level > max_level:
        return [info]
    
    main_info = info.copy()
    cube_2d = info.cube_2d
    cube_3d = info.cube_3d
    extrem = info.extrem
    # Run SLIC only on masked foreground
    segments = slic(img, n_segments=2, compactness=100, start_label=1, mask=foreground_mask)
    segment_ids = np.unique(segments)
    


    centers = []
    valid_ids = []
    
    for i in segment_ids:
        coords = np.column_stack(np.nonzero(segments == i))
        mean_yx = np.mean(coords, axis=0)
        y, x = map(int, mean_yx)
    
        # Check if center lies in the foreground mask
        if 0 <= y < foreground_mask.shape[0] and 0 <= x < foreground_mask.shape[1]:
            if foreground_mask[y, x]:  # Only include if inside the mask
                centers.append(mean_yx)
                valid_ids.append(i)

    if debug:
        plot_centers_and_edges(img, segments, np.array(centers))

    segment_infos = build_segment_info_list(segments, foreground_mask, vert_vp, hor_left_vp, hor_right_vp, debug=debug)


    hor_right_neg, hor_right_pos, vert_neg, vert_pos, hor_left_neg, hor_left_pos = extrem
    initial_internal_segments = []
    initial_external_segments = []
    used_labels = set()
    
    if not internal:
        #vert_neg
        segment_label = find_segment_for_point(segments, vert_neg)
        used_labels.add(segment_label)
        mask = (segments == segment_label)  # Get all pixels belonging to the segment
    
        # if debug:
            # highlite_segment(img, mask, segments, segment_label)
    
        segment_info = get_segment_by_id(segment_infos, segment_label)
        center_yx = segment_info.center[[1, 0]] #np.array(segment_info.center[1], segment_info.center[0])
        lower_face, upper_face, extrem = get_projected_box(mask.astype(np.uint8), center_yx, vert_vp, hor_left_vp, hor_right_vp, internal = False, debug=False)
        segment_info.cube_2d = get_projected_cube_faces(lower_face, upper_face)
        segment_info.extrem = extrem.copy()
        
        
        if segment_info is not None:
            nb_cube_2d = segment_info.cube_2d
            #left
            face1 = "left"
            face2 = "right"
            
            pts1 = np.float32(cube_2d[face1])
            arr = cube_3d[face1].copy()
            arr = arr[:, 1:]
            arr[:, [0, 1]] = arr[:, [1, 0]]  # swap y-z -> z-y
            pts2 = np.float32(arr)
            ipm = get_projective_transform(pts1, pts2)
        
            dist_2d = dist0(cube_2d[face1], cube_2d[face2])
            dist_3d = dist0(cube_3d[face1], cube_3d[face2])
            nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
            if nb_dist_2d > dist_2d: nb_dist_2d = dist_2d
            nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
        
            mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
            mapped = mapped[:, [1, 0]]  # swap z-y
        
        
            l_3d = np.column_stack((np.full(mapped.shape[0], cube_3d[face1][0, 0]), mapped))
            r_3d = l_3d.copy()
            r_3d[:, 0] += nb_dist_3d
            
            upper_3d = np.array([l_3d[1], r_3d[1], r_3d[2], l_3d[2]])
            lower_3d = np.array([l_3d[0], r_3d[0], r_3d[3], l_3d[3]])
            
            initial_external_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))

    for pos in [hor_right_neg, vert_pos, hor_left_pos]:
        if len(used_labels) >= 2: break
        segment_label = find_segment_for_point(segments, pos)
        if segment_label not in used_labels:
            used_labels.add(segment_label)
            mask = (segments == segment_label)  # Get all pixels belonging to the segment
        
            # if debug:
                # highlite_segment(img, mask, segments, segment_label)
        
            segment_info = get_segment_by_id(segment_infos, segment_label)
            if segment_info is not None:
                center_yx = segment_info.center[[1, 0]]
                lower_face, upper_face, extrem = get_projected_box(mask.astype(np.uint8), center_yx, vert_vp, hor_left_vp, hor_right_vp, internal = True, debug=False)
                segment_info.cube_2d = get_projected_cube_faces(lower_face, upper_face)
                segment_info.extrem = extrem.copy()
    
                nb_cube_2d = segment_info.cube_2d
    
                # backward
                face1 = "backward"
                face2 = "forward"
            
            
                pts1 = np.float32(cube_2d[face1])
                arr = cube_3d[face1].copy()
                arr = arr[:, [0, 2]]  # remove Y 
                arr[:, [0, 1]] = arr[:, [1, 0]]  # swap x-z -> z-x
                pts2 = np.float32(arr)
                ipm = get_projective_transform(pts1, pts2)
                
                dist_2d = dist0(cube_2d[face1], cube_2d[face2])
                dist_3d = dist0(cube_3d[face1], cube_3d[face2])
                nb_dist_2d = dist0(nb_cube_2d[face1], nb_cube_2d[face2])
                nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
            
                mapped = map_points_to_BEV(nb_cube_2d[face1], ipm)
                mapped = mapped[:, [1, 0]]  # swap z-x
                
                b_3d = np.column_stack((
                    mapped[:, 0],                                   # x
                    np.full(mapped.shape[0], cube_3d[face1][0, 1]), # FIX y
                    mapped[:, 1]                                    # Z
                ))
                f_3d = b_3d.copy()
                f_3d[:, 1] += nb_dist_3d
            
            
                upper_3d = np.array([b_3d[1], b_3d[2], f_3d[2], f_3d[1]])
                lower_3d = np.array([b_3d[0], b_3d[3], f_3d[3], f_3d[0]])
            
                
                initial_internal_segments.append((segment_label, lower_3d.copy(), upper_3d.copy()))

     # Dictionary to collect lower and upper cubes per segment
    segment_bottoms = defaultdict(list)
    segment_tops = defaultdict(list)

    lower = info.cube_3d["bottom"]
    upper = info.cube_3d["top"]
    height = (upper - lower)[0, 2]


    avg_bottoms = []
    avg_heights = []
    seg_infos = []

    print(f"initial_internal_segments: {len(initial_internal_segments)}")
    print(f"initial_external_segments: {len(initial_external_segments)}")

    for segment_label, lower_3d, upper_3d in initial_internal_segments:
        # Run Dijkstra from start
        segment_info = get_segment_by_id(segment_infos, segment_label) #next(iter(graph))  # e.g., first segment in graph
        if clip:
            segment_info.cube_3d = clip_cube_to_bounds(lower_3d, upper_3d, cube_3d)
        else:
            segment_info.cube_3d = get_projected_cube_faces(lower_3d, upper_3d)

        
        loc_infos = segment_mask_v2(img, (segments == segment_label), segment_info, curr_level+1,  max_level = max_level, internal = True, debug=debug)
        if loc_infos is not None:
            seg_infos += loc_infos.copy()

    for segment_label, lower_3d, upper_3d in initial_external_segments:
        # Run Dijkstra from start
        segment_info = get_segment_by_id(segment_infos, segment_label) #next(iter(graph))  # e.g., first segment in graph
        if clip:
            segment_info.cube_3d = clip_cube_to_bounds(lower_3d, upper_3d, cube_3d)
        else:
            segment_info.cube_3d = get_projected_cube_faces(lower_3d, upper_3d)
        
        loc_infos = segment_mask_v2(img, (segments == segment_label), segment_info, curr_level+1, max_level = max_level, internal = False, debug=debug)
        if loc_infos is not None:
            seg_infos += loc_infos.copy()
            
    if debug:       
        #draw_cubes_with_bounding_image(cube_img, bottoms, heights, alpha = 0.15)
        draw_segment_cubes_2d_3d(segment_infos, background_image=img, previous_segment_infos=[main_info], avg_paths=None, alpha=0.25, extrem = extrem)
    
    return seg_infos


def main():
    img, ref = load_images()

    # Compute foreground mask using difference
    method = "otsu"  # Can be "adaptive", "otsu", or "fixed"
    foreground_mask = compute_foreground_mask(img, ref, method=method, threshold_value = 0.5)
    
    if debug:
        # Visualize the mask
        plt.figure(figsize=(8, 8))
        plt.imshow(foreground_mask, cmap='gray')
        plt.title(f"Foreground Mask ({method} thresholding)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    mask_center = np.mean(foreground_mask, axis=0)

    # ys, xs = np.nonzero(foreground_mask)
    # center_y = np.mean(ys)
    # center_x = np.mean(xs)
    # center_xy = (center_x, center_y)    
    # mask_center = np.array(center_y, center_x)

    lower_face, upper_face, extrem = get_projected_box(foreground_mask, np.mean(foreground_mask, axis=0), vert_vp, hor_left_vp, hor_right_vp, internal = False, debug=True)
    print(f"lower face: {lower_face}")
    print(f"lower face: {upper_face}")
    #draw_cube(lower_face, upper_face,  background_image=img, color='red', linewidth=4)
    draw_cube(lower_face, upper_face, background_image=img, save_path="cube_overlay.png")
    
    bottom = np.array([[0, 0, 0], [g_xmax, 0, 0], [g_xmax, g_ymax, 0], [0, g_ymax, 0]])
    top = np.array([[0, 0, g_zmax], [g_xmax, 0, g_zmax], [g_xmax, g_ymax, g_zmax], [0, g_ymax, g_zmax]])
    cube_2d = get_projected_cube_faces(lower_face, upper_face)  # Get all 6 faces of the cube
    cube_3d = get_projected_cube_faces(bottom, top)
    
    #left
    face1 = "bottom"
    face2 = "bottom"
    
    # pts1 = np.float32(cube_2d[face1])
    # arr = cube_3d[face1].copy()[:, :2]
    # pts2 = np.float32(arr)
    # ipm_transform = get_projective_transform(pts1, pts2)
    # ipm_shape = (g_ymax *2, g_xmax)
    # warped_img = warp(img, ipm_transform, output_shape=ipm_shape)

  
    # # Optional debug view of warped image
    # if debug:
        # plt.figure(figsize=(10, 5))
        # plt.title("Warped Image (IPM)")
        # plt.imshow(warped_img)
        # plt.axis("off")
        # plt.show()

    center_xy = (mask_center[1], mask_center[0])

    segment_info = SegmentInfo(
                segment_id=1,
                center=center_xy,
                cube_2d=cube_2d,
                int = None,
                extrem = extrem
            )
    segment_info.cube_3d = cube_3d
    # segment_info.cube_2d = cube_2d
    # segment_info.extrem = extrem

    avg_bottoms = []
    avg_heights = []
    centers_3d = []
    max_level = 0

    print(center_xy)
    gray = color.rgb2gray(img)
    segment_infos = segment_mask_v3(gray, foreground_mask, segment_info, curr_level = 0, max_level = max_level, internal = False, debug = True, clip = True) #, ipm_transform= ipm_transform, warped_img = warped_img, ipm_shape = ipm_shape)
    

    draw_segment_cubes_2d_3d(segment_infos, background_image=img, previous_segment_infos=[segment_info], avg_paths=None, alpha=0.55, previous_alpha=0.25, extrem = extrem)

    for info in segment_infos:
        bottom = info.cube_3d["bottom"]
        top = info.cube_3d["top"]
        # height = (top - bottom)[0, 2]
        center = 0.5 * (bottom.mean(axis=0) + top.mean(axis=0))
        centers_3d.append(center)
        
        # avg_bottoms.append(bottom.copy())
        # avg_heights.append(height)
        
    # segment_infos = segment_mask(img, foreground_mask, segment_info, curr_level = 0, n_segments = 3, max_level = max_level, debug = False)

    # for info in segment_infos:
        # bottom = info.cube_3d["bottom"]
        # top = info.cube_3d["top"]
        # height = (top - bottom)[0, 2]
        
        # avg_bottoms.append(bottom.copy())
        # avg_heights.append(height)
    # Step 4: Draw simplified average cubes

    # draw_cubes_with_bounding_image(img, avg_bottoms, avg_heights, alpha=0.15)
    
    # draw_3d_triangulation(centers_3d)
    
    draw_alpha_shape(centers_3d, alpha = 0.2)
 

if __name__ == "__main__":
    main()

#TODO
# import imageio
# import numpy as np
# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
# from skimage.measure import regionprops
# from skimage.segmentation import slic
# from skimage.segmentation import mark_boundaries

# def rgb2gray(rgb):
    # return np.dot(rgb[..., :3], [0.2126, 0.7152, 0.0722])

# image = imageio.imread(img_file_path)
# segments_slic = slic(image, n_segments=250, compactness=100)
# regions = regionprops(segments_slic, intensity_image=rgb2gray(image))
# for props in regions:
    # cy, cx = props.centroid
    # plt.plot(cx, cy, 'ro')

# plt.imshow(mark_boundaries(image, segments_slic))
# plt.show()
