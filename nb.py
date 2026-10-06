import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu, threshold_local
from skimage.segmentation import slic, mark_boundaries
from scipy.spatial import Delaunay
from matplotlib.lines import Line2D
from skimage import img_as_float
import sys
import os


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










def compute_foreground_mask(img, ref, method="otsu", threshold_value=0.1):
    # Convert to grayscale
    gray_img = rgb2gray(img)
    gray_ref = rgb2gray(ref)

    # Absolute difference
    abs_diff = np.abs(gray_img - gray_ref)
    abs_diff_blur = gaussian(abs_diff, sigma=25.0)

    # Thresholding
    if method == "adaptive":
        block_size = 35
        local_thresh = threshold_local(abs_diff_blur, block_size)
        object_mask = abs_diff_blur > local_thresh
    elif method == "otsu":
        thresh_val = threshold_otsu(abs_diff_blur)
        object_mask = abs_diff_blur > thresh_val
    else:
        object_mask = abs_diff_blur > threshold_value

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

def main():
    img, ref = load_images()

    # Compute foreground mask using difference
    method = "otsu"  # Can be "adaptive", "otsu", or "fixed"
    foreground_mask = compute_foreground_mask(img, ref, method=method)
    
    # Visualize the mask
    plt.figure(figsize=(8, 8))
    plt.imshow(foreground_mask, cmap='gray')
    plt.title(f"Foreground Mask ({method} thresholding)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Run SLIC only on masked foreground
    segments = slic(img, n_segments=50, compactness=20, start_label=1, mask=foreground_mask)
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

    
    centers = np.array(centers)

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)
    
    # draw centers and neighbors
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    plt.imshow(mark_boundaries(img, segments))
    plt.scatter(centers[:,1],centers[:,0], c='y')
    
    # this contains the neighbors list: tri.vertex_neighbor_vertices
    indptr,indices = tri.vertex_neighbor_vertices
    
    # draw lines from each center to its neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        centerA = np.repeat([centers[i]], len(N), axis=0)
        centerB = centers[N]
    
        for y0,x0,y1,x1 in np.hstack([centerA,centerB]):
            l = Line2D([x0,x1],[y0,y1], alpha=0.5)
            ax.add_line(l)
            
    plt.show()
    
 

if __name__ == "__main__":
    main()
