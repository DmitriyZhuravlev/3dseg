import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import to_rgba
import numpy as np
from skimage.segmentation import mark_boundaries
import math
import random

from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import alphashape
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_mask(mask, method="otsu"):
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray')
    plt.title(f"Foreground Mask ({method} thresholding)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_centers_and_edges(img, segments, centers, tri = None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.imshow(mark_boundaries(img, segments))
    plt.scatter(centers[:, 1], centers[:, 0], c='y')

    if tri is not None:
        indptr, indices = tri.vertex_neighbor_vertices
        for i in range(len(indptr) - 1):
            N = indices[indptr[i]:indptr[i + 1]]
            centerA = np.repeat([centers[i]], len(N), axis=0)
            centerB = centers[N]
            for y0, x0, y1, x1 in np.hstack([centerA, centerB]):
                l = Line2D([x0, x1], [y0, y1], alpha=0.5)
                ax.add_line(l)

    plt.title("Superpixel Centers and Delaunay Neighbors")
    plt.axis('off')
    plt.show()

def plot_graph_with_avg_weights(centers, tri, avg_weights_dict):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.scatter(centers[:, 1], centers[:, 0], c='white', edgecolors='black', zorder=2)

    offset = 10
    colors = {"DFS": "red", "BFS": "green", "Dijkstra": "blue"}

    for method, weights in avg_weights_dict.items():
        for idx, (y, x) in enumerate(centers):
            if idx in weights:
                ax.text(x + offset, y + offset, f"{weights[idx]:.1f}", color=colors[method], fontsize=8)

    indptr, indices = tri.vertex_neighbor_vertices
    for i in range(len(indptr) - 1):
        N = indices[indptr[i]:indptr[i + 1]]
        centerA = np.repeat([centers[i]], len(N), axis=0)
        centerB = centers[N]
        for y0, x0, y1, x1 in np.hstack([centerA, centerB]):
            l = Line2D([x0, x1], [y0, y1], color='gray', alpha=0.3)
            ax.add_line(l)

    legend_lines = [Line2D([0], [0], color=c, label=m) for m, c in colors.items()]
    plt.legend(handles=legend_lines, loc='upper right')
    plt.title("Graph with Avg Path Cost to Each Node")
    plt.axis('off')
    plt.show()


def draw_cube(lower_face, upper_face, color='red', background_image=None, linewidth=2, save_path=None):
    """
    Draws a cube using lower and upper face 2D projections, optionally saving the image.

    Args:
        lower_face (np.array): Shape (4, 2), lower face coordinates.
        upper_face (np.array): Shape (4, 2), upper face coordinates.
        color (str): Color for cube edges.
        background_image (np.array): Background image to draw over (H x W x 3 or H x W).
        linewidth (int): Line thickness.
        save_path (str or None): If provided, saves the image to this path instead of showing.
    """
    lower_face = np.array(lower_face)
    upper_face = np.array(upper_face)

    # Determine figure size to match image resolution
    if background_image is not None:
        height, width = background_image.shape[:2]
        dpi = 100
        figsize = (width / dpi, height / dpi)
    else:
        figsize = (10, 10)
        dpi = 100

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if background_image is not None:
        ax.imshow(background_image, cmap='gray')

    # Draw bottom and top faces
    for face in [lower_face, upper_face]:
        xs, ys = zip(*np.vstack([face, face[0]]))  # close the loop
        ax.plot(xs, ys, color=color, linewidth=linewidth)

    # Draw vertical edges
    for i in range(4):
        x_vals = [lower_face[i][0], upper_face[i][0]]
        y_vals = [lower_face[i][1], upper_face[i][1]]
        ax.plot(x_vals, y_vals, color=color, linewidth=linewidth)

    ax.axis('off')
    plt.tight_layout(pad=0)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved cube image to: {save_path}")
    else:
        plt.show()

def draw_cube_old(lower_face, upper_face, color='red',  background_image=None, linewidth=2):
    """
    Draws a cube in matplotlib from lower and upper face coordinates.

    Args:
        ax (matplotlib.axes.Axes): Axes object to draw on.
        lower_face (np.array): Shape (4, 2), lower face coordinates.
        upper_face (np.array): Shape (4, 2), upper face coordinates.
        color (str): Color for cube edges.
        linewidth (int): Line thickness.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    if background_image is not None:
        ax.imshow(background_image, cmap='gray')
    
    lower_face = np.array(lower_face)
    upper_face = np.array(upper_face)

    # Draw bottom and top faces
    for face in [lower_face, upper_face]:
        xs, ys = zip(*np.vstack([face, face[0]]))  # close the loop
        ax.plot(xs, ys, color=color, linewidth=linewidth)

    # Draw vertical edges
    for i in range(4):
        x_vals = [lower_face[i][0], upper_face[i][0]]
        y_vals = [lower_face[i][1], upper_face[i][1]]
        ax.plot(x_vals, y_vals, color=color, linewidth=linewidth)

    ax.set_title("Projected Cube")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def draw_all_segment_cubes_matplotlib(segment_info_list, background_image=None, avg_paths=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    if background_image is not None:
        ax.imshow(background_image, cmap='gray')

    for info in segment_info_list:
        lower = info.cube_2d["bottom"]
        upper = info.cube_2d["top"]
        for i in range(4):
            x = [lower[i][0], lower[(i+1)%4][0]]
            y = [lower[i][1], lower[(i+1)%4][1]]
            ax.plot(x, y, color='cyan')

            x = [upper[i][0], upper[(i+1)%4][0]]
            y = [upper[i][1], upper[(i+1)%4][1]]
            ax.plot(x, y, color='cyan')

            x = [lower[i][0], upper[i][0]]
            y = [lower[i][1], upper[i][1]]
            ax.plot(x, y, color='cyan')

        if avg_paths and info.segment_id in avg_paths:
            cx, cy = info.center[1], info.center[0]
            ax.text(cx, cy, f"{avg_paths[info.segment_id]:.1f}", color='red', fontsize=9, ha='center')

    ax.set_title("Projected Cubes with Avg Path Length")
    ax.axis('off')
    plt.tight_layout()

    # Convert plot to a numpy array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    buf = buf[:, :, [1, 2, 3]]  # drop alpha and reorder ARGB to RGB

    plt.close(fig)
    return buf

def darken_color(color, factor=0.7):
    rgb = mcolors.to_rgb(color)  # Convert to (R, G, B)
    return tuple([max(0, c * factor) for c in rgb])  # Darken by factor

def draw_3d_bounding_box(ax, lower_face, h, color="blue", alpha = 0.25):
    lower_face = np.array(lower_face)

    # if color is None:
        # _, color = generate_random_color()

    # edge_color = darken_color(color)  # Compute darker edge color

    # Convert to 3D by adding z=0 for the lower face if in 2D
    if lower_face.shape[1] == 2:
        lower_face = np.hstack([lower_face, np.zeros((4, 1))])

    # Create the upper face by adding height h
    upper_face = lower_face.copy()
    upper_face[:, 2] += h

    verts = [
        [lower_face[0], lower_face[1], upper_face[1], upper_face[0]],
        [lower_face[1], lower_face[2], upper_face[2], upper_face[1]],
        [lower_face[2], lower_face[3], upper_face[3], upper_face[2]],
        [lower_face[3], lower_face[0], upper_face[0], upper_face[3]],
        [lower_face[0], lower_face[1], lower_face[2], lower_face[3]],
        [upper_face[0], upper_face[1], upper_face[2], upper_face[3]]
    ]

    # Ensure the color is valid RGBA
    try:
        face_color = to_rgba(color, alpha)
    except Exception:
        face_color = to_rgba("blue", alpha)  # fallback

    edge_color = darken_color(face_color)  # Compute darker edge color

    box = Poly3DCollection(verts, facecolors=[face_color], edgecolors=[edge_color], linewidths=1)
    ax.add_collection3d(box)

def draw_segment_cubes_2d_3d(segment_info_list, background_image=None, avg_paths=None,
                             previous_segment_infos=None, alpha=0.25, previous_alpha=0.55, segments=None, extrem = []):
    def random_color():
        return tuple(random.random() for _ in range(3))

    def draw_3d_bounding_box(ax, lower_face, height, color, alpha=0.25):
        lower = np.array(lower_face)
        upper = lower.copy()
        upper[:, 2] += height

        verts = []
        for i in range(4):
            verts.append([lower[i], lower[(i+1)%4], upper[(i+1)%4], upper[i]])
        verts.append(lower)
        verts.append(upper)

        pc = Poly3DCollection(verts, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha)
        ax.add_collection3d(pc)

    fig = plt.figure(figsize=(14, 6))
    ax_img = fig.add_subplot(121)
    ax_3d = fig.add_subplot(122, projection='3d')
    ax_3d.view_init(elev=30, azim=-145, roll=-8)
    ax_img.set_title("2D Projected Cubes")
    ax_3d.set_title("3D Cubes")

    if background_image is not None:
        if segments is not None:
            # Overlay segment boundaries in blue
            marked_img = mark_boundaries(background_image, segments, mode='thick', color=(0, 0, 1))
            ax_img.imshow(marked_img)
        else:
            ax_img.imshow(background_image, cmap='gray')
    ax_img.axis('off')

    # ===== Draw Previous Level Segments First =====
    if previous_segment_infos:
        for info in previous_segment_infos:
            color = (0.6, 0.6, 0.6)  # Fixed gray color for previous level

            lower_2d = info.cube_2d["bottom"]
            upper_2d = info.cube_2d["top"]
            for i in range(4):
                ax_img.plot([lower_2d[i][0], lower_2d[(i+1)%4][0]],
                            [lower_2d[i][1], lower_2d[(i+1)%4][1]], color=color, alpha=previous_alpha)
                ax_img.plot([upper_2d[i][0], upper_2d[(i+1)%4][0]],
                            [upper_2d[i][1], upper_2d[(i+1)%4][1]], color=color, alpha=previous_alpha)
                ax_img.plot([lower_2d[i][0], upper_2d[i][0]],
                            [lower_2d[i][1], upper_2d[i][1]], color=color, alpha=previous_alpha)

            lower_3d = np.array(info.cube_3d["bottom"])
            upper_3d = np.array(info.cube_3d["top"])
            height = upper_3d[0, 2] - lower_3d[0, 2]
            draw_3d_bounding_box(ax_3d, lower_3d, height, color=color, alpha=previous_alpha)

    # ===== Draw Current Segments =====
    for info in segment_info_list:
        if info.cube_3d is None:
            continue
        color = random_color()

        lower_2d = info.cube_2d["bottom"]
        upper_2d = info.cube_2d["top"]
        for i in range(4):
            ax_img.plot([lower_2d[i][0], lower_2d[(i+1)%4][0]],
                        [lower_2d[i][1], lower_2d[(i+1)%4][1]], color=color)
            ax_img.plot([upper_2d[i][0], upper_2d[(i+1)%4][0]],
                        [upper_2d[i][1], upper_2d[(i+1)%4][1]], color=color)
            ax_img.plot([lower_2d[i][0], upper_2d[i][0]],
                        [lower_2d[i][1], upper_2d[i][1]], color=color)

        if avg_paths and info.segment_id in avg_paths:
            cx, cy = info.center[1], info.center[0]
            ax_img.text(cx, cy, f"{avg_paths[info.segment_id]:.1f}", color='red', fontsize=9, ha='center')

        lower_3d = np.array(info.cube_3d["bottom"])
        upper_3d = np.array(info.cube_3d["top"])
        height = upper_3d[0, 2] - lower_3d[0, 2]
        draw_3d_bounding_box(ax_3d, lower_3d, height, color=color, alpha=alpha)
    
    for corner in extrem:
        ax_img.scatter(*corner, color="r", s=30)

    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

def draw_cubes_with_bounding_image(bounding_box_image, lower_faces, heights, color = "blue", alpha = 0.25):
    """
    Display the bounding_box_image on the left and draw 3D bounding boxes on the right.
    """
    #plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(12, 6))  # Create a wide figure for side-by-side display

    # Create subplots
    ax_img = fig.add_subplot(121)  # Left subplot for the bounding_box_image
    ax_3d = fig.add_subplot(122, projection='3d')  # Right subplot for the 3D bounding boxes
    ax_3d.view_init(elev=30, azim=-145, roll = -8)  # Align axes like in a mathematical system

    if True: #while True:
        # Left: Display the bounding_box_image
        ax_img.cla()  # Clear the image plot
        ax_img.imshow(bounding_box_image)
        ax_img.axis('off')  # Turn off axis for better visualization
        ax_img.set_title("Bounding Box Image")

        # Right: Display the 3D bounding boxes
        ax_3d.cla()  # Clear the 3D plot
        #colors = [(128, 128, 128) if c is None else c for c in (colors or [])]
        for lower_face, height in zip(lower_faces, heights):
            # if color is None:
            #color = [c / 255.0 for c in cv_colors.BLUE.value[::-1]] 
            draw_3d_bounding_box(ax_3d, lower_face, height, color, alpha) #, color=color)


        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title("3D Bounding Boxes")
        #set_equal_axes(ax_3d)  # Ensure equal scaling for the 3D plot
        ax_3d.set_aspect('equal', adjustable='box')

        # plt.draw()
        # plt.pause(0.001)  # Small delay for interactive updates

        # # Break the loop if a key is pressed
        # # print("Press any key to render new boxes...")
        # # if plt.waitforbuttonpress():
            # # break

    # plt.ioff()  # Turn off interactive mode
    # plt.close(fig)  # Close the figure when done
    plt.show()


    
def draw_segment_graph(segment_info_list, segments = None, background_image=None, draw_cubes=True, draw_faces=True):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if background_image is not None:
        ax.imshow(background_image, cmap='gray')


    for src in segment_info_list:
        src_center = src.center
        ax.plot(src_center[0], src_center[1], 'o', color='blue')  # Draw source center

        if draw_cubes:
            lower = src.cube_2d["bottom"]
            upper = src.cube_2d["top"]

            for i in range(4):
                x = [lower[i][0], lower[(i + 1) % 4][0]]
                y = [lower[i][1], lower[(i + 1) % 4][1]]
                ax.plot(x, y, color='cyan')

                x = [upper[i][0], upper[(i + 1) % 4][0]]
                y = [upper[i][1], upper[(i + 1) % 4][1]]
                ax.plot(x, y, color='cyan')

                x = [lower[i][0], upper[i][0]]
                y = [lower[i][1], upper[i][1]]
                ax.plot(x, y, color='cyan')

        for dst, weight, direction, inter in src.neighbors:
            dst_center = dst.center
            #dst = neighbor_info.dst
            dst_center = dst.center
            # weight = neighbor_info.weight
            # direction = neighbor_info.direction
            # inter = neighbor_info.intersection

            # Draw arrow from src to dst
            ax.annotate(
                '',
                xy=(dst_center[0], dst_center[1]),
                xytext=(src_center[0], src_center[1]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
            )

            # Compute middle point
            mid_x = (src_center[0] + dst_center[0]) / 2
            mid_y = (src_center[1] + dst_center[1]) / 2

            # Compute direction vector from src to dst
            dir_x = dst_center[0] - src_center[0]
            dir_y = dst_center[1] - src_center[1]
            
            # Normalize the direction
            norm = math.hypot(dir_x, dir_y)
            if norm == 0:
                norm = 1e-6
            dir_x /= norm
            dir_y /= norm

            # Compute perpendicular vector
            perp_x = -dir_y
            perp_y = dir_x

            # Small shift
            shift_amount = 15  # pixels
            shifted_mid_x = mid_x + perp_x * shift_amount
            shifted_mid_y = mid_y + perp_y * shift_amount

            # Draw direction label
            ax.text(shifted_mid_x, shifted_mid_y, direction, color='green', fontsize=8)

            # Draw intersection face
            if draw_faces and inter is not None and not inter.is_empty:
                x, y = inter.exterior.xy
                ax.fill(x, y, color='yellow', alpha=0.08)

    if segments is not None:
        ax.imshow(mark_boundaries(background_image, segments, mode = 'thick', color = (0, 0, 255)))

    ax.set_title("Segment Graph with Connections and Faces")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
import math
import matplotlib.pyplot as plt

def draw_path(segment_infos, path, start_segment, target_segment):
    # Visualize full graph and highlight shortest path
    plt.figure(figsize=(10, 8))
    
    # Draw all edges
    for seg in segment_infos:
        for neighbor, _, _, _ in seg.neighbors:
            x_vals = [seg.center[0], neighbor.center[0]]  # x = col
            y_vals = [seg.center[1], neighbor.center[1]]  # y = row
            plt.plot(x_vals, y_vals, 'lightgray', linewidth=0.5)
    
    # Draw all nodes
    for seg in segment_infos:
        plt.plot(seg.center[0], seg.center[1], 'o', color='gray')
        plt.text(seg.center[0], seg.center[1], str(seg.segment_id), fontsize=8, color='black', ha='center', va='center')
    
    # Highlight path with arrows and directions
    for i in range(len(path) - 1):
        a = path[i].center
        b = path[i + 1].center

        # Draw path line
        plt.plot([a[0], b[0]], [a[1], b[1]], 'r', linewidth=2)

        # Draw arrow
        plt.annotate(
            '',
            xy=(b[0], b[1]),
            xytext=(a[0], a[1]),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
        )

        # Find relative direction between path[i] and path[i+1]
        rel_dir = None
        for neighbor, _, direction, _ in path[i].neighbors:
            if neighbor == path[i + 1]:
                rel_dir = direction
                break

        if rel_dir is not None:
            # Compute middle point
            mid_col = (a[0] + b[0]) / 2
            mid_row = (a[1] + b[1]) / 2

            # Compute direction vector (for small shift)
            dir_col = b[0] - a[0]
            dir_row = b[1] - a[1]
            norm = math.hypot(dir_col, dir_row)
            if norm == 0:
                norm = 1e-6
            dir_col /= norm
            dir_row /= norm

            # Perpendicular shift to avoid overlapping the arrow
            perp_col = -dir_row
            perp_row = dir_col
            shift_amount = 5  # pixels
            shifted_mid_col = mid_col + perp_col * shift_amount
            shifted_mid_row = mid_row + perp_row * shift_amount

            # Draw direction label
            plt.text(shifted_mid_col, shifted_mid_row, rel_dir, color='green', fontsize=8, ha='center', va='center')
    
    # Highlight start and end
    plt.plot(start_segment.center[0], start_segment.center[1], 'go', markersize=10, label="Start")
    plt.plot(target_segment.center[0], target_segment.center[1], 'ro', markersize=10, label="Target")
    
    plt.gca().invert_yaxis()
    plt.title("Shortest Path on Segment Graph with Directions")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def highlite_segment(img, mask, segments, segment_label):
    # Create a copy of the image to highlight the segment
    highlighted_img = img.copy()
    
    # Optionally dim everything else except the selected segment
    highlighted_img[~mask] = highlighted_img[~mask] * 0.3  # or set to grayscale
    
    # Plot the image with segment boundaries
    fig, ax = plt.subplots()
    ax.imshow(mark_boundaries(highlighted_img, segments, color=(0, 1, 1)))  # cyan boundaries
    
    # Draw a rectangle around the selected segment
    ys, xs = np.where(mask)
    if len(xs) > 0 and len(ys) > 0:
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
    
        import matplotlib.patches as patches
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='cyan', facecolor='none'
        )
        ax.add_patch(rect)
    
        # Optional: add label text
        ax.text(
            x_min, y_min - 5, f'Segment {segment_label}',
            color='cyan', fontsize=10, fontweight='bold', backgroundcolor='black'
        )
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def draw_3d_triangulation(centers_3d):
    """
    Draws a 3D Delaunay triangulation of the given 3D centers using Matplotlib.
    
    Parameters:
    - centers_3d: (N, 3) numpy array of 3D points
    """
    centers_3d = np.asarray(centers_3d)
    if centers_3d.shape[1] != 3:
        raise ValueError("centers_3d must be an (N, 3) array")
    
    # Compute 3D Delaunay triangulation
    tri = Delaunay(centers_3d)
    
    # Extract line segments for edges of tetrahedra
    edge_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    lines = []

    for simplex in tri.simplices:
        pts = centers_3d[simplex]
        for i, j in edge_indices:
            lines.append([pts[i], pts[j]])

    # Plot using Matplotlib
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    line_collection = Line3DCollection(lines, colors='blue', linewidths=1.0, alpha=0.6)
    ax.add_collection3d(line_collection)

    # Scatter plot of centers
    ax.scatter(centers_3d[:, 0], centers_3d[:, 1], centers_3d[:, 2], color='red', s=30)

    # Set axes limits
    ax.set_xlim(centers_3d[:, 0].min(), centers_3d[:, 0].max())
    ax.set_ylim(centers_3d[:, 1].min(), centers_3d[:, 1].max())
    ax.set_zlim(centers_3d[:, 2].min(), centers_3d[:, 2].max())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Delaunay Triangulation of Centers')
    plt.tight_layout()
    plt.show()


def draw_alpha_shape(points, alpha=1.0):
    import trimesh

    points = np.array(points)
    shape = alphashape.alphashape(points, alpha)

    if isinstance(shape, trimesh.Trimesh):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(shape.triangles, alpha=0.5, facecolor='cyan')
        ax.add_collection3d(mesh)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', s=5)
        #ax.auto_scale_xyz(points[:, 0], points[:, 1], points[:, 2])
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    else:
        print("Alpha shape did not produce a 3D mesh.")
