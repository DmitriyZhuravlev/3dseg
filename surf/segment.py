import numpy as np
import heapq
import math
from collections import defaultdict, deque

from scipy.spatial import Delaunay
from skimage.transform import ProjectiveTransform

from shapely.geometry import Polygon

from cube import *

face_pairs = {
    "left": ("left", "right"),
    "right": ("right", "left"),
    "top": ("top", "bottom"),
    "bottom": ("bottom", "top"),
    "forward": ("forward", "backward"),
    "backward": ("backward","forward" )
    
}

def get_projective_transform(pts_src, pts_dst):
    transform = ProjectiveTransform()
    success = transform.estimate(np.array(pts_src), np.array(pts_dst))
    if not success:
        raise ValueError("Projective transform estimation failed")
    return transform


def default_weight(a, b):
    return float(np.linalg.norm(a - b))

class SegmentInfo:
    def __init__(self, segment_id, center, cube_2d, int, extrem):
        self.segment_id = segment_id
        self.center = np.array(center)
        self.cube_2d = cube_2d.copy()
        self.int = int.copy() if int is not None else None
        self.extrem = extrem.copy()
        self.intersection = None
        self.cube_3d = None
        self.face_index = None  # or initialize with a valid default
        self.neighbors = []  # List of (SegmentInfo, distance, direction)

    def add_neighbor(self, other, weight, direction, intersection):
        #direction = other.center - self.center
        intersection = Polygon(intersection.exterior.coords)
        self.neighbors.append((other, weight, direction, intersection))

    def copy(self):
        new_instance = SegmentInfo(
            self.segment_id,
            self.center.copy(),
            self.cube_2d.copy(),
            self.int,
            self.extrem.copy()
        )
        new_instance.cube_2d = self.cube_2d
        new_instance.extrem = self.extrem
        new_instance.intersection = self.intersection  # shallow copy; deep copy if needed
        new_instance.cube_3d = self.cube_3d  # shallow copy; deep copy if needed
        new_instance.int = self.int  # shallow copy; deep copy if needed
        new_instance.face_index = self.face_index
        new_instance.neighbors = list(self.neighbors)  # optionally deep copy if needed
        return new_instance
    
    def __repr__(self):
        return f"SegmentInfo(id={self.segment_id}, center={self.center})"
        
def get_segment_by_id(list_info, segment_id, debug = True):
    """
    Retrieve a SegmentInfo node from the graph by its segment_id.
    
    Args:
        graph: dict mapping SegmentInfo -> list of neighbors.
        segment_id: the ID to search for.

    Returns:
        SegmentInfo instance with matching segment_id, or None if not found.
    """
    for segment in list_info:
        if segment.segment_id == segment_id:
            return segment
    if debug: print("Segment not found")
    raise ValueError('A very specific bad thing happened.')
    return None

def build_segment_info_list(segments, foreground_mask, vert_vp, hor_left_vp, hor_right_vp, debug=False):
    segment_info_list = []
    segment_ids = np.unique(segments)

    for segment_id in segment_ids:
        print(f"Proccessing {segment_id}")
        mask = (segments == segment_id) & (foreground_mask > 0)
        if np.sum(mask) == 0:
            #print("continue")
            continue

        coords = np.column_stack(np.nonzero(mask))
        if coords.size == 0:
            #print("continue")
            continue

        center_yx = np.mean(coords, axis=0)# Swap y,x to x,y
        center_xy = (center_yx[1], center_yx[0])
        
        # ys, xs = np.nonzero(foreground_mask)
        # center_y = np.mean(ys)
        # center_x = np.mean(xs)
        # center_xy = (center_x, center_y)
        # center_yx = (center_y, center_x)

        try:
            lower_face, upper_face, extrem = get_projected_box(mask.astype(np.uint8), center_yx, vert_vp, hor_left_vp, hor_right_vp, debug=debug)
            cube_2d = get_projected_cube_faces(lower_face, upper_face)
            lower_face2, upper_face2, extrem = get_internal_box(mask.astype(np.uint8), center_yx, vert_vp, hor_left_vp, hor_right_vp, debug=debug)
            int_2d = get_projected_cube_faces(lower_face2, upper_face2)
            segment_info = SegmentInfo(
                segment_id=segment_id,
                center=center_xy,
                cube_2d=cube_2d,
                int=int_2d,
                extrem = extrem
            )
            segment_info_list.append(segment_info)
        except Exception as e:
             print(f"Segment {segment_id} skipped due to projection error: {e}")

    return segment_info_list

# def get_relative_position(center1, center2):
    # dy = center2[1] - center1[1]
    # dx = center2[0] - center1[0]

    # directions = set()
    # if dy >= 0:
        # directions.add("bottom")
        # directions.add("backward")
    # else:
        # directions.add("top")
        # directions.add("forward")

    # if dx >= 0:
        # directions.add("right")
    # else:
        # directions.add("left")

    # return directions
    
def get_relative_position(center1, center2, right_hor_vp, left_hor_vp, vert_vp):
    center1 = np.array(center1)
    center2 = np.array(center2)
    vec = center2 - center1
    vec_norm = vec / np.linalg.norm(vec)

    # Define unit vectors from center1 toward vanishing points
    dir_x = np.array(right_hor_vp) - center1
    dir_x /= np.linalg.norm(dir_x)

    dir_y = np.array(left_hor_vp) - center1
    dir_y /= np.linalg.norm(dir_y)

    dir_z = np.array(vert_vp) - center1
    dir_z /= np.linalg.norm(dir_z)

    # Compute dot products with each axis direction
    dot_x = np.dot(vec_norm, dir_x)
    dot_y = np.dot(vec_norm, dir_y)
    dot_z = np.dot(vec_norm, dir_z)

    # Find the direction with the largest absolute alignment
    candidates = {
        "right": dot_x,
        "left": -dot_x,
        "forward": dot_y,
        "backward": -dot_y,
        "top": dot_z,
        "bottom": -dot_z
    }

    # Choose the closest direction
    closest_direction = max(candidates.items(), key=lambda x: x[1])[0]
    return [closest_direction]
    
def inverse_direction(dir):
    return {
        "left": "right", "right": "left",
        "top": "bottom", "bottom": "top",
        "forward": "backward", "backward": "forward"
    }.get(dir, dir)


def build_segment_info_graph(segment_info_list, foreground_mask, vert_vp, hor_left_vp, hor_right_vp,
                              weight_func=default_weight, debug=False):
    centers = np.array([s.center for s in segment_info_list])
    graph = defaultdict(list)

    if centers.shape[0] < 2:
        print(f"[Warning] Too few points ({centers.shape[0]}) to build graph.")
        return graph

    elif centers.shape[0] == 2:
        # Directly connect the two segments
        seg_a, seg_b = segment_info_list[0], segment_info_list[1]

        for src, dst in [(seg_a, seg_b), (seg_b, seg_a)]:
            rel_dirs = get_relative_position(src.center, dst.center, hor_right_vp, hor_left_vp, vert_vp)
            #print(f"Relative directions: {rel_dirs}")

            cube_src = src.cube_2d
            cube_dst = dst.cube_2d

            best_iou = 0
            best_direction = None
            best_intersection = None

            for direction in rel_dirs: #["left", "right", "forward", "backward", "top", "bottom"]:
                if direction in face_pairs:
                    face1, face2 = face_pairs[direction]
                    intersection = faces_intersection(cube_src[face1], cube_dst[face2])
                    if intersection is None:
                        continue
                    iou = intersection.area
                    if debug:
                        print(f"Checking {src.segment_id} -> {dst.segment_id} | {direction}: IoU = {iou:.2f}")
                    if iou > best_iou:
                        best_iou = iou
                        best_direction = direction
                        best_intersection = intersection

            if best_iou > 0 and best_direction is not None:
                weight = weight_func(src.center, dst.center)
                src.add_neighbor(dst, weight, best_direction, best_intersection)
                dst.add_neighbor(src, weight, inverse_direction(best_direction), best_intersection)

        return graph

    else:
        try:
            tri = Delaunay(centers)
        except QhullError as e:
            print(f"[QhullError] Delaunay failed: {e}")
            return graph

        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    seg_a = segment_info_list[simplex[i]]
                    seg_b = segment_info_list[simplex[j]]

                    for src, dst in [(seg_a, seg_b), (seg_b, seg_a)]:
                        rel_dirs = get_relative_position(src.center, dst.center, hor_right_vp, hor_left_vp, vert_vp)
                        print(f"Relative directions: {rel_dirs}")

                        cube_src = src.cube_2d
                        cube_dst = dst.cube_2d

                        best_iou = 0
                        best_direction = None
                        best_intersection = None

                        for direction in rel_dirs:#["left", "right", "forward", "backward", "top", "bottom"]:
                            if direction in face_pairs:

                                face1, face2 = face_pairs[direction]
                                intersection = faces_intersection(cube_src[face1], cube_dst[face2])
                                if intersection is None:
                                    print(f"Intersection is None")
                                    #continue
                                iou = intersection.area
                                if debug:
                                    print(f"Checking {src.segment_id} -> {dst.segment_id} | {direction}: IoU = {iou:.2f}")

                                if iou > best_iou:
                                    best_iou = iou
                                    best_direction = direction
                                    best_intersection = intersection
                            else:
                                print("Direction is not in face pairs")

                        if best_iou > 0 and best_direction is not None:
                            weight = weight_func(src.center, dst.center)
                            src.add_neighbor(dst, weight, best_direction, best_intersection)
                            dst.add_neighbor(src, weight, inverse_direction(best_direction), best_intersection)

        return graph

def build_segment_info_graph_tri(segment_info_list, foreground_mask, vert_vp, hor_left_vp, hor_right_vp,
                              weight_func=default_weight, debug=False):
    # segment_info_list = build_segment_info_list(
        # segments, foreground_mask, vert_vp, hor_left_vp, hor_right_vp, debug
    # )

    centers = np.array([s.center for s in segment_info_list])
    
    
    # Defensive check
    if centers.shape[0] < 3:
        print(f"[Warning] Too few points ({centers.shape[0]}) for Delaunay triangulation.")
        return None  # or alternative fallback

    tri = Delaunay(centers)

    graph = defaultdict(list)



    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                seg_a = segment_info_list[simplex[i]]
                seg_b = segment_info_list[simplex[j]]

                for src, dst in [(seg_a, seg_b), (seg_b, seg_a)]:
                    rel_dirs = get_relative_position(src.center, dst.center, hor_right_vp, hor_left_vp, vert_vp)

                    cube_src = src.cube_2d
                    cube_dst = dst.cube_2d

                    best_iou = 0
                    best_direction = None
                    best_intersection = None
    
                    for direction in ["left", "right", "forward", "backward", "top", "bottom"]: #rel_dirs:
                        if direction in face_pairs:
                            face1, face2 = face_pairs[direction]
                            intersection = faces_intersection(cube_src[face1], cube_dst[face2])
                            if intersection is None:
                                continue
                            iou = intersection.area
                            if debug:
                                print(f"Checking {src.segment_id} -> {dst.segment_id} | {direction}: IoU = {iou:.2f}")
                            if iou > best_iou:
                                best_iou = iou
                                best_direction = direction
                                best_intersection = intersection
    
                    if best_iou > 0 and best_direction is not None:
                        weight = weight_func(src.center, dst.center)
                        src.add_neighbor(dst, weight, best_direction, best_intersection)
                        dst.add_neighbor(src, weight, inverse_direction(best_direction), best_intersection)
                        #graph[src].append((dst, weight, best_direction))


    # if debug:
        # print("\nGraph with SegmentInfo nodes and filtered edges with direction:")
        # for seg, edges in graph.items():
            # edge_info = [
                # f"{n.segment_id} (w={w:.2f}, rel='{rel_dir}')"
                # for n, w, rel_dir in edges
            # ]
            # print(f"{seg.segment_id}: {edge_info}")

    return graph

# def build_segment_info_graph_old(segments, foreground_mask, vert_vp, hor_left_vp, hor_right_vp, weight_func=default_weight, debug=False):
    # # Step 1: Build SegmentInfo list
    # segment_info_list = build_segment_info_list(
        # segments, foreground_mask, vert_vp, hor_left_vp, hor_right_vp, debug
    # )

    # # Step 2: Delaunay triangulation on centers
    # centers = np.array([s.center for s in segment_info_list])
    # tri = Delaunay(centers)
    # graph = defaultdict(list)

    # for simplex in tri.simplices:
        # for i in range(3):
            # for j in range(i + 1, 3):
                # seg_i = segment_info_list[simplex[i]]
                # seg_j = segment_info_list[simplex[j]]

                # w = weight_func(seg_i.center, seg_j.center)
                # seg_i.add_neighbor(seg_j, w)
                # seg_j.add_neighbor(seg_i, w)

                # direction_ij = seg_j.center - seg_i.center
                # direction_ji = -direction_ij

                # graph[seg_i].append((seg_j, w, direction_ij))
                # graph[seg_j].append((seg_i, w, direction_ji))

    # if debug:
        # print("Graph with SegmentInfo nodes and relative directions:")
        # for seg, edges in graph.items():
            # edge_info = [
                # f"{n.segment_id} (w={w:.2f}, dir={dir.round(2).tolist()})"
                # for n, w, dir in edges
            # ]
            # print(f"{seg.segment_id}: {edge_info}")

    # return graph

# def dijkstra_segment_graph(start_segment):
    # distances = {start_segment: 0}
    # prev = {}
    # visited = set()
    # queue = [(0, start_segment)]

    # while queue:
        # curr_dist, current = heapq.heappop(queue)
        # if current in visited:
            # continue
        # visited.add(current)

        # for neighbor, weight, _ in current.neighbors:
            # distance = curr_dist + weight
            # if neighbor not in distances or distance < distances[neighbor]:
                # distances[neighbor] = distance
                # prev[neighbor] = current
                # heapq.heappush(queue, (distance, neighbor))

    # return distances, prev


def dijkstra_segment_graph(start_segment):

    distances = {start_segment: 0}
    prev = {}
    visited = set()
    queue = [(0, start_segment)]

    # Initialize 3D cuboid for the start segment
    #start_segment.cub_3d = start_segment.initial_cub_3d  # Set externally before calling this function
    
    

    while queue:
        curr_dist, current = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)

        for neighbor, weight, direction, inter in current.neighbors:
            distance = curr_dist + weight

            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                prev[neighbor] = current
                heapq.heappush(queue, (distance, neighbor))

                # === Compute cub_3d of neighbor based on current.cub_3d and direction ===

                print(f"direction: {direction}")
                if direction in face_pairs:
                    face1, face2 = face_pairs[direction]
                    cube_2d = current.cube_2d
                    cube_3d = current.cube_3d

                    nb_cube_2d = neighbor.cube_2d

                    if direction in ["right", "left"]:
                        pts1 = np.float32(cube_2d[face1])
                        arr = cube_3d[face1].copy()
                        arr = arr[:, 1:]
                        arr[:, [0, 1]] = arr[:, [1, 0]]  # swap y-z -> z-y
                        pts2 = np.float32(arr)
                        ipm = get_projective_transform(pts1, pts2)
    
                        dist_2d = dist(cube_2d[face1], cube_2d[face2])
                        dist_3d = dist(cube_3d[face1], cube_3d[face2])
                        nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                        nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
    
                        mapped = map_points_to_BEV(nb_cube_2d[face2], ipm)
                        mapped = mapped[:, [1, 0]]  # swap z-y
                        if direction == "right": #OK
                            l_3d = np.column_stack((np.full(mapped.shape[0], cube_3d[face1][0, 0]), mapped))
                            r_3d = l_3d.copy()
                            r_3d[:, 0] += nb_dist_3d
                            
                        elif direction == "left":#OK
                            r_3d = np.column_stack((np.full(mapped.shape[0], cube_3d[face1][0, 0]), mapped))
                            l_3d = r_3d.copy()
                            l_3d[:, 0] -= nb_dist_3d

                        upper_3d = np.array([l_3d[1], r_3d[1], r_3d[2], l_3d[2]])
                        lower_3d = np.array([l_3d[0], r_3d[0], r_3d[3], l_3d[3]])


                    if direction in ["forward", "backward"]:
                        pts1 = np.float32(cube_2d[face1])
                        arr = cube_3d[face1].copy()
                        arr = arr[:, [0, 2]]  # remove Y 
                        arr[:, [0, 1]] = arr[:, [1, 0]]  # swap x-z -> z-x
                        pts2 = np.float32(arr)
                        ipm = get_projective_transform(pts1, pts2)
                        
                        dist_2d = dist(cube_2d[face1], cube_2d[face2])
                        dist_3d = dist(cube_3d[face1], cube_3d[face2])
                        nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                        nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0
    
                        mapped = map_points_to_BEV(nb_cube_2d[face2], ipm)
                        mapped = mapped[:, [1, 0]]  # swap z-x
                        
                        if direction == "forward":
                            l_3d = np.column_stack((
                                mapped[:, 0],                                   # x
                                np.full(mapped.shape[0], cube_3d[face1][0, 1]),  # FIX y
                                mapped[:, 1]                                    # Z
                            ))
                            r_3d = l_3d.copy()
                            r_3d[:, 1] += nb_dist_3d
                        
                        elif direction == "backward":
                            r_3d = np.column_stack((
                                mapped[:, 0],                                   # x
                                np.full(mapped.shape[0], cube_3d[face1][0, 1]),  # FIX y
                                mapped[:, 1]                                    # Z
                            ))
                            l_3d = r_3d.copy()
                            l_3d[:, 1] -= nb_dist_3d   # Move along Z

                        upper_3d = np.array([l_3d[1], l_3d[2], r_3d[2], r_3d[1]])
                        lower_3d = np.array([l_3d[0], l_3d[3], r_3d[3], r_3d[0]])

                    elif direction == "top":
                        pts1 = np.float32(cube_2d[face1])
                        arr = cube_3d[face1].copy()[:, :2]
                        pts2 = np.float32(arr)
                        ipm = get_projective_transform(pts1, pts2)

                        dist_2d = dist(cube_2d[face1], cube_2d[face2])
                        dist_3d = dist(cube_3d[face1], cube_3d[face2])
                        nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                        nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0

                        mapped = map_points_to_BEV(nb_cube_2d[face2], ipm)
                        lower_3d = np.column_stack((mapped.copy(), np.full(mapped.shape[0], cube_3d[face1][0, 2])))
                        upper_3d = lower_3d.copy()
                        upper_3d[:, 2] += nb_dist_3d
                    # OK
                    elif direction == "bottom":
                        pts1 = np.float32(cube_2d[face1])
                        arr = cube_3d[face1].copy()[:, :2]
                        pts2 = np.float32(arr)
                        ipm = get_projective_transform(pts1, pts2)

                        dist_2d = dist(cube_2d[face1], cube_2d[face2])
                        dist_3d = dist(cube_3d[face1], cube_3d[face2])
                        nb_dist_2d = dist(nb_cube_2d[face1], nb_cube_2d[face2])
                        nb_dist_3d = (dist_3d / dist_2d) * nb_dist_2d if dist_2d != 0 else 0

                        mapped = map_points_to_BEV(nb_cube_2d[face2], ipm)
                        upper_3d = np.column_stack((mapped.copy(), np.full(mapped.shape[0], cube_3d[face1][0, 2])))
                        lower_3d = upper_3d.copy()
                        lower_3d[:, 2] -= nb_dist_3d

                    # Assign cub_3d to neighbor
                    neighbor.cube_3d = get_projected_cube_faces(lower_3d, upper_3d)

    return distances, prev

def reconstruct_path(prev, target_segment):
    path = []
    current = target_segment
    while current in prev:
        path.append(current)
        current = prev[current]
    path.append(current)  # start node
    return list(reversed(path))

def find_segment_for_point(segments, point):

    x, y = int(point[0]), int(point[1])

    # Bounds check
    if x < 0 or y < 0 or y >= segments.shape[0] or x >= segments.shape[1]:
        return None

    return int(segments[y, x])
    
def is_internal_segment(segments, segment_id):
    """
    Returns True if the segment is internal (not touching image border).
    
    Parameters:
        segments (ndarray): 2D array with segment labels.
        segment_id (int): Label of the segment to check.
    
    Returns:
        bool: True if internal, False if on border.
    """
    mask = (segments == segment_id)
    h, w = segments.shape

    # Check all four borders
    if np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
        return False  # touches the border
    return True 
 
    
# def build_segment_info_graph_old(segment_infos, tri, weight_func=default_weight, debug=False):
    # graph = defaultdict(list)

    # # Map from center index to SegmentInfo
    # index_to_segment = {i: segment_infos[i] for i in range(len(segment_infos))}
    # centers = [s.center for s in segment_infos]

    # indptr, indices = tri.vertex_neighbor_vertices

    # for i in range(len(indptr) - 1):
        # neighbors = indices[indptr[i]:indptr[i + 1]]
        # for j in neighbors:
            # seg_i = index_to_segment[i]
            # seg_j = index_to_segment[j]
            # w = weight_func(seg_i.center, seg_j.center)
            # direction_ij = seg_j.center - seg_i.center
            # direction_ji = -direction_ij  # Reverse direction

            # seg_i.neighbors.append((seg_j, w, direction_ij))
            # seg_j.neighbors.append((seg_i, w, direction_ji))

            # graph[seg_i].append((seg_j, w, direction_ij))
            # graph[seg_j].append((seg_i, w, direction_ji))  # undirected

    # if debug:
        # print("Graph with SegmentInfo nodes and relative directions:")
        # for seg, edges in graph.items():
            # edge_info = [
                # f"{n.segment_id} (w={w:.2f}, dir={dir.round(2).tolist()})"
                # for n, w, dir in edges
            # ]
            # print(f"{seg.segment_id}: {edge_info}")
    
    # return graph
    
    
# def find_avg_path_distance_segment(graph, start_segment, end_segment):
    # """
    # Dijkstra on SegmentInfo nodes.
    # """
    # queue = [(0, start_segment)]
    # visited = set()
    # distances = {start_segment: 0}

    # while queue:
        # current_dist, node = heapq.heappop(queue)

        # if node in visited:
            # continue
        # visited.add(node)

        # if node == end_segment:
            # return current_dist

        # for neighbor, weight, direction in graph.get(node, []):
            # if neighbor in visited:
                # continue
            # new_dist = current_dist + weight
            # if new_dist < distances.get(neighbor, np.inf):
                # distances[neighbor] = new_dist
                # heapq.heappush(queue, (new_dist, neighbor))

    # return float('inf')

# def find_avg_path_distance_segment_all(graph, start_segment, end_segment, max_depth=10**6):
    # """
    # Find the average distance of all simple (no-cycle) paths from start_segment to end_segment.
    # Uses DFS with path tracking.
    # """
    # all_distances = []

    # def dfs(current, target, visited, current_distance):
        # if current == target:
            # all_distances.append(current_distance)
            # return
        # if len(visited) > max_depth:
            # return  # prevent runaway recursion

        # for neighbor, weight, _ in graph.get(current, []):
            # if neighbor in visited:
                # continue
            # dfs(neighbor, target, visited | {neighbor}, current_distance + weight)

    # dfs(start_segment, end_segment, {start_segment}, 0)

    # if not all_distances:
        # return float('inf')
    # return sum(all_distances) / len(all_distances)
