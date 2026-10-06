import numpy as np
import heapq
from collections import defaultdict, deque


def find_all_paths(graph, start, end, max_depth=6, max_weight=np.inf):
    """
    Returns all paths from `start` to `end` with depth and weight constraints.
    """
    paths = []
    
    def dfs(current, path, visited, total_weight):
        if current == end:
            paths.append((list(path), total_weight))
            return
        if len(path) >= max_depth:
            return
        for neighbor, weight in graph.get(current, []):
            if neighbor not in visited and total_weight + weight <= max_weight:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, path, visited, total_weight + weight)
                path.pop()
                visited.remove(neighbor)
    
    dfs(start, [start], {start}, 0)
    return paths

def compute_avg_path_distances(graph, valid_ids, segment_labels, max_depth=6, max_weight=np.inf, debug=False):
    """
    valid_ids: list of segment labels (like 1, 2, 3)
    segment_labels: list of corresponding segment IDs in same order as centers and graph nodes (e.g., [5, 7, 9])
    graph: uses center indices (0-based) as keys
    """
    # Map segment label → graph node index
    label_to_idx = {label: idx for idx, label in enumerate(segment_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    avg_distances = {}

    for src_label in valid_ids:
        for dst_label in valid_ids:
            if src_label == dst_label:
                continue

            src = label_to_idx.get(src_label, None)
            dst = label_to_idx.get(dst_label, None)

            if src is None or dst is None:
                avg_distances[(src_label, dst_label)] = float('inf')
                continue

            all_paths = find_all_paths(graph, src, dst, max_depth=max_depth, max_weight=max_weight)
            if not all_paths:
                avg_distances[(src_label, dst_label)] = float('inf')
                if debug:
                    print(f"No path from {src_label} to {dst_label}")
                continue

            total = sum(w for _, w in all_paths)
            avg = total / len(all_paths)
            avg_distances[(src_label, dst_label)] = avg

            if debug:
                print(f"{len(all_paths)} paths from {src_label} to {dst_label}, avg = {avg:.3f}")
    
    return avg_distances


def build_graph(centers, tri, weight_func, debug=False):
    indptr, indices = tri.vertex_neighbor_vertices
    graph = defaultdict(list)

    for i in range(len(indptr) - 1):
        neighbors = indices[indptr[i]:indptr[i + 1]]
        for j in neighbors:
            w = weight_func(centers[i], centers[j])
            graph[i].append((j, w))
            graph[j].append((i, w))  # undirected

    if debug:
        print("Graph structure:")
        for k, v in graph.items():
            print(f"{k}: {v}")
    return graph

def default_weight(a, b):
    return float(np.linalg.norm(a - b))

def dfs(graph, start=0, debug=False):
    paths = defaultdict(list)
    stack = [(start, 0.0, [start])]

    while stack:
        node, cost, path = stack.pop()
        paths[node].append(cost)

        for neighbor, w in graph[node]:
            if neighbor not in path:  # avoid cycles
                stack.append((neighbor, cost + w, path + [neighbor]))

    avg = {v: sum(c) / len(c) for v, c in paths.items()}
    if debug:
        print("DFS average path cost to each vertex:", avg)
    return avg

def bfs(graph, start=0, debug=False):
    paths = defaultdict(list)
    queue = deque([(start, 0.0, [start])])

    while queue:
        node, cost, path = queue.popleft()
        paths[node].append(cost)

        for neighbor, w in graph[node]:
            if neighbor not in path:
                queue.append((neighbor, cost + w, path + [neighbor]))

    avg = {v: sum(c) / len(c) for v, c in paths.items()}
    if debug:
        print("BFS average path cost to each vertex:", avg)
    return avg

def dijkstra(graph, start=0, debug=False):
    paths = defaultdict(list)
    queue = [(0.0, start, [start])]

    while queue:
        cost, node, path = heapq.heappop(queue)
        paths[node].append(cost)

        for neighbor, w in graph[node]:
            if neighbor not in path:
                heapq.heappush(queue, (cost + w, neighbor, path + [neighbor]))

    avg = {v: sum(c) / len(c) for v, c in paths.items()}
    if debug:
        print("Dijkstra average path cost to each vertex:", avg)
    return avg



def get_relative_position(center1, center2):
    """
    Get direction(s) of center2 relative to center1.

    Args:
        center1 (tuple): (y, x)
        center2 (tuple): (y, x)

    Returns:
        set[str]: Direction set: {"top", "bottom", "left", "right"}
    """
    dy = center2[0] - center1[0]
    dx = center2[1] - center1[1]

    directions = set()
    if dy <= 0:
        directions.add("top")
    elif dy > 0:
        directions.add("bottom")
    if dx <= 0:
        directions.add("left")
    elif dx > 0:
        directions.add("right")

    return directions
    
def build_neighbors_from_graph(graph, centers, debug=False):
    """
    Construct neighbors_dict with directional info from a graph and centers.

    Args:
        graph (dict): {node: [(neighbor, weight), ...]} representation.
        centers (dict): {node: (y, x)} center coordinates for each node.
        debug (bool): If True, print debug information.

    Returns:
        dict: neighbors_dict[node1][node2] = direction set (e.g., {"left", "top"})
    """
    neighbors_dict = {node: {} for node in graph}

    for node, neighbors in graph.items():
        for neighbor, _ in neighbors:
            if neighbor not in neighbors_dict[node]:
                dir_to_neighbor = get_relative_position(centers[node], centers[neighbor])
                dir_to_node = get_relative_position(centers[neighbor], centers[node])
                neighbors_dict[node][neighbor] = dir_to_neighbor
                neighbors_dict[neighbor][node] = dir_to_node

    if debug:
        print("Constructed neighbors_dict from graph:")
        for node, neighbors in neighbors_dict.items():
            for neighbor, directions in neighbors.items():
                print(f"{node} → {neighbor}: {directions}")

    return neighbors_dict
    
def find_avg_path_distance(graph, start_idx, end_idx, max_weight=np.inf, max_depth=np.inf):
    """
    Find the average path distance from start_idx to end_idx using Dijkstra.
    """
    queue = [(0, start_idx)]
    visited = set()
    distances = {start_idx: 0}

    while queue:
        current_dist, node = heapq.heappop(queue)

        if node in visited:
            continue
        visited.add(node)

        if node == end_idx:
            return current_dist  # Found shortest path

        if current_dist > max_weight:
            break

        for neighbor, weight in graph.get(node, []):
            if neighbor in visited:
                continue
            distance = current_dist + weight
            if distance < distances.get(neighbor, np.inf):
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))

    return float('inf')  # No path found
