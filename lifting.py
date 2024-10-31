import numpy as np

def projection_on_bird(M, p, float_type=True):
    """
    Projects a 2D point `p` onto the bird's-eye view using the inverse perspective matrix `M`.

    Args:
    - p: A 2D point (x, y) as a list, tuple, or array.
    - M: A 3x3 inverse perspective matrix.

    Returns:
    - px: The projected 2D point in the BEV space.
    """
    # Convert p to a NumPy array if it's not already
    if isinstance(p, (list, tuple)):
        p = np.array(p)
    # Ensure p has two dimensions (x, y)
    if p.shape[0] != 2:
        raise ValueError("Input point p should be a 2D point with two elements.")

    # Ensure M is a 3x3 matrix
    if M.shape != (3, 3):
        raise ValueError(f"Matrix M should be 3x3, but got shape {M.shape}.")

    # Apply the inverse perspective mapping formula (homogeneous transformation)
    px = (M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2]) / (M[2, 0] * p[0] + M[2, 1] * p[1] + M[2, 2])
    py = (M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2]) / (M[2, 0] * p[0] + M[2, 1] * p[1] + M[2, 2])

    if float_type: return np.array([px, py], dtype=np.float32)
    return np.array([int(px), int(py)], dtype=np.int32)

def convrt2Bird(transferI2B, img, shape):
    return cv2.warpPerspective(img, transferI2B, (shape[1], shape[0]))

def convrt2Image(transferB2I, bird, shape):
    return cv2.warpPerspective(bird, self.transferB2I, (shape[1], shape[0]))

def move_to_second_quadrant(angle):
    """
    Normalizes the input angle and moves it to the second quadrant.
    The second quadrant corresponds to the range [π/2, π] radians.
    
    Parameters:
    - angle: The input angle in radians.

    Returns:
    - The adjusted angle in the second quadrant.
    """
    # Normalize the angle to the range [0, 2π)
    normalized_angle = angle % (2 * np.pi)
    if normalized_angle < 0:
        normalized_angle += 2 * np.pi

    # Move the angle to the second quadrant
    if 0 < normalized_angle < np.pi/2:  # First quadrant
        return np.pi/2 + normalized_angle
    elif np.pi/2 < normalized_angle < np.pi:  # Already in second quadrant
        return normalized_angle
    elif np.pi < normalized_angle < 3*np.pi/2:  # Third quadrant
        return normalized_angle - np.pi/2
    elif  3*np.pi/2 < normalized_angle < 2*np.pi: # Fourth quadrant
        return normalized_angle - np.pi
    else:
        return normalized_angle

        # 0 1
        # 3 2

# Convert radians to degrees
def rad_to_deg(angle_in_radians):
    """Convert radians to degrees."""
    return np.degrees(angle_in_radians)

# Compute the internal angle between two vectors
def compute_internal_angle(v1, v2, debug=False):
    """Compute the internal angle between two vectors v1 and v2."""
    dot_product = np.dot(v1, v2)
    cross_product = np.cross(v1, v2)
    
    # Compute the angle in radians
    angle = np.arctan2(np.linalg.norm(cross_product), dot_product)
    
    # Ensure the angle is within the range [0, pi] for internal angles
    if angle > np.pi:
        angle = 2 * np.pi - angle

    if debug:
        print(f"compute_internal_angle: v1 = {v1}, v2 = {v2}")
        print(f"dot_product = {dot_product}, cross_product = {cross_product}")
        print(f"angle = {angle} radians ({rad_to_deg(angle)} degrees)")
    
    return angle

        # 0 1
        # 3 2
# Compute the orientation based on a quadrangle
def compute_orientation(quadrangle, l, w, debug=False):
    # Ensure that quadrangle has at least 4 points
    assert len(quadrangle) >= 4, "Quadrangle must have at least 4 points."
    
    # Compute d, which is the distance between the last two points of the quadrangle
    d = np.linalg.norm(quadrangle[0] - quadrangle[3])
    
    # Compute alpha: internal angle at quadrangle[-1]
    #
    v1 = quadrangle[3] - quadrangle[0]
    v2 = quadrangle[1] - quadrangle[0]
    alpha = compute_internal_angle(v1, v2, debug=debug)
    
    # Compute beta: internal angle at quadrangle[3]
    v1 = quadrangle[2] - quadrangle[3]
    v2 = quadrangle[0] - quadrangle[3]
    beta = compute_internal_angle(v1, v2, debug=debug)
    
    # Step 1: Compute epsilon, phi, and gamma
    epsilon = np.sin(alpha) * (l * np.sin(beta) + w * np.cos(beta))
    phi = np.sin(beta) * (l * np.cos(alpha) + w * np.sin(alpha))
    gamma = d * np.sin(alpha) * np.sin(beta)
    
    if debug:
        print(f"compute_orientation: d = {d}")
        print(f"alpha = {alpha} radians ({rad_to_deg(alpha)} degrees), beta = {beta} radians ({rad_to_deg(beta)} degrees)")
        print(f"epsilon = {epsilon}, phi = {phi}, gamma = {gamma}")

    # Step 2: Compute theta and psi
    psi = np.arccos(np.clip(gamma / np.sqrt(epsilon**2 + phi**2), -1.0, 1.0))
    theta = np.arctan2(phi, epsilon)

    if debug:
        print(f"psi = {psi} radians ({rad_to_deg(psi)} degrees), theta = {theta} radians ({rad_to_deg(theta)} degrees)")
        
    # Step 3: Compute varphi (orientation)
    varphi = [theta - psi, theta + psi]  # Return both -psi and +psi variants

    if debug:
        print(f"varphi = {np.rad2deg(varphi)} degrees")
    
    return varphi

        # 0 1
        # 3 2
    
def compute_angles(quadrangle, debug=False):
    # Ensure that quadrangle has at least 4 points
    assert len(quadrangle) >= 4, "Quadrangle must have at least 4 points."
    
    # Compute alpha: internal angle at quadrangle[-1]
    #
    v1 = quadrangle[3] - quadrangle[0]
    v2 = quadrangle[1] - quadrangle[0]
    alpha = compute_internal_angle(v1, v2, debug=debug)
    
    # Compute beta: internal angle at quadrangle[-2]
    v1 = quadrangle[2] - quadrangle[3]
    v2 = quadrangle[0] - quadrangle[3]
    beta = compute_internal_angle(v1, v2, debug=debug)
    
    # Compute d, which is the distance between the last two points of the quadrangle
    d = np.linalg.norm(quadrangle[0] - quadrangle[3])
    
 
    if debug:
        print(f"Lower alpha = {rad_to_deg(alpha):.2f} degrees")
        print(f"Lower beta = {rad_to_deg(beta):.2f} degrees")
        print(f"Lower Distance = {d:.2f}")

    return d, alpha, beta


        # 0 1
        # 3 2
def compute_upper_angles(quadrangle, debug=False):
    """
    Compute the internal angles at the upper two corners of a quadrangle.
    
    Parameters:
    - quadrangle: A list of at least 4 points representing the quadrangle in 2D space.
    
    Returns:
    - d: Distance between the first two points of the quadrangle (side length).
    - alpha: Internal angle at the last point (quadrangle[-1]).
    - beta: Internal angle at the second-to-last point (quadrangle[-2]).
    """
    # Ensure that the quadrangle has at least 4 points
    assert len(quadrangle) >= 4, "Quadrangle must have at least 4 points."
    
    # Use named points for clarity
    P1, P2, P3, P4 = quadrangle[0], quadrangle[1], quadrangle[2], quadrangle[3] #quadrangle[-4], quadrangle[-3], quadrangle[-2], quadrangle[-1]
    
    # Compute alpha: internal angle at P2 (quadrangle[-1])
    v1 = P3 - P2  # Vector from P4 to P3
    v2 = P1 - P2  # Vector from P4 to P1
    alpha = compute_internal_angle(v1, v2, debug=debug)
    
    # Compute beta: internal angle at P3 (quadrangle[-2])
    v1 = P2 - P3  # Vector from P3 to P2
    v2 = P4 - P3  # Vector from P3 to P4
    beta = compute_internal_angle(v1, v2, debug=debug)
    
    # Compute d: distance between the first two points of the quadrangle (side length)
    d = np.linalg.norm(P2 - P3)
    
    if debug:
        print(f"Upper alpha = {alpha:.2f} radians ({rad_to_deg(alpha):.2f} degrees)")
        print(f"Upper beta = {beta:.2f} radians ({rad_to_deg(beta):.2f} degrees)")
        print(f"Upper Distance = {d:.2f}")
    
    return d, alpha, beta

        # 0 1
        # 3 2
# Compute global orientation from a quadrangle
def compute_global_orientation(quadrangle, l, w, debug=False):
    # Step 1: Compute the local orientation angle (relative to the reference edge)
    local_varphi = compute_orientation(quadrangle, l, w, debug=debug)
    
    # Step 2: Compute the global angle of the reference edge (quadrangle[-1], quadrangle[-2])
    # TODO check angle orientation -2 -1 
    reference_edge = quadrangle[3] - quadrangle[0]
    global_reference_angle = np.arctan2(reference_edge[1], reference_edge[0])
    
    # Step 3: Compute the global orientation angle
    global_varphi = [global_reference_angle + np.pi - varphi for varphi in local_varphi]
    
    # Normalize to range [0, 2*pi]
    global_varphi = [angle % (2 * np.pi) for angle in global_varphi]
    
    if debug:
        print(f"compute_global_orientation: reference_edge = {reference_edge}")
        print(f"global_reference_angle = {global_reference_angle} radians ({rad_to_deg(global_reference_angle)} degrees)")
        print(f"global_varphi (before normalization) = {[rad_to_deg(angle) for angle in global_varphi]} degrees")
    
    return global_varphi

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

# Function to compute the center of the bounding rectangle
def compute_bottom_rectangle_center(q, orientation, a, b, debug=False):
    # Extract the four vertices of the quadrangle
    # TODO arrange
    A, R, T, B = q[0], q[1], q[2], q[3] #q[3], q[0], q[1], q[2]
    
    # Step 1: Draw a ray from point B with the given orientation angle
    # Compute the direction vector for the orientation
    orientation_vector = np.array([np.cos(orientation), np.sin(orientation)])
    ray_endpoint = B + orientation_vector * 1000  # Extend the ray far enough
    
    # Step 2: Find the intersection point E of the ray with the line AR
    E = get_intersect(B, ray_endpoint, A, R)
    if E is None:
        if debug: print("No intersection found (E is None).")
        return None
    
    # Step 3: Compute the distance l from B to E (l = EB)
    l = np.linalg.norm(B - E)
    
    # Step 4: Compute the position of point C using the formula for similar triangles
    # C_x = (l - a) * A[0] + a * B[0]
    # C_y = (l - a) * A[1] + a * B[1]
    C = ((l - a) * A + a * B)/l
    #C = np.array([C_x, C_y]) / l
    
    # Step 5: Compute point K, which lies on the segment AB with length KC = a
    #K = A + (C - A) * a / np.linalg.norm(C - A)
    K = get_intersect(C, C + orientation_vector * 1000, A, R)
    
    # Step 6: Compute point D, which lies on the perpendicular from C with length CD = b
    # The perpendicular direction can be found by rotating the orientation vector by 90 degrees
    perpendicular_vector = np.array([-orientation_vector[1], orientation_vector[0]])
    D = C + perpendicular_vector * b
    
    # Step 7: Compute the center of the rectangle formed by K, C, and D
    center = (K + D) / 2
    
    # Debugging output
    if debug:
        print(f"A: {A}, B: {B}, R: {R}, T: {T}")
        print(f"Orientation vector: {orientation_vector}")
        print(f"Intersection point E: {E}")
        print(f"Distance l: {l}")
        print(f"Point C: {C}")
        print(f"Point K: {K}")
        print(f"Point D: {D}")
        print(f"Center of the bounding rectangle: {center}")
    
    return center
    
def compute_bottom_rectangle(q, orientation, a, b, debug=False):
    # Extract the four vertices of the quadrangle
    # TODO arrange
    #assert orientation < np.pi/2
    A, R, T, B = q[0], q[1], q[2], q[3] #q[3], q[0], q[1], q[2]
    
    # Step 1: Draw a ray from point B with the given orientation angle
    # Compute the direction vector for the orientation
    orientation_vector = np.array([np.cos(orientation), np.sin(orientation)])
    ray_endpoint = B + orientation_vector * 1000  # Extend the ray far enough
    
    # Step 2: Find the intersection point E of the ray with the line AR
    E = get_intersect(B, ray_endpoint, A, R)
    if E is None:
        if debug: print("No intersection found (E is None).")
        return None
    
    # Step 3: Compute the distance l from B to E (l = EB)
    l = np.linalg.norm(B - E)
    
    # Step 4: Compute the position of point C using the formula for similar triangles
    # C_x = (l - a) * A[0] + a * B[0]
    # C_y = (l - a) * A[1] + a * B[1]
    C = ((l - a) * A + a * B)/l
    #C = np.array([C_x, C_y]) / l
    
    # Step 5: Compute point K, which lies on the segment AB with length KC = a
    #K = A + (C - A) * a / np.linalg.norm(C - A)
    K = get_intersect(C, C + orientation_vector * 1000, A, R)
    
       # Step 6: Compute point D, which lies on the perpendicular from C with length CD = b
    # The perpendicular direction can be found by rotating the orientation vector by 90 degrees
    perpendicular_vector = np.array([orientation_vector[1], -orientation_vector[0]])
    
    # Compute the centroid of the quadrangle
    centroid = np.mean([A, R, T, B], axis=0)
    
    # Check if the perpendicular vector points towards the inside of the quadrangle
    centroid_direction = centroid - C
    if np.dot(perpendicular_vector, centroid_direction) < 0:
        # Reverse the perpendicular vector if it points outward
        perpendicular_vector = -perpendicular_vector
    
    # Now you can use the adjusted perpendicular_vector for further calculations
    # Example: computing point D as C + b * perpendicular_vector
    D = C + b * perpendicular_vector
    #D = np.array(get_intersect(C, C + perpendicular_vector * 1000, T, B), dtype=np.float32)
    
    # Step 7: Compute the center of the rectangle formed by K, C, and D
    center = (K + D) / 2
    T = K + (D - C)
    
    # Debugging output
    if debug:
        print(f"A: {A}, B: {B}, R: {R}, T: {T}")
        print(f"Orientation vector: {orientation_vector}")
        print(f"Intersection point E: {E}")
        print(f"Distance l: {l}")
        print(f"Point C: {C}")
        print(f"Point K: {K}")
        print(f"Point T: {T}")
        print(f"Point D: {D}")
        print(f"Center of the bounding rectangle: {center}")
    
    return np.array([center, C, K, T, D], dtype=np.float64)


def compute_bottom_rectangle_with(q, orientation, C, debug=False):
    # Extract the four vertices of the quadrangle
    # TODO arrange
    #assert orientation < np.pi/2
    A, R, T, B = q[0], q[1], q[2], q[3] #q[3], q[0], q[1], q[2]
    
    # Step 1: Draw a ray from point B with the given orientation angle
    # Compute the direction vector for the orientation
    orientation_vector = np.array([np.cos(orientation), np.sin(orientation)])
    ray_endpoint = B + orientation_vector * 1000  # Extend the ray far enough
    
    K = get_intersect(C, C + orientation_vector * 1000, A, R)
    if K is None or K[1] > R[1]:
        K = R #get_intersect(C, C + orientation_vector * 1000, T, R)
    if K[1] < A[1]: K = A
    
       # Step 6: Compute point D, which lies on the perpendicular from C with length CD = b
    # The perpendicular direction can be found by rotating the orientation vector by 90 degrees
    perpendicular_vector = np.array([orientation_vector[1], -orientation_vector[0]])
    
    # Compute the centroid of the quadrangle
    centroid = np.mean([A, R, T, B], axis=0)
    
    # Check if the perpendicular vector points towards the inside of the quadrangle
    centroid_direction = centroid - C
    if np.dot(perpendicular_vector, centroid_direction) < 0:
        # Reverse the perpendicular vector if it points outward
        perpendicular_vector = -perpendicular_vector
    
    # Now you can use the adjusted perpendicular_vector for further calculations
    # Example: computing point D as C + b * perpendicular_vector
    #D = C + b * perpendicular_vector
    D = get_intersect(C, C + perpendicular_vector * 1000, T, B)
    if D is None: D = T
    D = np.array(D, dtype=np.float32)
    if D[1] > T[1]:
        D = T #np.array(get_intersect(C, C + perpendicular_vector * 1000, T, B), dtype=np.float32)
    if D[1] < B[1]: D = B
    
    # Step 7: Compute the center of the rectangle formed by K, C, and D
    center = (K + D) / 2
    T = K + (D - C)
    
    # Debugging output
    if debug:
        print(f"A: {A}, B: {B}, R: {R}, T: {T}")
        print(f"Orientation vector: {orientation_vector}")
        # print(f"Intersection point E: {E}")
        # print(f"Distance l: {l}")
        print(f"Point C: {C}")
        print(f"Point K: {K}")
        print(f"Point T: {T}")
        print(f"Point D: {D}")
        print(f"Center of the bounding rectangle: {center}")
    
    return np.array([center, C, K, T, D], dtype=np.float64)

def compute_top_rectangle(q, orientation, a, b, debug=False):
    # Extract the four vertices of the quadrangle
    # TODO arrange
    A, R, T, B = q[0], q[1], q[2], q[3]# q[3], q[0], q[1], q[2]
    
    # Step 1: Draw a ray from point B with the given orientation angle
    # Compute the direction vector for the orientation
    orientation_vector = np.array([np.cos(orientation), np.sin(orientation)])
    ray_endpoint = T + orientation_vector * 1000  # Extend the ray far enough
    
    # Step 2: Find the intersection point E of the ray with the line AR
    E = get_intersect(T, ray_endpoint, A, R)
    if E is None:
        if debug: print("No intersection found (E is None).")
        return None
    
    # Step 3: Compute the distance l from B to E (l = EB)
    l = np.linalg.norm(T - E)
    
    # Step 4: Compute the position of point C using the formula for similar triangles
    # C_x = (l - a) * A[0] + a * B[0]
    # C_y = (l - a) * A[1] + a * B[1]
    C = ((l - a) * R + a * T)/l
    #C = np.array([C_x, C_y]) / l
    
    # Step 5: Compute point K, which lies on the segment AB with length KC = a
    #K = A + (C - A) * a / np.linalg.norm(C - A)
    K = get_intersect(C, C + orientation_vector * 1000, A, R)
    
    # Step 6: Compute point D, which lies on the perpendicular from C with length CD = b
    # The perpendicular direction can be found by rotating the orientation vector by 90 degrees
    perpendicular_vector = np.array([orientation_vector[1], -orientation_vector[0]])
    D = C - perpendicular_vector * b
    
    # Step 7: Compute the center of the rectangle formed by K, C, and D
    center = (K + D) / 2
    T = K + (D - C)
    
    # Debugging output
    if debug:
        print(f"A: {A}, B: {B}, R: {R}, T: {T}")
        print(f"Orientation vector: {orientation_vector}")
        print(f"Intersection point E: {E}")
        print(f"Distance l: {l}")
        print(f"Point C: {C}")
        print(f"Point K: {K}")
        print(f"Point T: {T}")
        print(f"Point D: {D}")
        print(f"Center of the bounding rectangle: {center}")
    
    return np.array([center, C, K, T, D], dtype=np.float64)
    
def compute_top_rectangle_with(q, orientation, C, debug=False):
    # Extract the four vertices of the quadrangle
    # TODO arrange
    A, R, T, B = q[0], q[1], q[2], q[3]# q[3], q[0], q[1], q[2]
    
    # Step 1: Draw a ray from point B with the given orientation angle
    # Compute the direction vector for the orientation
    orientation_vector = np.array([np.cos(orientation), np.sin(orientation)])
    ray_endpoint = T + orientation_vector * 1000  # Extend the ray far enough
    
    K = get_intersect(C, C + orientation_vector * 1000, A, R)
    if K is None or K[1] < A[1]: K = A
    if K[1] > R[1]: K = R
    
    # Step 6: Compute point D, which lies on the perpendicular from C with length CD = b
    # The perpendicular direction can be found by rotating the orientation vector by 90 degrees
    perpendicular_vector = np.array([orientation_vector[1], -orientation_vector[0]])

    # Compute the centroid of the quadrangle
    centroid = np.mean([A, R, T, B], axis=0)
    
    # Check if the perpendicular vector points towards the inside of the quadrangle
    centroid_direction = centroid - C
    if np.dot(perpendicular_vector, centroid_direction) < 0:
        # Reverse the perpendicular vector if it points outward
        perpendicular_vector = -perpendicular_vector

    #D = C - perpendicular_vector * b
    D = get_intersect(C, C + perpendicular_vector * 1000, T, B)
    if D is None: D = B
    D = np.array(D, dtype=np.float32)
    if D[1] < B[1]: D = B
    if D[1] > T[1]: D = T
    
    # Step 7: Compute the center of the rectangle formed by K, C, and D
    center = (K + D) / 2
    T = K + (D - C)
    
    # Debugging output
    if debug:
        print(f"A: {A}, B: {B}, R: {R}, T: {T}")
        print(f"Orientation vector: {orientation_vector}")
        print(f"Point C: {C}")
        print(f"Point K: {K}")
        print(f"Point T: {T}")
        print(f"Point D: {D}")
        print(f"Center of the bounding rectangle: {center}")
    
    return np.array([center, T, K, C, D], dtype=np.float64)



def compute_rectangle_points(center, angle, width, length):
    """
    Compute the four corner points of a rectangle based on its center, orientation angle, width, and length.
    
    Args:
    - center: tuple or np.array of shape (2,), the (x, y) coordinates of the rectangle's center.
    - angle: float, the orientation angle of the rectangle in radians.
    - width: float, the width of the rectangle.
    - length: float, the length of the rectangle.
    
    Returns:
    - points: np.array of shape (4, 2), the four corner points of the rectangle.
    """
    # Half width and length
    half_width = width / 2
    half_length = length / 2

    # Define the rectangle in its local coordinate system (before rotation)
    local_points = np.array([
        [-half_length, -half_width],  # Bottom-left
        [half_length, -half_width],   # Bottom-right
        [half_length, half_width],    # Top-right
        [-half_length, half_width]    # Top-left
    ])

    # Rotation matrix for the given angle
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    # Rotate and translate the points to global coordinates
    global_points = np.dot(local_points, rotation_matrix.T) + center
    
    # Find the index of the lower-left corner (lowest y, then lowest x if tied)
    lower_left_index = np.lexsort((global_points[:, 0], global_points[:, 1]))[0]


    return global_points, lower_left_index

# Calculate the bottom center of a quadrangle based on orientation
def get_bottom_center(q, orient, length, e=10, debug=False):
    # TODO fix order
    # Reorder quadrangle to start from q[3] and proceed counterclockwise
    quadrangle = [q[0], q[1], q[2], q[3]] #q[3], q[0], q[1], q[2]]

    # Step 1: Find intersection point `k` of the line through quadrangle[3] in the orientation direction with the line quadrangle[0]->quadrangle[1]
    k = get_intersect(quadrangle[3], quadrangle[3] + orient, quadrangle[0], quadrangle[1])
    if k is None:
        if debug: print("No intersection found (k is None).")
        return None

    # Step 2: Compute the length `l` from quadrangle[3] to `k`
    l = np.linalg.norm(quadrangle[3] - k)
    if l == 0:
        l = 0.0001  # Avoid division by zero

    # Step 3: Compute `c` using a weighted average between quadrangle[0] and quadrangle[3]
    c = ((l - length) * quadrangle[0] + length * quadrangle[3]) / l

    # Step 4: Verify that `c` lies within the x-bounds of quadrangle[0] and quadrangle[3] within a margin of `e`
    if not (quadrangle[0][0] - e < c[0] < quadrangle[3][0] + e):
        if debug: print(f"Assertion failed: quadrangle[0][0] - {e} < {c[0]} < quadrangle[3][0] + {e}")
        return None

    # Step 5: Find the intersection point `b` of the line through `c` in the orientation direction with the line quadrangle[0]->quadrangle[1]
    b = get_intersect(c, c + orient, quadrangle[0], quadrangle[1])
    if b is None:
        if debug: print("No intersection found for b.")
        return None

    # Step 6: Find the intersection point `d` of the line through `c` rotated by 90 degrees with the line quadrangle[3]->quadrangle[2]
    d = get_intersect(c, c + np.array([-orient[1], orient[0]]), quadrangle[3], quadrangle[2])
    if d is None:
        if debug: print("No intersection found for d.")
        return None

    # Step 7: Compute the center as the midpoint of `b` and `d`
    center = (np.array(b) + np.array(d)) / 2  # Convert tuples to NumPy arrays

    # Debugging information
    if debug:
        print(f"Intersection point k: {k}")
        print(f"Length l: {l}")
        print(f"Point c: {c}")
        print(f"Intersection point b: {b}")
        print(f"Intersection point d: {d}")
        print(f"Center point: {center}")

    return center

def normalize_angle(ang):
    """
    Normalize an angle to the range (0, 2*pi).
    
    Parameters:
    ang (float): The angle to normalize, in radians.
    
    Returns:
    float: The normalized angle in the range (0, 2*pi).
    """
    normalized_ang = ang % (2 * np.pi)
    
    # Ensure that the angle is positive and non-zero
    if normalized_ang <= 0:
        normalized_ang += 2 * np.pi
    
    return normalized_ang

def calculate_solution(orient1, alpha1, beta1, dist1, orient2, alpha2, beta2, dist2, debug=False):
    # Set a small value for approximation
    small_value = 1e-6

    # Approximate small sine values for alpha1 and alpha2
    sin_alpha1 = np.sin(alpha1) if np.abs(np.sin(alpha1)) >= small_value else small_value
    sin_alpha2 = np.sin(alpha2) if np.abs(np.sin(alpha2)) >= small_value else small_value
    sin_beta1 = np.sin(beta1) if np.abs(np.sin(beta1)) >= small_value else small_value
    sin_beta2 = np.sin(beta2) if np.abs(np.sin(beta2)) >= small_value else small_value

    # assert sin_alpha1 > np.pi/2
    # assert sin_alpha2 > np.pi/2
    # assert sin_beta1 > np.pi/2
    # assert sin_beta2 > np.pi/2

    b = np.array([dist1, dist2])
    A = np.array([
        [np.sin(alpha1 + orient1) / sin_alpha1, np.cos(beta1 - orient1) / sin_beta1],
        [np.sin(alpha2 + orient2) / sin_alpha2, np.cos(beta2 - orient2) / sin_beta2]
    ])

    # Use least squares to solve Ax = b
    x, y = np.linalg.lstsq(A, b, rcond=None)[0]  # Using least squares solution

    # Ensure positive solutions for x and y
    if x > 0 and y > 0:
        if debug:
            print(f"x : {x}, y : {y}")
        return x, y

    return None, None

# def calculate_solution(orient1, alpha1, beta1, dist1, orient2, alpha2, beta2, dist2, debug=False):
    # # Set a small value for approximation
    # small_value = 1e-6

    # # Approximate small sine values for alpha1 and alpha2 to avoid division by 0
    # sin_alpha1 = np.sin(alpha1) if np.abs(np.sin(alpha1)) >= small_value else small_value
    # sin_alpha2 = np.sin(alpha2) if np.abs(np.sin(alpha2)) >= small_value else small_value
    # sin_beta1 = np.sin(beta1) if np.abs(np.sin(beta1)) >= small_value else small_value
    # sin_beta2 = np.sin(beta2) if np.abs(np.sin(beta2)) >= small_value else small_value

    # b = np.array([dist1, dist2])
    
    # # Define the matrix A for the system of equations
    # A = np.array([
        # [np.sin(alpha1 + orient1) / sin_alpha1, np.cos(beta1 - orient1) / sin_beta1],
        # [np.sin(alpha2 + orient2) / sin_alpha2, np.cos(beta2 - orient2) / sin_beta2]
    # ])
    
    # # Use least squares to solve the system Ax = b
    # x, y = np.linalg.lstsq(A, b, rcond=None)[0]  # Using least squares solution

    # if debug:
        # print(f"Initial solution: x = {x}, y = {y}")
    
    # # Add regularization for positivity
    # if x <= 0 or y <= 0:
        # # Penalize negative values and use least squares again with a small shift
        # regularization_term = small_value * np.array([1, 1])
        # A_reg = A + regularization_term
        # x, y = np.linalg.lstsq(A_reg, b, rcond=None)[0]
        
        # if debug:
            # print(f"After regularization: x = {x}, y = {y}")

    # # Ensure positive solutions for x and y
    # if x > 0 and y > 0:
        # return x, y

    # # Adjust angles or distances slightly if the solution is not positive
    # for adjustment in np.linspace(-small_value, small_value, 10):
        # A_adjusted = np.array([
            # [np.sin(alpha1 + orient1 + adjustment) / sin_alpha1, np.cos(beta1 - orient1 + adjustment) / sin_beta1],
            # [np.sin(alpha2 + orient2 + adjustment) / sin_alpha2, np.cos(beta2 - orient2 + adjustment) / sin_beta2]
        # ])
        
        # x, y = np.linalg.lstsq(A_adjusted, b, rcond=None)[0]
        
        # if debug:
            # print(f"Adjusted solution with adjustment {adjustment}: x = {x}, y = {y}")
        
        # # If the solution is positive after adjustment, return it
        # if x > 0 and y > 0:
            # return x, y

    # # If no positive solution was found, return None
    # return None, None

def iv(a):
    return (a[0], -a[1])


def to_iv(array):
    if array is None:
        return None
    converted = []
    for a in array:
        converted.append(iv(a))

    return np.array(converted)
