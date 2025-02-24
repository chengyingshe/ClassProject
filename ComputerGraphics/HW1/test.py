import math
import numpy as np
from convex_hull_3d import generate_random_points, plot_points

class Point:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __lt__(self, other):
        if self.x != other.x:
            return self.x < other.x
        if self.y != other.y:
            return self.y < other.y
        return self.z < other.z

class Face:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

def cross_product(A, B, C):
    AB = Point(B.x - A.x, B.y - A.y, B.z - A.z)
    AC = Point(C.x - A.x, C.y - A.y, C.z - A.z)
    return Point(AB.y * AC.z - AB.z * AC.y,
                 AB.z * AC.x - AB.x * AC.z,
                 AB.x * AC.y - AB.y * AC.x)

def point_to_plane_distance(A, B, C, P):
    normal = cross_product(A, B, C)
    return abs(normal.x * (P.x - A.x) + normal.y * (P.y - A.y) + normal.z * (P.z - A.z)) / \
           (math.sqrt(normal.x**2 + normal.y**2 + normal.z**2) + 1e-15)

def signed_distance_to_plane(A, B, C, P):
    normal = cross_product(A, B, C)
    return normal.x * (P.x - A.x) + normal.y * (P.y - A.y) + normal.z * (P.z - A.z)

def find_furthest_point(points, face):
    max_distance = -1
    furthest_point = None

    for p in points:
        distance = point_to_plane_distance(face.a, face.b, face.c, p)
        if distance > max_distance:
            max_distance = distance
            furthest_point = p

    return furthest_point

def quick_hull_recursive(points, face):
    hull = set()
    furthest_point = find_furthest_point(points, face)
    if furthest_point is None or point_to_plane_distance(face.a, face.b, face.c, furthest_point) < 1e-9:
        hull.add(face.a)
        hull.add(face.b)
        hull.add(face.c)
        return hull

    left_of_ab = []
    left_of_bc = []
    left_of_ca = []

    for p in points:
        if signed_distance_to_plane(face.a, face.b, furthest_point, p) > 0:
            left_of_ab.append(p)
        if signed_distance_to_plane(face.b, face.c, furthest_point, p) > 0:
            left_of_bc.append(p)
        if signed_distance_to_plane(face.c, face.a, furthest_point, p) > 0:
            left_of_ca.append(p)

    hull.update(quick_hull_recursive(left_of_ab, Face(face.a, face.b, furthest_point)))
    hull.update(quick_hull_recursive(left_of_bc, Face(face.b, face.c, furthest_point)))
    hull.update(quick_hull_recursive(left_of_ca, Face(face.c, face.a, furthest_point)))
    
    return hull
def quick_hull(points):
    points.sort(key=lambda p: (p.x, p.y, p.z))
    min_x = points[0]
    max_x = points[-1]
    min_y = min(points, key=lambda p: p.y)
    max_y = max(points, key=lambda p: p.y)
    min_z = min(points, key=lambda p: p.z)
    max_z = max(points, key=lambda p: p.z)
    
    initial_face = Face(min_x, max_x, max_y)
    hull = quick_hull_recursive(points, initial_face)
    return hull

# points_array = np.array([
#     [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 4], [1, 1, 1], 
#     [2, 0, 0], [0, 2, 0], [2, 3, 0], [0, 2, 3], [0, 2, 4]
# ])
points_array = np.array([
    [0, 7, 5],
    [4, 0, 4],
    [0, 6, 0],
    [1, 7, 1],
    [2, 2, 1],
    [3, 9, 4],
    [4, 5, 4],
    [9, 6, 2],
    [8, 9, 5],
    [1, 3, 0]
])
# points_array = generate_random_points(10)
print(f'points: {points_array}')
points = [Point(p[0], p[1], p[2]) for p in points_array.tolist()]
convex_hull = quick_hull(points)
hull_array = np.array([[p.x, p.y, p.z] for p in convex_hull])
print(f'convex: {hull_array}')
plot_points(points_array, hull_array, True)