import math
import numpy as np
from utils import generate_random_points, plot_points

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

EPSOLON = 1e-9

def cross_product(A, B, C):
    AB = Point(B.x - A.x, B.y - A.y, B.z - A.z)
    AC = Point(C.x - A.x, C.y - A.y, C.z - A.z)
    return Point(AB.y * AC.z - AB.z * AC.y,
                 AB.z * AC.x - AB.x * AC.z,
                 AB.x * AC.y - AB.y * AC.x)

def dot_product(v1, v2):
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def length(v):
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

def point_to_plane_distance(A, B, C, P):
    normal = cross_product(A, B, C)
    AP = Point(P.x - A.x, P.y - A.y, P.z - A.z)
    return abs(dot_product(normal, AP)) / (length(normal) + EPSOLON)

def signed_distance_to_plane(A, B, C, P):
    normal = cross_product(A, B, C)
    AP = Point(P.x - A.x, P.y - A.y, P.z - A.z)
    return dot_product(normal, AP)

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
        if signed_distance_to_plane(furthest_point, face.b, face.a, p) > 0:
            left_of_ab.append(p)
        if signed_distance_to_plane(furthest_point, face.c, face.b, p) > 0:
            left_of_bc.append(p)
        if signed_distance_to_plane(furthest_point, face.a, face.c, p) > 0:
            left_of_ca.append(p)

    hull.update(quick_hull_recursive(left_of_ab, Face(furthest_point, face.b, face.a)))
    hull.update(quick_hull_recursive(left_of_bc, Face(furthest_point, face.c, face.b)))
    hull.update(quick_hull_recursive(left_of_ca, Face(furthest_point, face.a, face.c)))
    
    return hull
def quick_hull(points):
    points.sort(key=lambda p: (p.x, p.y, p.z))
    min_x = points[0]
    max_x = points[-1]
    min_y = min(points, key=lambda p: p.y)
    max_y = max(points, key=lambda p: p.y)
    min_z = min(points, key=lambda p: p.z)
    max_z = max(points, key=lambda p: p.z)
    
    initial_face = Face(min_x, max_x, max_z)
    hull = quick_hull_recursive(points, initial_face)
    return hull

def quick_hull_np(points_array: np.array) -> np.array:
    points = [Point(p[0], p[1], p[2]) for p in points_array.tolist()]
    hull = quick_hull(points)
    hull_array = np.array([[p.x, p.y, p.z] for p in hull])
    return hull_array

if __name__ == "__main__":
    points_array = generate_random_points(15)
    print(f'points: {points_array}')
    hull_array = quick_hull_np(points_array)
    print(f'convex: {hull_array}')
    plot_points(points_array, hull_array, True)