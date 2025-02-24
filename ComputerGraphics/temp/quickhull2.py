import numpy as np

from utils import generate_random_points, plot_points

class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def orientation(p, q, r):
    # 计算向量pq和pr的叉积
    return (q.y - p.y) * (r.z - p.z) - (q.z - p.z) * (r.y - p.y)

def quickhull(points):
    if len(points) < 4:
        return points

    points.sort(key=lambda p: (p.x, p.y, p.z))
    min_x = points[0]
    max_x = points[-1]
    min_y = min(points, key=lambda p: p.y)
    max_y = max(points, key=lambda p: p.y)
    min_z = min(points, key=lambda p: p.z)
    max_z = max(points, key=lambda p: p.z)
    
    # 找到最小和最大点
    # min_point = max_point = points[0]
    # for p in points:
    #     if p.x < min_point.x:
    #         min_point = p
    #     if p.x > max_point.x:
    #         max_point = p
    
    min_point = min_x
    max_point = max_x
    
    # 分割点集为两部分
    left_set = []
    right_set = []
    
    for p in points:
        if p != min_point and p != max_point:
            if orientation(min_point, max_point, p) > 0:
                left_set.append(p)
            else:
                right_set.append(p)

    # 递归求解两个部分的凸包
    hull = [min_point, max_point]
    hull += find_hull(left_set, min_point, max_point)
    hull += find_hull(right_set, max_point, min_point)

    return hull

def find_hull(points, p, q):
    if not points:
        return []
    
    # 找到与pq线段最远的点
    farthest_point = points[0]
    max_distance = -1
    
    for p_r in points:
        distance = np.linalg.norm(np.cross(np.array([q.x - p.x, q.y - p.y, q.z - p.z]),
                                            np.array([p_r.x - p.x, p_r.y - p.y, p_r.z - p.z]))) / np.linalg.norm([q.x - p.x, q.y - p.y, q.z - p.z])
        if distance > max_distance:
            max_distance = distance
            farthest_point = p_r

    # 移除最远点
    points.remove(farthest_point)

    # 将点集分为两个部分
    left_set = []
    right_set = []
    
    for p_r in points:
        if orientation(p, farthest_point, p_r) > 0:
            left_set.append(p_r)
        else:
            right_set.append(p_r)

    # 递归计算两个部分的凸包
    return find_hull(left_set, p, farthest_point) + [farthest_point] + find_hull(right_set, farthest_point, q)

def quickhull_np(points_array: np.array) -> np.array:
    points = [Point3D(p[0], p[1], p[2]) for p in points_array.tolist()]
    hull = quickhull(points)
    hull_array = np.array([[p.x, p.y, p.z] for p in hull])
    return hull_array

def cross_product(A, B, C):
    AB = Point3D(B.x - A.x, B.y - A.y, B.z - A.z)
    AC = Point3D(C.x - A.x, C.y - A.y, C.z - A.z)
    return Point3D(AB.y * AC.z - AB.z * AC.y,
                 AB.z * AC.x - AB.x * AC.z,
                 AB.x * AC.y - AB.y * AC.x)

def dot_product(v1, v2):
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


def create_face(p1, p2, p3):
    # 创建面，包含三点
    return [p1, p2, p3]
def point_in_triangle(pt, p1, p2, p3):
    # 检查点是否在三角形内
    normal = cross_product(p1, p2, p3)
    p1_pt = Point3D(pt.x - p1.x, pt.y - p1.y, pt.z - p1.z)
    return dot_product(normal, p1_pt) >= 0

def incremental_convex_hull(points):
    # 初始面，由前三个点构成
    if len(points) < 4:
        return points
    # points.sort(key=lambda p: (p.x, p.y, p.z))
    
    hull_faces = [create_face(points[0], points[1], points[2])]

    for p in points[3:]:
        new_faces = []
        for face in hull_faces:
            if point_in_triangle(p, *face):
                # 如果点在面内，则该面需要被替换
                continue
            new_faces.append(face)

        # 创建新的面，连接新点和面上的边
        for face in hull_faces:
            if face not in new_faces:
                new_faces.append(create_face(face[0], face[1], p))
                new_faces.append(create_face(face[1], face[2], p))
                new_faces.append(create_face(face[2], face[0], p))

        hull_faces = new_faces

    # 提取凸包的点
    unique_points = set()
    for face in hull_faces:
        unique_points.update(face)

    return list(unique_points)    

def incremental_convex_hull_np(points_array: np.array) -> np.array:
    points = [Point3D(p[0], p[1], p[2]) for p in points_array.tolist()]
    hull = incremental_convex_hull(points)
    hull_array = np.array([[p.x, p.y, p.z] for p in hull])
    return hull_array

def main1():
    points = [
        Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(2, 0, 0),
        Point3D(0, 2, 0), Point3D(0, 0, 2), Point3D(1, 0, 2),
        Point3D(0, 1, 2), Point3D(2, 1, 1), Point3D(1, 2, 1)
    ]

    convex_hull = quickhull(points)

    print("Convex Hull Points:")
    for p in convex_hull:
        print(f"({p.x}, {p.y}, {p.z})")

 
def main2():
    points_array = generate_random_points(10)
    hull_array = incremental_convex_hull_np(points_array)
    plot_points(points_array, hull_array, True)
    
# 测试代码
if __name__ == "__main__":
    main2()
