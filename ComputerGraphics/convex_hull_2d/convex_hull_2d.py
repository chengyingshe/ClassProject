import random
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps
import time
import math
from enum import Enum

def timer(f):
    """ Timer decorator """
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f"T[{f.__name__}] = {(end - start)*1e3:.4f} ms")
        return result
    return wrapper

def generate_random_points(xx=[-50, 50], yy=[-50, 50], num_points=50):
    """
    生成指定数量的随机坐标点集合。
    
    参数:
    xx (list|tuple): 横坐标取值范围
    yy (list|tuple): 纵坐标取值范围
    num_points (int): 需要生成的坐标点的数量。
    
    返回:
    list: 包含随机坐标点的列表，每个点是一个元组 (x, y)
    """
    x_min, x_max = xx
    y_min, y_max = yy
    points = []
    for _ in range(num_points):
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        points.append((x, y))
    return points

def plot_points(points, poly_points=None, plot_polygon=True):
    """
    将给定的点集绘制在图上，并可选地绘制一个多边形。
    
    参数:
    points (list): 包含坐标点的列表，每个点是一个元组 (x, y)。
    poly_points (list, optional): 包含坐标点的列表，用于绘制封闭多边形，默认为 None。
    """
    # 绘制随机点
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    plt.scatter(x_values, y_values, color='blue', label='random points')

    # 绘制多边形
    if poly_points:  # poly_points不为None和[]
        poly_x_values = []
        poly_y_values = []

        for point in poly_points:
            poly_x_values.append(point[0])
            poly_y_values.append(point[1])

        # 将最后一个点与第一个点相连，形成封闭多边形
        poly_x_values.append(poly_x_values[0])
        poly_y_values.append(poly_y_values[0])

        plt.scatter(poly_x_values, poly_y_values, color='red', marker='x', label='extreme points')
        if plot_polygon:
            def sort_hull(poly_x_values, poly_y_values):
                """按极角对凸包点集进行排序"""
                # 计算质心
                centroid = (sum(poly_x_values) / len(poly_x_values), sum(poly_y_values) / len(poly_x_values))
                # 根据与质心的极角进行排序
                idx = list(range(len(poly_x_values)))
                idx = sorted(idx, key=lambda i: math.atan2(poly_y_values[i] - centroid[1], poly_x_values[i] - centroid[0]))
                return [poly_x_values[i] for i in idx], [poly_y_values[i] for i in idx]
            poly_x_values, poly_y_values = sort_hull(poly_x_values, poly_y_values)
            poly_x_values.append(poly_x_values[0])
            poly_y_values.append(poly_y_values[0])
            plt.plot(poly_x_values, poly_y_values, color='red', label='extreme edges')

    plt.title('Convex Hull')
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')
    plt.legend(bbox_to_anchor=(1.05, 1.2), loc='upper right')
    plt.grid(True)
    plt.show()

def check_is_extreme_edge(points, line) -> bool:
    """判断line是否为极值边"""
    p1, p2 = line
    p1_p2 = np.array(p2) - np.array(p1)
    cross_arr = []
    for p in points:
        if p != p1 and p != p2:
            p1_p = np.array(p) - np.array(p1)
            cross_arr.append(int(np.cross(p1_p2, p1_p)))
    return (np.array(cross_arr) <= 0).all() or (np.array(cross_arr) >= 0).all()
    
@timer
def extreme_edged_algo(points) -> list[tuple]:
    """极值边算法，O(n^3)"""
    res_points = set()
    for p1 in points:
        for p2 in points:
            if p1 == p2: continue
            if check_is_extreme_edge(points, [p1, p2]):
                res_points.add(p1)
                res_points.add(p2)
    return list(res_points)

def cross_product(v1: list[tuple], v2: list[tuple]):
    """ 计算向量v1和向量v2的叉积 """
    return np.cross(np.array(v1[1]) - np.array(v1[0]), np.array(v2[1]) - np.array(v2[0]))

def cross_product(v1: list[tuple], v2: list[tuple]):
    """ 计算向量v1和向量v2的叉积 """
    return np.cross(np.array(v1[1]) - np.array(v1[0]), np.array(v2[1]) - np.array(v2[0]))

class Orientation(Enum):
    CLOCKWISE = 1
    ANTI_CLOCKWISE = 2
    COLLINEATION = 0
    
def orientation(p, q, r):
    """ 计算 p, q, r 三个点的方向 """
    val = cross_product([p, q], [q, r])
    if val == 0:
        return Orientation.COLLINEATION
    return Orientation.ANTI_CLOCKWISE if val > 0 else Orientation.CLOCKWISE

@timer
def gift_wrapping_algo(points) -> list[tuple]:
    """ 使用 Gift Wrapping Algorithm 计算凸包 """
    if len(points) < 3:
        return points  # 凸包至少需要 3 个点

    # 找到最右下角的点
    rightmost = max(points, key=lambda p: (p[0], p[1]))
    convex_hull = []

    p = rightmost
    while True:
        convex_hull.append(p)
        # 在当前点 p 的下一个点 q 之前，选择所有其他点 r
        q = points[0]
        for r in points[1:]:
            # 选择 q 为与 p 的连接线形成顺时针方向的点
            if (q == p) or (orientation(p, q, r) == 1):  # 顺时针方向
                q = r
        p = q
        # 如果下一个点是最右下角的点，结束循环
        if p == rightmost:
            break

    return convex_hull

def find_side(p, q, r):
    """确定点 r 相对于线段 pq 的位置:
       返回正数 -> r 在pq的左边
       返回负数 -> r 在pq的右边
       返回0 -> r 在pq上
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0

def distance_p2l(point, line: list[tuple]):
    """计算点 point 到直线 line 的垂直距离"""
    p, q = line
    return abs((q[1] - p[1]) * (point[0] - p[0]) - (q[0] - p[0]) * (point[1] - p[1]))

def find_side(p, q, r):
    """确定点 r 相对于线段 pq 的位置:
       返回正数 -> r 在pq的左边
       返回负数 -> r 在pq的右边
       返回0 -> r 在pq上
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0

@timer
def quick_hull_algo(points) -> list[tuple]:
    """主函数，计算点集的凸包"""
    if len(points) < 3:
        return points
    
    def quick_hull(points, p, q, side):
        """递归寻找凸包上的点"""
        idx = -1
        max_dist = 0

        # 寻找点集中离线段 pq 距离最远的点
        for i in range(len(points)):
            temp = distance_p2l(points[i], [p, q])
            if find_side(p, q, points[i]) == side and temp > max_dist:
                idx = i
                max_dist = temp

        # 如果没有外侧点，则 p-q 是凸包的一条边
        if idx == -1:
            return [p, q]

        # 递归求解：找离 p-r 和 r-q 两条边最远的点
        chull = []
        chull.extend(quick_hull(points, points[idx], p, -find_side(points[idx], p, q)))
        chull.extend(quick_hull(points, points[idx], q, -find_side(points[idx], q, p)))
        return chull
    
    # 找到最左边和最右边的点
    min_x = min(points, key=lambda p: p[0])
    max_x = max(points, key=lambda p: p[0])

    # 上方点集和下方点集分别计算
    upper_hull = quick_hull(points, min_x, max_x, 1)
    lower_hull = quick_hull(points, min_x, max_x, -1)

    # 合并上下凸包，去重
    convex_hull = upper_hull + lower_hull
    unique_hull = []
    for pt in convex_hull:
        if pt not in unique_hull:
            unique_hull.append(pt)

    return unique_hull

def polar_angle(p0, p1):
    """返回点 p1 相对于点 p0 的极角"""
    return math.atan2(p1[1] - p0[1], p1[0] - p0[0])

@timer
def graham_scan_algo(points) -> list[tuple]:
    """ 使用 Graham Algorithm 求凸包 """
    # 第一步：找到最下方的点，如果有多个则选择最左边的点
    start = min(points, key=lambda p: (p[1], p[0]))

    # 第二步：按与基准点的极角进行排序
    sorted_points = sorted(points, key=lambda p: (polar_angle(start, p), -p[1], p[0]))

    # 第三步：构建凸包，使用栈保存当前的边界
    hull = []
    for p in sorted_points:
        while len(hull) > 1 and orientation(hull[-2], hull[-1], p) != 2:
            hull.pop()  # 如果不是逆时针，则移除最后的点
        hull.append(p)

    return hull

@timer
def incremental_algo(points):
    """ Incremental Algorithm 算法求凸包 """
    points = sorted(points, key=lambda x: (x[0], x[1]))  # 将点按x坐标排序，如果x相同则按y坐标排序
    if len(points) < 3:
        return points  # 点数少于3个，直接返回这些点
    lower_hull = []
    for p in points:
        # 保证每次加入的点都保持凸性，删除不满足条件的点
        while len(lower_hull) >= 2 and cross_product([lower_hull[-2], lower_hull[-1]], [lower_hull[-2], p]) <= 0:
            lower_hull.pop()
        lower_hull.append(p)

    upper_hull = []
    for p in reversed(points):
        while len(upper_hull) >= 2 and cross_product([upper_hull[-2], upper_hull[-1]], [upper_hull[-2], p]) <= 0:
            upper_hull.pop()
        upper_hull.append(p)

    # 合并上半部分和下半部分，最后一个点重复，所以去掉一个
    return lower_hull[:-1] + upper_hull[:-1]