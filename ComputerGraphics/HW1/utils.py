import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors

# ********************* Preparation *********************

def generate_random_points(
        num_points=10, 
        x_range=[0, 10], y_range=[0, 10], z_range=[0, 10]
    ) -> np.array:
    """ Generate random points within specified ranges """
    x = np.random.randint(x_range[0], x_range[1], num_points).reshape(-1, 1)
    y = np.random.randint(y_range[0], y_range[1], num_points).reshape(-1, 1)
    z = np.random.randint(z_range[0], z_range[1], num_points).reshape(-1, 1)
    points = np.concatenate([x, y, z], axis=1)
    return points

def plot_points(points, hull_points=None, show_gt=False):
    """ Visualization of 3D points and convex hull """
    if points is None or len(points) == 0:
        raise ValueError("No points to plot")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               color="red", label="Points")
    
    if hull_points is not None and len(hull_points) > 0:
        ax.scatter(
            hull_points[:, 0],
            hull_points[:, 1],
            hull_points[:, 2],
            color="blue",
            label="Computed Convex Hull Points",
            s=60,
            marker="o",
        )
    
    if show_gt:   
        def compute_convex_hull_with_scipy(points):
            """ using scipy library to compute the ground truth convex hull """
            convex_hull = ConvexHull(points)
            hull_points = [points[i] for i in convex_hull.vertices]
            return np.array(hull_points)
        gt_points = compute_convex_hull_with_scipy(points)
        ax.scatter(
            gt_points[:, 0],
            gt_points[:, 1],
            gt_points[:, 2],
            color="green",
            label="Ground Truth Convex Hull Points",
            marker='x',
            s=80)
    
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Points with Convex Hull")
    plt.legend(bbox_to_anchor=(1, 1.2), loc='upper right')
    plt.show()


# displaying the convex hull with matplotlib
def display_hull(hull, points):
    max_coord = points[0][0]
    min_coord = points[0][0]
    hull_points = []

    for p in points:
        point_max = max(p)
        point_min = min(p)
        if point_max > max_coord:
            max_coord = point_max

        if point_min < min_coord:
            min_coord = point_min

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([min_coord, max_coord])
    ax.set_ylim([min_coord, max_coord])
    ax.set_zlim([min_coord, max_coord])
    
    for f in hull.faces:
        triangle = [f.get_points()[0], f.get_points()[1], f.get_points()[2]]
        hull_points.extend(f.get_points())
        face = Poly3DCollection([triangle])
        face.set_color(colors.rgb2hex(np.random.rand(3)))
        face.set_edgecolor('k')
        face.set_alpha(0.7)
        ax.add_collection3d(face)

    # uncomment lines below to draw vertices
    vertices = np.array(points)
    hull_vertices = np.array(hull_points)
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='red')
    ax.scatter(hull_vertices[:, 0], hull_vertices[:, 1], hull_vertices[:, 2], color='black')
    
    plt.axis('off')
    plt.show()
    

# ********************* Helper functions *********************

def cross(a, b):
    return a.x * b.y - b.x * a.y


def collinear(v1, v2, v3):
    return ((v3.z - v1.z) * (v2.y - v1.y) - (v2.z - v1.z) * (v3.y - v1.y) == 0
            and (v2.z - v1.z) * (v3.x - v1.x) - (v2.x - v1.x) * (v3.z - v1.z) == 0
            and (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x) == 0)


def orient(v1, v2, v3, v4):
    det = (-(v1.z - v4.z) * (v2.y - v4.y) * (v3.x - v4.x) + (v1.y - v4.y) * (v2.z - v4.z) * (v3.x - v4.x)
           + (v1.z - v4.z) * (v2.x - v4.x) * (v3.y - v4.y) - (v1.x - v4.x) * (v2.z - v4.z) * (v3.y - v4.y)
           - (v1.y - v4.y) * (v2.x - v4.x) * (v3.z - v4.z) + (v1.x - v4.x) * (v2.y - v4.y) * (v3.z - v4.z))

    if det < 0:
        return -1
    elif det > 0:
        return 1
    else:
        return 0


# same as orient except it takes a face as a parameter instead of 3 points
def visible(f, v):
    face_vertices = f.get_vertices()

    # 1 = not visible, -1 = visible, 0 = coplanar
    return orient(face_vertices[2], face_vertices[1], face_vertices[0], v)



if __name__ == "__main__":
    points = generate_random_points(10)
    print(f'points: \n{points}')
    plot_points(points)
