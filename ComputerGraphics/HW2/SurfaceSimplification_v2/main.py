import argparse
import os
import time
import polyscope as ps
import numpy as np
import heapq
from qem_utils import load_obj, save_obj

MINIMUM_FACES = 5
NUM_VERTICES = 0
labels = []
coordinates = []
faces = []
neighbours = []
Qs = []
ps.init()

def get_all_edges_of_vertex(v):
    """Get all edges from faces that contain the vertex v"""
    edges = set()
    edges.add(v)
    for face in faces:
        if v in face:
            edges.update(face)
    edges.remove(v)
    return list(edges)

def init_lists():
    """Initialize all lists: labels, coordinates, faces and neighbours"""
    global labels, neighbours, NUM_VERTICES
    NUM_VERTICES = len(coordinates)
    print(f"Vertex number: {NUM_VERTICES}")
    labels = list(range(NUM_VERTICES))
    neighbours = [get_all_edges_of_vertex(i) for i in labels]

def vector(p1, p2):
    """Calculate vector from point p1 to p2"""
    return np.array(p2) - np.array(p1)


def cross_product(v1, v2):
    """Calculate cross product of two vectors"""
    return np.cross(v1, v2)


def vector_norm(v):
    """Calculate the norm of a vector"""
    return float(np.linalg.norm(v))


def dot_product(v1, v2):
    """Calculate dot product of two vectors v1 and v2"""
    return np.dot(v1, v2)

def get_valid_pairs():
    """Retrieve all valid pairs of vertices for contraction"""
    valid_pairs = []
    for v in range(NUM_VERTICES):
        for n in neighbours[v]:
            if v < n:
                valid_pairs.append((v, n))
    return valid_pairs

def get_coord(i):
    """Get coordinate of a vertex from index"""
    return coordinates[i]

def set_coord(i, coord):
    """Set coordinate of a vertex from index"""
    global coordinates
    coordinates[i] = coord


def get_all_faces_contain_vertex(v):
    """Retrieve all faces containing a given vertex"""
    containing_faces = []
    for face in faces:
        if v in face:
            containing_faces.append(face)
    return containing_faces


def calculate_Kp(face):
    """Calculate the Kp matrix of the face"""
    p1, p2, p3 = face
    p1_p3 = vector(p1, p3)
    p1_p2 = vector(p1, p2)
    normal = cross_product(p1_p2, p1_p3)
    norm = vector_norm(normal)
    n = (normal / norm).tolist()
    d = float(dot_product(normal, p1) / norm)
    n.append(d)
    return n  # [nx, ny, nz, d]


def calculate_all_Kp(v):
    """Calculate all Kp matrices"""
    faces_contain_vertex = get_all_faces_contain_vertex(v)
    Kp = []
    
    for face in faces_contain_vertex:
        v1 = get_coord(face[0])
        v2 = get_coord(face[1])
        v3 = get_coord(face[2])
        Kp.append(calculate_Kp([v1, v2, v3]))

    return Kp

def calculate_Q(v):
    """Calculate the Q matrix of a vertex"""
    Kps = calculate_all_Kp(v)
    Q = []
    for matrix in Kps:
        matrix_row = np.matrix(matrix).reshape(1, 4)
        matrix_column = matrix_row.reshape(4, 1)
        Q.append(matrix_column * matrix_row)
    
    Q = np.sum(Q, axis=0)
    return Q

def calculate_all_Q():
    """Calculate all Q matrices"""
    global Qs
    Qs = [calculate_Q(i) for i in range(NUM_VERTICES)]

def calculate_quadratic_error(v, Q):
    """Calculate quadratic error for a vertex v and matrix Q"""
    return (v.T * Q * v)[0, 0]

def midpoint(v1, v2):
    """Calculate midpoint of two vertices"""
    return [(v1[i] + v2[i]) / 2 for i in range(3)]


def calculate_collapse_error(v1, v2):
    """Calculate the error resulting from collapsing two vertices"""
    coord_v1 = get_coord(v1)
    coord_v2 = get_coord(v2)
    Q1 = Qs[v1]
    Q2 = Qs[v2]
    coord_v3 = midpoint(coord_v1, coord_v2)
    Q3 = Q1 + Q2
    V3 = np.matrix([coord_v3[0], coord_v3[1], coord_v3[2], 1]).T
    error = calculate_quadratic_error(V3, Q3)
    return [error, (v1, v2)]


def compute_optimal_collapse_position(v1, v2):
    """Determine the optimal position to minimize quadratic error during contraction"""
    coord_v1 = get_coord(v1)
    coord_v2 = get_coord(v2)
    Q1 = Qs[v1]
    Q2 = Qs[v2]
    coord_v3 = midpoint(coord_v1, coord_v2)
    Q3 = Q1 + Q2
    error_v1 = calculate_quadratic_error(
        np.matrix([coord_v1[0], coord_v1[1], coord_v1[2], 1]).T, Q1
    )
    error_v2 = calculate_quadratic_error(
        np.matrix([coord_v2[0], coord_v2[1], coord_v2[2], 1]).T, Q2
    )
    error_v3 = calculate_quadratic_error(
        np.matrix([coord_v3[0], coord_v3[1], coord_v3[2], 1]).T, Q3
    )

    if error_v1 < error_v2 and error_v1 < error_v3:
        return coord_v1
    elif error_v2 < error_v1 and error_v2 < error_v3:
        return coord_v2
    else:
        return coord_v3

def calculate_collapse_costs(pairs):
    """Compute the cost of all valid point pairs, and store them in a heap"""
    costs = [calculate_collapse_error(pair[0], pair[1]) for pair in pairs]
    costs = sorted(costs, key=lambda x: x[0])
    heapq.heapify(costs)
    return costs


def sorted_heap(heap):
    """Sort the heap"""
    sorted_list = []
    while heap:
        item = heapq.heappop(heap)
        sorted_list.append([item[0], item[1]])
    return sorted_list


def get_label(i):
    while i != labels[i]:
        i = labels[i]
    return i

def union(i, j):
    global labels
    labels[get_label(i)] = get_label(j)

def get_num_of_labels():
    res = []
    for i in range(len(labels)):
        if i == get_label(i):
            res.append(i)
    return len(res)

def get_pairs_with_vertex(heap, v):
    res = []
    for i in range(len(heap)):
        if heap[i][1][0] == v or heap[i][1][1] == v:
            res.append(i)
    return res


def update_pairs_with_vertex(heap, list):
    for i in range(len(list)):
        heap[list[i]][0] = calculate_collapse_error(
            get_label(heap[list[i]][1][0]), 
            get_label(heap[list[i]][1][1]))[0]

def delete_same_pair(heap, v1, v2):
    res = []
    pairv1 = get_pairs_with_vertex(heap, v1)
    pairv2 = get_pairs_with_vertex(heap, v2)
    for pair in pairv1:
        if heap[pair][1][0] == v1:
            for pair2 in pairv2:
                if (heap[pair][1][1] == heap[pair2][1][0]) or (
                    heap[pair][1][1] == heap[pair2][1][1]
                ):
                    if not (pair in res):
                        res.append(pair)
            if not (pair in res):
                heap[pair][1] = (v2, heap[pair][1][1])
        else:
            for pair2 in pairv2:
                if (heap[pair][1][0] == heap[pair2][1][0]) or (
                    heap[pair][1][0] == heap[pair2][1][1]
                ):
                    if not (pair in res):
                        res.append(pair)
                if not (pair in res):
                    heap[pair][1] = (heap[pair][1][0], v2)
    return res

def simplify_pairs(ratio, heap):
    global Qs, labels
    num_to_simplify = int(NUM_VERTICES * ratio)
    num_simplified = 0
    
    while (num_simplified < num_to_simplify and 
           num_simplified < NUM_VERTICES - MINIMUM_FACES):

        # collapse pair from the heap
        cost, pair = heapq.heappop(heap)
        v1 = get_label(pair[0])
        v2 = get_label(pair[1])

        set_coord(v2, compute_optimal_collapse_position(v1, v2))
        union(v1, v2)
        Qs[v2] = Qs[v1] + Qs[v2]

        pair_to_del = delete_same_pair(heap, v1, v2)
        pair_to_del.reverse()
        for i in pair_to_del:
            heap.pop(i)

        update_pairs_with_vertex(heap, get_pairs_with_vertex(heap, v1))
        update_pairs_with_vertex(heap, get_pairs_with_vertex(heap, v2))

        heapq.heapify(heap)
        heap = sorted_heap(heap)
        num_simplified += 1

    final_label = []
    for label in labels:
        l = get_label(label)
        if l not in final_label:
            final_label.append(l)
            
    ps_coord = []
    for i in range(len(coordinates)):
        ps_coord.append(coordinates[get_label(i)])
    ps_coord = np.array(ps_coord)
    ps_faces = faces
    
    labels = final_label
    print(f"Vertex number(after simplification): {len(labels)}")
    return ps_coord, ps_faces


def main(args):
    ratio = args.ratio
    saved = args.saved
    obj_path = args.input
    visualization = args.visualization
    print(f"Ratio of simplification: {ratio}")
    ps_coord, ps_faces = np.array(coordinates), faces
    t = 0
    if 0 < ratio < 1:
        start = time.time()
        
        init_lists()  # initialization
        calculate_all_Q()  # calculate Q matrices for all vertices
        valid_pairs = get_valid_pairs()  # get all point pairs of the edges
        heap = calculate_collapse_costs(valid_pairs)  # calculate all costs while collapsing pairs
        ps_coord, ps_faces = simplify_pairs(ratio, heap)
        t = time.time() - start
        print(f"Time: {t}")
        
    print(f"Visualization: {visualization}")
    if visualization:
        ps.register_surface_mesh("QEM", ps_coord, ps_faces)
        ps.show()
    if saved:
        root, ext = os.path.splitext(obj_path)
        output_obj_path = f"{root}_output_r{ratio}_v{len(labels) if len(labels) > 0 else len(coordinates)}_t{t:.3f}{ext}"
        save_obj(ps_coord, ps_faces, output_obj_path)

def parse_args():
    """Parse the input arguments"""
    parser = argparse.ArgumentParser(
        description="Simplify the 3D model using the method of QEM"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="bunny.obj",
        help="Path to the input .obj file",
    )
    parser.add_argument(
        "-r",
        "--ratio",
        type=float,
        default=0.5,
        help="The ratio of simplification (default: 0.5)",
    )
    parser.add_argument(
        "-s",
        "--saved",
        default=False,
        action="store_true",
        help="Save the simplified 3D mesh to output.obj",
    )
    parser.add_argument(
        "-v",
        "--visualization",
        default=False,
        action="store_true",
        help="Visualize the simplified 3D mesh",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    obj_path = args.input
    coordinates, faces = load_obj(obj_path)
    main(args)