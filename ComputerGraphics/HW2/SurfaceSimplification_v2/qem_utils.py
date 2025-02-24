import numpy as np
from scipy.spatial.distance import cdist

def load_obj(obj_path):
    """Load .obj file"""
    vertices = []
    faces = []
    with open(obj_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if len(line) == 0 or (len(line) > 0 and line[0] == '#'): 
                # skip the empty lines and comments
                continue
            line_split = line.split(' ')
            line_split = [i for i in line_split if i != '']
            if line_split[0] in ['v', 'f']:
                if line_split[0] == 'v':
                    vertices.append([float(v) for v in line_split[1:]])
                elif line_split[0] == 'f':
                    faces.append([int(f) - 1 for f in line_split[1:]])  # index start from 1
            else:
                print(f"Error: [{i + 1}] {line}")
    return vertices, faces

def save_obj(coords, faces, output_path="output.obj"):
    """Save the mesh to output.obj"""
    with open(output_path, "w") as f:
        for coord in coords:
            f.write(f"v {coord[0]} {coord[1]} {coord[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    print(f"Successfully save to {output_path}!")

def chamfer_distance(coords1, coords2):
    """Calculate the CD (Chamfer Distance) of two vertices list"""
    dist_matrix = cdist(coords1, coords2, metric='euclidean')
    
    cd1 = np.sum(np.min(dist_matrix, axis=1))
    cd2 = np.sum(np.min(dist_matrix, axis=0))
    
    return cd1 / len(coords1) + cd2 / len(coords2)

def hausdorff_distance(coords1, coords2):
    """Calculate the HD (Hausdorff Distance) of two vertices list"""
    dist_matrix = cdist(coords1, coords2, metric='euclidean')
    
    hd1 = np.max(np.min(dist_matrix, axis=1))
    hd2 = np.max(np.min(dist_matrix, axis=0))
    
    return max(hd1, hd2)

if __name__ == "__main__":
    obj_path1 = 'models/octopus.obj'
    obj_path2 = 'models/octopus_output_r0.8_v419_t8.375.obj'
    coords1, faces1 = load_obj(obj_path1)
    coords2, faces2 = load_obj(obj_path2)

    cd = chamfer_distance(coords1, coords2)
    hd = hausdorff_distance(coords1, coords2)

    print(f"Chamfer Distance: {cd}")
    print(f"Hausdorff Distance: {hd}")