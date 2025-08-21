import numpy as np
import random
import math
from stl import mesh # Import the mesh module from numpy-stl
from scipy.spatial import distance # Import scipy's distance functions for efficiency

# You might need to install scipy if you haven't already:
# pip install scipy

def calculate_face_normal(v1, v2, v3):
    """Calculates the normal vector for a triangle face."""
    # Calculate two edge vectors of the triangle
    edge1 = v2 - v1
    edge2 = v3 - v1

    # The normal is the cross product of the two edge vectors
    normal = np.cross(edge1, edge2)

    # Normalize the normal vector
    norm = np.linalg.norm(normal)
    if norm == 0:
        # Handle degenerate faces by returning a zero vector;
        # these normals won't contribute meaningfully to vertex normals.
        return np.array([0.0, 0.0, 0.0])
    return normal / norm

def calculate_vertex_normals(vertices, faces):
    """
    Calculates an approximate normal vector for each vertex by averaging
    the normals of the faces connected to it.
    """
    num_vertices = len(vertices)
    # Initialize vertex normals array
    vertex_normals = np.zeros((num_vertices, 3))
    # Initialize array to count how many faces contribute to each vertex normal
    vertex_face_counts = np.zeros(num_vertices)

    # Iterate through each face
    for face_indices in faces:
        # Get the vertex coordinates for the current face
        # Ensure indices are valid before accessing vertices
        try:
            v_coords = [vertices[i] for i in face_indices]
        except IndexError:
            print(f"Warning: Invalid vertex index found in face {face_indices}. Skipping face.")
            continue # Skip this face if indices are invalid

        # Calculate the face normal
        face_normal = calculate_face_normal(v_coords[0], v_coords[1], v_coords[2])

        # Add the face normal to the normals of the vertices in this face
        for v_index in face_indices:
            # Ensure index is within bounds before adding
            if 0 <= v_index < num_vertices:
                vertex_normals[v_index] += face_normal
                vertex_face_counts[v_index] += 1
            else:
                print(f"Warning: Face index {v_index} out of bounds for vertex array of size {num_vertices}.")


    # Normalize the vertex normals by the number of faces connected to each vertex
    # Avoid division by zero for isolated vertices or vertices not part of any face
    for i in range(num_vertices):
        if vertex_face_counts[i] > 0:
            vertex_normals[i] /= vertex_face_counts[i]
            # Final normalization to unit vector
            norm = np.linalg.norm(vertex_normals[i])
            if norm > 1e-6: # Use a small tolerance to avoid division by near-zero
                 vertex_normals[i] /= norm
            else:
                 # If the sum of normals is zero or near-zero, set normal to zero
                 vertex_normals[i] = np.array([0.0, 0.0, 0.0])


    return vertex_normals

def find_boundary_vertices_and_edges(vertices, faces):
    """
    Identifies vertices that lie on the boundary of the mesh and lists the boundary edges.
    Assumes a manifold mesh where non-boundary edges are shared by exactly two faces.

    Returns:
        tuple: (list of boundary vertex indices, list of boundary edge tuples (v_idx1, v_idx2))
    """
    edge_counts = {} # Dictionary to store how many faces each edge belongs to

    # Iterate through each face and its edges
    for face_indices in faces:
        v_indices = face_indices[:3] # Take first 3 for triangles

        if len(v_indices) != 3:
             print(f"Warning: Skipping non-triangle face {face_indices}")
             continue

        edges = [(v_indices[0], v_indices[1]),
                 (v_indices[1], v_indices[2]),
                 (v_indices[2], v_indices[0])]

        canonical_edges = [tuple(sorted(edge)) for edge in edges]

        for edge in canonical_edges:
            if edge in edge_counts:
                edge_counts[edge] += 1
            else:
                edge_counts[edge] = 1

    boundary_edges_canonical = [edge for edge, count in edge_counts.items() if count == 1]

    boundary_vertices = set()
    # Convert canonical edges back to original order for easier distance calculation later if needed,
    # but for just listing boundary vertices, the canonical edges are fine.
    boundary_edges_list = []
    for edge in boundary_edges_canonical:
         # Ensure edge indices are valid before adding to set and list
        if 0 <= edge[0] < len(vertices):
            boundary_vertices.add(edge[0])
        else:
            print(f"Warning: Invalid edge index {edge[0]} found in boundary edge calculation.")
        if 0 <= edge[1] < len(vertices):
             boundary_vertices.add(edge[1])
        else:
            print(f"Warning: Invalid edge index {edge[1]} found in boundary edge calculation.")
        # Add the edge to the list (using canonical form is fine for distance check later)
        boundary_edges_list.append(edge)


    return list(boundary_vertices), boundary_edges_list

def min_distance_to_edges(point, edge_vertices):
    """
    Calculates the minimum distance from a point to a set of line segments (edges).

    Args:
        point (np.ndarray): The point coordinates (shape: (3,)).
        edge_vertices (np.ndarray): An array of edge endpoint coordinates
                                    (shape: (num_edges, 2, 3)).

    Returns:
        float: The minimum distance from the point to any of the edges.
    """
    min_dist = float('inf')

    for edge in edge_vertices:
        p1 = edge[0]
        p2 = edge[1]

        # Calculate the squared length of the edge
        edge_length_sq = np.sum((p2 - p1)**2)

        if edge_length_sq == 0: # Handle zero-length edges (degenerate)
            dist = np.linalg.norm(point - p1)
        else:
            # Calculate the projection of the point onto the line containing the edge
            t = np.dot(point - p1, p2 - p1) / edge_length_sq

            # Clamp t to the range [0, 1] to find the closest point on the line segment
            t = max(0, min(1, t))

            # Calculate the closest point on the line segment
            closest_point_on_segment = p1 + t * (p2 - p1)

            # Calculate the distance from the point to the closest point on the segment
            dist = np.linalg.norm(point - closest_point_on_segment)

        min_dist = min(min_dist, dist)

    return min_dist


def facet_mesh(vertices, faces, max_displacement=0.7, min_distance_threshold=1.0):
    """
    Moves interior vertices of a mesh along their normal vector by a random amount,
    excluding vertices too close to edges.

    Args:
        vertices (np.ndarray): A numpy array of vertex coordinates
                               (shape: (num_vertices, 3)).
        faces (np.ndarray): A numpy array of face definitions.
                                    Each face is an array of vertex indices
                                    (shape: (num_faces, 3)). Assumes triangle faces.
        max_displacement (float): The maximum absolute displacement for vertices.
                                  The units of this displacement are assumed to be
                                  the SAME as the units of the input mesh vertices.
                                  Vertices will move between -max_displacement and +max_displacement.
        min_distance_threshold (float): The minimum distance (in the same units
                                        as the input mesh) an interior vertex must
                                        be from any edge to be displaced.

    Returns:
        np.ndarray: A numpy array of the modified vertex coordinates.
    """
    # Ensure inputs are numpy arrays
    vertices = np.array(vertices, dtype=float)
    faces = np.array(faces, dtype=int)

    print("Calculating vertex normals...")
    vertex_normals = calculate_vertex_normals(vertices, faces)
    print("Vertex normal calculation complete.")

    print("Finding boundary vertices and edges...")
    boundary_vertex_indices, boundary_edge_indices = find_boundary_vertices_and_edges(vertices, faces)
    print(f"Found {len(boundary_vertex_indices)} boundary vertices and {len(boundary_edge_indices)} boundary edges.")

    # Create a set for quick lookup of boundary vertices
    boundary_vertex_set = set(boundary_vertex_indices)

    # Get the coordinates of the boundary edge vertices for distance calculations
    # Reshape boundary_edge_indices from (num_edges, 2) to (num_edges, 2, 3)
    # where the last dimension contains the x, y, z coordinates of the edge endpoints.
    boundary_edge_vertices = vertices[np.array(boundary_edge_indices)]


    # Create a copy of the vertices array to store the modified positions
    modified_vertices = np.copy(vertices)

    print(f"Displacing interior vertices (excluding those within {min_distance_threshold} of an edge)...")
    num_interior_vertices = 0
    num_displaced_vertices = 0

    # Iterate through all vertices by index
    for i in range(len(vertices)):
        # Check if the vertex is NOT a boundary vertex
        if i not in boundary_vertex_set:
            num_interior_vertices += 1

            # Calculate the minimum distance from this interior vertex to any boundary edge
            dist_to_nearest_edge = min_distance_to_edges(vertices[i], boundary_edge_vertices)

            # Check if the vertex is far enough from edges to be displaced
            if dist_to_nearest_edge > min_distance_threshold:
                num_displaced_vertices += 1
                # Generate a random displacement value between -max_displacement and +max_displacement
                displacement = random.uniform(-max_displacement, max_displacement)

                # Get the normal for this vertex
                normal = vertex_normals[i]

                # Move the vertex along its normal by the displacement amount
                modified_vertices[i] = vertices[i] + displacement * normal
            # else: This vertex is an interior vertex but too close to an edge, so it's not displaced.


    print(f"Found {num_interior_vertices} interior vertices.")
    print(f"Displaced {num_displaced_vertices} interior vertices (those not near edges).")

    return modified_vertices

# --- Main Script Execution ---

input_filename = "input.stl"
output_filename = "faceted_output.stl"

# --- ADJUST THESE VALUES ---
# This value represents the maximum displacement in the UNITS OF YOUR INPUT STL FILE.
# If your input STL is in millimeters, set this to 0.7 for 0.7mm displacement.
# If your input STL is in meters, set this to 0.0007 for 0.7mm displacement.
# If your input STL is in inches, set this to 0.7 / 25.4 for 0.7mm displacement.
max_displacement = 1.2 # Assuming input STL is in millimeters

# This value represents the minimum distance (in the UNITS OF YOUR INPUT STL FILE)
# an interior vertex must be from any edge to be displaced.
# Set to 0.0 to displace all interior vertices regardless of edge distance.
min_distance_threshold = 1.0 # Minimum distance from edge to displace vertex (in mm)
# --------------------------

# --- OPTIONAL: Adjust this value to scale the output mesh ---
# If your importing software consistently misinterprets the units,
# you can apply a global scale factor here.
# For example, if your input was effectively meters and you want the output
# to be interpreted as millimeters, set this to 1000.
# Set to 1.0 for no scaling.
output_scale_factor = 0.1 # Use the scaling factor you found
# ----------------------------------------------------------


print(f"Attempting to load mesh from {input_filename}...")
try:
    # Load the mesh from the STL file using numpy-stl
    your_mesh = mesh.Mesh.from_file(input_filename)
    print(f"Successfully loaded mesh from {input_filename}.")

    # numpy-stl stores vertices and faces slightly differently than a simple vertex/face list.
    # It stores each face as a row of 3 vertices in the `vectors` array.
    # To work with our faceting function, which expects unique vertices and face indices,
    # we need to extract those.

    # Get unique vertices and their inverse indices (mapping from original vertex in faces to unique vertex index)
    # np.unique returns unique rows, their indices in the original array, and inverse indices.
    # We need the unique vertices and the inverse indices to reconstruct faces with indices.
    # Reshape the vectors array from (num_faces, 3, 3) to (num_vertices_in_vectors_array, 3)
    vertices_in_vectors = your_mesh.vectors.reshape(-1, 3)
    unique_vertices, inverse_indices = np.unique(vertices_in_vectors, return_inverse=True, axis=0)

    # Reshape the inverse indices to match the original face structure
    # Each face has 3 vertices, and we have inverse indices for each vertex in each face
    face_indices = inverse_indices.reshape(-1, 3)

    print(f"Original mesh has {len(your_mesh.vectors)} faces and {len(unique_vertices)} unique vertices.")
    print(f"Applying maximum displacement of {max_displacement} (in units of input STL).")
    print(f"Excluding interior vertices within {min_distance_threshold} of an edge.")


    # Apply the faceting process to the unique vertices
    print("Starting mesh faceting process...")
    modified_unique_vertices = facet_mesh(
        unique_vertices,
        face_indices,
        max_displacement=max_displacement,
        min_distance_threshold=min_distance_threshold
    )
    print("Mesh faceting process complete.")

    # --- Apply optional scaling ---
    if output_scale_factor != 1.0:
        print(f"Applying output scale factor of {output_scale_factor}...")
        modified_unique_vertices *= output_scale_factor
        print("Scaling complete.")
    # ----------------------------

    # Create a new mesh object with the modified vertices
    # The faces remain the same (same connectivity), but they will now reference the modified vertex positions.
    # We need to recreate the 'vectors' array for the new mesh object using the modified unique vertices
    # and the original face indices.
    modified_vectors = modified_unique_vertices[face_indices]

    # Create a new mesh instance with the same number of faces
    faceted_mesh = mesh.Mesh(np.zeros(modified_vectors.shape[0], dtype=mesh.Mesh.dtype))
    # Assign the modified vertex coordinates to the new mesh's vectors
    faceted_mesh.vectors = modified_vectors

    # Save the modified mesh to a new STL file
    print(f"Saving modified mesh to {output_filename}...")
    faceted_mesh.save(output_filename)
    print(f"Modified mesh saved successfully to {output_filename}.")

    print("\n--- Important Notes ---")
    print("1. The scale of the output mesh depends on how your 3D viewer interprets the units.")
    print(f"   This script applied a maximum displacement of {max_displacement} (in units of input STL).")
    print(f"   Interior vertices within {min_distance_threshold} (in units of input STL) of an edge were not displaced.")
    print(f"   An output scale factor of {output_scale_factor} was applied to the final vertex coordinates.")
    print("2. Ensure your input STL is saved in millimeters from your CAD software, OR adjust the 'max_displacement', 'min_distance_threshold', and 'output_scale_factor' values accordingly.")
    print("3. Check your 3D viewer/importing software settings. Ensure the import units are set correctly (e.g., to millimeters).")
    print("4. To clearly see the faceting, ensure 'flat shading' or 'wireframe' is enabled in your viewer.")


except FileNotFoundError:
    print(f"Error: Input file '{input_filename}' not found in the same directory.")
    print("Please make sure 'input.stl' is in the same folder as the script.")
except ImportError:
     print("Error: Could not import required libraries.")
     print("Please ensure you have installed 'numpy-stl' and 'scipy'.")
     print("Try running: pip install numpy-stl scipy")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc() # Print detailed traceback for other errors

