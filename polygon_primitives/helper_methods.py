import math
import numpy as np
from sklearn.cluster import KMeans
from line_extraction_primitives.line import Line
from line_extraction_primitives.extraction_helper_methods import get_heights, rotate_line_about_point
from line_extraction_primitives.grid import Grid

def normalize(v):
    norm = np.linalg.norm(v)

    if norm == 0: 
       return v

    return v / norm

def compare_vector(v1, v2):
    normv1 = normalize(v1)
    normv2 = normalize(v2)

    if min(normv1 - normv2, normv1 + normv2) < 0.01:
        return True
    return False

def collinear(p0, p1, p2, dist=7.0):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]

    coll = abs(x1 * y2 - x2 * y1)
    return coll < dist

def get_collinearity(p0, p1, p2):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]

    coll = abs(x1 * y2 - x2 * y1)
    return coll

def distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def angle_from_horizon(point1, point2):
    v1 = normalize(point1 - point2)
    v2 = normalize(np.array([point1[0], 0]))
    return angle_between(v1, v2)

def angle_between(v1, v2):
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def is_point_left(point1, point2, comp_point):
    v1 = np.array(point2) - np.array(point1)
    v2 = np.array(comp_point) - np.array(point1)

    cross = np.cross(v1, v2)
    if np.sign(cross) >= 0:
        return False
    else:
        return True

def signed_angle_between(v1, v2):
    theta = np.rad2deg(np.arctan2(v2[1],v2[0]) - np.arctan2(v1[1],v1[0]))
    if theta > 200.0:
        theta = -360.0 + theta
    elif theta < -200.0:
        theta = 360.0 + theta
    return theta


#Checks if too points are within a distance epsilon from one another
def compare_points(point1, point2, epsilon=0.8):
    if distance(point1, point2) < epsilon:
        return True
    return False

def get_intersecting_edge_point(edge1, edge2, dist=0.8):
    point = None
    start1 = edge1.get_start()
    end1 = edge1.get_end()
    start2 = edge2.get_start()
    end2 = edge2.get_end()

    if compare_points(start1, start2, epsilon=dist) or compare_points(start1, end2, epsilon=dist):
        point = start1
    elif compare_points(end1, start2, epsilon=dist) or compare_points(end1, end2, epsilon=dist):
        point = end1

    return point

#Reverses the edges in a list
def reverse_edges(edges):
    for edge in edges:
        edge.reverse_direction()
    edges.reverse()
    return edges


#Returns an ordered list of corner points (i.e an ordering in which subsequent corners are connected by a set of edges)
def get_corner_ordering(edges, dist=0.3):
    
    if len(edges) != 4:
        return []
    points = [edges[0].get_start(), edges[0].get_end()]
    while len(points) < len(edges):
        added = False
        for edge in edges:
            if compare_points(edge.get_start(), points[-1], dist) and not compare_points(edge.get_end(), points[-2], dist):
                points.append(edge.get_end())
                added = True
            elif compare_points(edge.get_end(), points[-1], dist) and not compare_points(edge.get_start(), points[-2], dist):
                points.append(edge.get_start())
                added = True
        if not added:
            return []
            
    return points

#Gets total length of a group of edges
def get_edges_length(edges):
    lengths = [edge.get_length() for edge in edges]
    return sum(lengths)

#Gets the the unique points from a list of edges. Note that the points are returned in the order that they appear in the edges.
#Calling this after ordering a polygon's edges with polygon.order_edges() will return the points in order around a polygon 
def get_unique_points(edges):
    points = []
    for each in edges:
        point1 = each.get_start()
        point2 = each.get_end()

        b_point1 = True
        b_point2 = True
        for other_point in points:
            if compare_points(point1, other_point):
                b_point1 = False
            if compare_points(point2, other_point):
                b_point2 = False
        if b_point1:
            points.append(point1)
        if b_point2:
            points.append(point2)
    return points

#Searches for the edges that lie between two corners. Input edges do NOT need to be ordered.
def search_between_corners(edges, corner1, corner2):
    between_edges = []
    dist = distance(corner1, corner2)
    for edge in edges:
        if collinear(corner1, corner2, edge.get_start(), dist=dist) and collinear(corner1, corner2, edge.get_end(), dist=dist):
            between_edges.append(edge)
    return between_edges

#Returns a list of edges that lie between point1 and point2. Input edges need to be ordered (but the order direction is not important).
def find_edges_between_points(ordered_edges, point1, point2, dist=0.8):
    between_edges = []
    is_reversed = False
    is_adding_edges = False
    
    for i in range(len(ordered_edges)):
        edge = ordered_edges[i]
        if not is_adding_edges:
            edge_is_start = compare_points(edge.get_start(), point1, dist)
            edge_is_end = compare_points(edge.get_start(), point2, dist)

            if edge_is_start or edge_is_end:
                between_edges.append(edge)
                is_adding_edges = True

            if edge_is_end:
                is_reversed = True

        else:
            between_edges.append(edge)
            edge_is_start = compare_points(edge.get_end(), point1, dist)
            edge_is_end = compare_points(edge.get_end(), point2, dist)

            if is_reversed:
                if edge_is_start:
                    break
            else:
                if edge_is_end:
                    break
    return between_edges


def edge_list_above_min_length(edges, min_length):
    edge_list = []
    for edge in edges:
        edge_list.append(edge)
        length = get_edges_length(edge_list)
        if length > min_length:
            break

    if get_edges_length(edge_list) > min_length:
        return edge_list
    return []

def search_along_line(ordered_edges, start_point, end_point=None):
    edges = []
    add_edges = False
    for edge in ordered_edges:
        if compare_points(edge.get_start(), start_point) or add_edges:
            edges.append(edge)
            add_edges = True
        if end_point is not None and compare_points(edge.get_end(), end_point):
            break
    return edges

def choose_min_distance_points(points):
    if len(points) < 2:
        return []
    min_dist = float("inf")
    min_points = []
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                continue
            dist = distance(points[i], points[j])
            if dist < min_dist:
                min_dist = dist
                min_points = [points[i], points[j]]
    return min_points

def get_total_edge_length(edges):
    length = 0
    for edge in edges:
        length += edge.get_length()
    return length

def move_edge_points(poly_edges, curr_loc, new_loc):
    for edge in poly_edges:
        if compare_points(edge.get_start(), curr_loc):
            edge.set_start(new_loc)
        if compare_points(edge.get_end(), curr_loc):
            edge.set_end(new_loc)

#Gets all points around an edge, and computes the 30th and 70th percentiles for height
def get_spatial_percentiles(edge, grid):
    line = Line()
    line.set_points(edge.get_start(), edge.get_end())
    transformed_line = grid.transform_line_to_grid_space(line)

    edge_length = edge.get_length()

    #Parameter to determine how many cells at the ends of each line to remove from consideration
    b_cell_cutoff = True
    if edge_length / 0.4 < 6:
        b_cell_cutoff = False
        num_cells = int(edge_length / 0.2)
        width = edge_length / float(num_cells)
    elif edge_length / 0.4 < 15.0:
        num_cells = int(edge_length / 0.3)
        width = edge_length / float(num_cells)
    else:
        num_cells = int(edge_length / 0.4)
        width = edge_length / float(num_cells)
        
    if num_cells == 0 or width == float("inf"):
        return [0.0, 0.0]

    height = 1.0
    points = grid.get_points_from_line_segment(transformed_line, width=height)

    if len(points) > 0:
        theta = line.find_rotation_angle()
        rotation_point = np.append(line.get_start(), 0.0)
        rotated_points = rotate_line_about_point(points, theta, rotation_point)

        line_grid = Grid(rotated_points, width * 2, height * 2)

        if b_cell_cutoff:
            heights = get_heights(line_grid.get_cells()[3:-3])
        else:
            heights = get_heights(line_grid.get_cells())

        if len(heights) > 0:
            percentiles = np.percentile(heights, [30, 70])
        else:
            percentiles = [0.0, 0.0]
    else:
        percentiles = [0.0, 0.0]
        
    return percentiles

def get_intersection_point(v1, v2, start1, start2):
    if v2[1] == 0.0 or (v1[0] - v1[1] * v2[0] / v2[1]) == 0.0:
        return None

    denom_a = (v1[0] - v1[1] * v2[0] / v2[1])
    a = ((start1[1] - start2[1]) * v2[0] / v2[1] + (start2[0] - start1[0])) / denom_a

    intersection_point = start1 + a * v1
    return intersection_point

#Gets corners from an ordered list of edges
def search_for_corners(edges, dist=0.8):
    corners = []
    for i in range(len(edges)):
        edge = edges[i]
        next_index = (i + 1) % len(edges)
        other_edge = edges[next_index]
        shared_point = edge.get_shared_point(other_edge, dist)
        if shared_point is not None:
            angle = edge.get_angle_between_edges(other_edge, dist)
            if (angle > 60 and angle < 120) or (angle < -60 and angle > -120):
                in_corners = False
                for corner in corners:
                    if compare_points(corner, shared_point, dist):
                        in_corners = True
                if not in_corners:
                    corners.append(shared_point)
    return corners