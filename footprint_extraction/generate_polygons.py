import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from line_extraction_primitives.line import Line, load_lines, save_lines
from line_extraction_primitives.grid import Grid
from footprint_extraction.extract_lines import construct_grid

import numpy as np
import math
import polygon_primitives.primitive_polygon as pp
import polygon_primitives.primitive_edge as pe
import polygon_primitives.helper_methods as phm
import polygon_primitives.file_writer as fw
import plotting

minimum_wall_height = 2.0
minimum_wall_length = 2.0
filename = "../LAS/alamedaprocessed.las"

"""Given an incoming edge and a polygon, determines the next edge to add to the polygon.""" 
def search_along_edge(edge, incoming_edge, polygon, original_edge, init_dir=0, swapped_dir = False, dist=0.5):
    
    #Walks along the edge (i.e. starting from the incoming edge) and adds all edges connected to the new endpoint to the list of potential edges
    potential_edges = []
    if incoming_edge is None:
        if init_dir == 0:
            potential_edges = edge.get_start_neighbors()
        else:
            potential_edges = edge.get_end_neighbors()
    else:
        potential_edges = edge.walk_along_edge(incoming_edge)
    
    #Chooses the edge with the minimum angle along the traversal direction
    next_edge = polygon.choose_member_edge(potential_edges, edge, dist=dist)

    #If we reach an endpoint with no viable neighbors, we go back to the starting endpoint and traverse the other direction
    if next_edge is None:
        if swapped_dir:
            return
        polygon.dir = 1 - polygon.dir
        swapped_dir = True
        poly_edges = polygon.get_edges()
        next_edge = original_edge

        edge = None
        init_dir = 1 - init_dir
    
    #If we return to the starting edge, we have completed the traversal
    elif next_edge in polygon.get_edges():
        polygon.completed = True
        return
    else:
        polygon.add_edge(next_edge, prepend=swapped_dir)
    
    search_along_edge(next_edge, edge, polygon, original_edge, init_dir, swapped_dir, dist)

"""Constructs a list of the partial polygons from the list of edges."""
def construct_polygons(edges, grid=None, dist=0.5):
    polygons = []
    
    
    for edge in edges:
        num_new_polygons = 2
        for i in range(num_new_polygons):
            polygon = pp.Polygon(uid=len(polygons), edges=[])
            polygon.dir = 0
            polygon.add_edge(edge)
            
            
            search_along_edge(edge, None, polygon, original_edge=edge, init_dir=i, dist=dist)
            polygon.set_polygon_edges_direction()
            if len(polygon.get_corners()) >= 1:
                display_edges = polygon.get_edges()                
                polygons.append(polygon)
    return polygons

"""Completes the partial polygons based on various criteria. See the definition for complete_polygon in primitive_polygon.py"""
def get_completed_polygons(polygons, density, edges, display_grid, dist=0.5):
    completed_polygons = []
    for polygon in polygons:  

        completed_polygon = pp.Polygon(uid=len(completed_polygons), edges=[])

        new_edges, old_edges = polygon.complete_polygon(dist=dist)

        if len(new_edges) == 0:
            if polygon.is_closed():
                for edge in polygon.edges:
                    completed_polygon.add_edge(edge)
                completed_polygons.append(completed_polygon)
            continue
        display_edges = polygon.get_edges()
        for new_edge in new_edges:            
            if new_edge.get_length() > 0.3:
                completed_polygon.add_edge(new_edge)

        for edge in old_edges:
            completed_polygon.add_edge(edge)

        for edge in new_edges:
            for other_edge in completed_polygon.get_edges():
                edge.check_and_set_neighbor(other_edge, dist=dist)
        
        if completed_polygon.is_closed(dist=dist):
            completed_polygons.append(completed_polygon)
    return completed_polygons

"""Filters the list of completed polygons based on whether they form a valid shapely polygon."""
def filter_polys(completed_polygons):
    filtered_polys = []
    for polygon in completed_polygons:
        is_polygon = polygon.set_internal_polygon(dist=1.0)
        if is_polygon:
            filtered_polys.append(polygon)
    return filtered_polys

"""Computes the height of each polygon, by averaging the heights of the internal points."""
def extract_internal_poly_points(polygons, grid, dist):
    print("Computing polygon heights...")
    dims = grid.get_dimensions()
    for polygon in polygons:
        polygon.set_internal_polygon(dist=dist)
        polygon.set_polygon_center()
        polygon.reset_height()
    
    if len(polygons) == 0:
        return
        
    for i in range(dims[0]):
        for j in range(dims[1]):
            cell = grid.get_cell_at_location(i, j)
            if cell.get_count() == 0:
                continue
                            
            comp_point = [cell.get_comp_point()[0], cell.get_comp_point()[1]]
            
            for polygon in polygons:
                in_boundary = polygon.is_point_within_boundary(comp_point)
                if in_boundary:
                    polygon.update_height(cell.percentile_95, cell.get_count())

"""Determines the 25th and 75th percentile for point height along the edge, and ensures they are less than a threshold height difference.
Also ensures that the 25th percentile for point height is greater than the minimum wall height."""
def check_polygon_edge_percentiles(polygon, grid, height_diff=1.0, min_wall_height=1.0):
    for edge in polygon.get_edges():
        if edge.temp_edge:
            percentiles = phm.get_spatial_percentiles(edge, grid)
            if percentiles[1] - percentiles[0] > height_diff * 2.0 or percentiles[0] < min_wall_height:
                return False
    return True

"""Extracts the polygons with sufficient average point height, and whose edges return True for check_polygon_edge_percentiles."""
def get_valid_polygons(polygons, grid, minimum_wall_height):
    valid_polygons = []
    valid_polygon_centers = []
    invalid_polygon_centers = []
    for polygon in polygons:
        if polygon.is_closed():
            if polygon.average_point_height > 2.0 and check_polygon_edge_percentiles(polygon, grid, minimum_wall_height, minimum_wall_height):
                valid_polygons.append(polygon)
                valid_polygon_centers.append(polygon.get_polygon_center())
            else:
                invalid_polygon_centers.append(polygon.get_polygon_center())
    return valid_polygons, valid_polygon_centers, invalid_polygon_centers

"""Extracts the edges of the valid_polygons that were added as a result of their completion process, and adds them to the list of all edges.
Also ensures that the edges are not redundant, and splits edges that they intersect."""
def get_new_edges(valid_polygons, edges,dist=1.0):
    
    added_edges = []
    for i in range(len(valid_polygons)):
        polygon = valid_polygons[i]
        new_edges = []
        poly_edges = polygon.get_edges()
        for j in range(len(poly_edges)):
        
            edge = poly_edges[j]
            if edge.temp_edge:
                edge.join_with_nearby_edge(edges, polygon, dist=dist)
            
            all_edges = edges + added_edges
            
            redundant_edge = edge.check_new_edges(all_edges)
            if redundant_edge is None:
                
                added_edges.append(edge)
            else:
                if redundant_edge.edge_id == -1:
                    redundant_edge.edge_id = len(all_edges)                
                polygon.replace_edge(redundant_edge, edge)
                polygon.move_edge_endpoint(edge.get_start(), redundant_edge.get_start())
                polygon.move_edge_endpoint(edge.get_end(), redundant_edge.get_end())
                        
    all_edges = edges
    for edge in added_edges:
        edge.edge_id = len(all_edges)
        all_edges.append(edge)
        split_edges = edge.split_edges(all_edges)
        if len(split_edges) > 0:
            for split_edge in split_edges:
                if split_edge.get_length() > 0.1:
                    split_edge.edge_id = len(all_edges)
                    all_edges.append(split_edge)
    out_edges = []
    for edge in all_edges:
        if edge.get_length() > 0.1:
            edge.edge_id = len(out_edges)
            out_edges.append(edge)
    return out_edges

"""If extending an edge a short distance would cause it to intersect another edge, this is performed, and an additional edge is created."""
def post_process_edges(edges, min_wall_length):
    i = 0
    while i < len(edges):
        edge = edges[i]
        for j in range(len(edges)):
            if i == j:
                continue
            other_edge = edges[j]
            new_edges = edge.split_edge_intersect(other_edge, min_wall_length)
            edges = edges + new_edges
        i += 1
    return edges

"""Copies a list of edges by creating new edges with the same endpoints and edge id. Also assigns their neighbors."""
def deepcopy_edges(edges):
    dup_edges = []
    for edge in edges:
        dup_edge = pe.Edge(edge.get_start(), edge.get_end(), edge_id=edge.edge_id, temp_edge=edge.temp_edge)
        dup_edges.append(dup_edge)
        
    for edge in dup_edges:
        for other_edge in dup_edges:
            edge.check_and_set_neighbor(other_edge, 1.0)
    return dup_edges

"""Copies a list of polygons by creating new edges and new polygons."""
def deepcopy_polygons(polygons):
    dup_polygons = []
    edges = []
    for polygon in polygons:
        dup_edges = deepcopy_edges(polygon.get_edges())
        dup_polygon = pp.Polygon(polygon.id, dup_edges)
        dup_polygons.append(dup_polygon)
    return dup_polygons

"""Determines whether a polygon has already been discovered at a previous iteration, and if so, stores them in a list of old polygons.
Also filters polygons that were invalid at previous iterations."""
def filter_completed_polygons(new_completed, valid_polygon_centers, invalid_polygon_centers):
    new_polygons = []
    old_polygons = []
    for polygon in new_completed:
        away_from_centers = True        
        polygon_center = polygon.set_polygon_center()
        for valid_poly_center in valid_polygon_centers:
            dist = phm.distance(valid_poly_center, polygon_center)
            if dist < 0.5:
                old_polygons.append(polygon)
                away_from_centers = False
                break
        for invalid_polygon_center in invalid_polygon_centers:
            dist = phm.distance(invalid_polygon_center, polygon_center)
            if dist < 0.5:
                away_from_centers = False
                break
        if away_from_centers:
            new_polygons.append(polygon)
    return new_polygons, old_polygons
            
"""Ensures that equivalent edges have the same edge id across polygons."""
def label_polygon_edges(polygons):
    edge_list = []
    for polygon in polygons:
        for edge in polygon.get_edges():
            in_list = False
            for other_edge in edge_list:
                if edge == other_edge:
                    edge.edge_id = other_edge.edge_id
                    in_list = True
                    break
            if not in_list:
                edge.edge_id = len(edge_list)
                edge_list.append(edge)

"""Converts a set of line segments into a set of polygons through an iterative polygon completion process. This involves creating partial
polygons and adding additional edges to complete them, which has a domino effect that allows additional polygons to be completed."""
def iterative_polygon_detection(edge_list, density, grid, iterations=4, valid_centers = None, invalid_centers = None, dist=0.5, verbose=False):
    if valid_centers is None:
        valid_centers = []
    if invalid_centers is None:
        invalid_centers = []
    all_valid_polys = [] 
    print("Creating polygons...")
    for i in range(iterations):
        print("Iteration " + str(i))   
        edge_list = deepcopy_edges(edge_list)
        new_edge_list = []
        for edge in edge_list:
            if edge.get_length() > 0.8:
                new_edge_list.append(edge)
        edge_list = new_edge_list
        for edge in edge_list:
            edge.reset_edge()
        for edge in edge_list:
            for other_edge in edge_list:
                edge.check_and_set_neighbor(other_edge, dist=dist)  
        polygons = construct_polygons(edge_list, grid, dist=dist)
        next_polygons = get_completed_polygons(polygons, density, edge_list, grid, dist=dist)
        next_polygons = filter_polys(next_polygons)
        new_polygons, old_polygons = filter_completed_polygons(next_polygons, valid_centers, invalid_centers) 
        extract_internal_poly_points(new_polygons, grid, dist=0.8)
        valid_polygons, next_valid_centers, next_invalid_centers = get_valid_polygons(new_polygons, grid, minimum_wall_height)
        valid_centers = valid_centers + next_valid_centers
        invalid_centers = invalid_centers + next_invalid_centers
        all_valid_polys = old_polygons + valid_polygons
        if verbose:
            print("Number of valid polygons detected: " + str(len(valid_polygons)))
            plotting.plot_polygons(grid, valid_polygons)
        if len(valid_polygons) == 0:
            break
        duplicate_edges = deepcopy_edges(edge_list)
        duplicate_polygons = deepcopy_polygons(valid_polygons)
        new_edge_list = get_new_edges(duplicate_polygons, duplicate_edges, dist=1.0)
        new_edge_list = post_process_edges(new_edge_list, minimum_wall_length)
        edge_list = new_edge_list
    label_polygon_edges(all_valid_polys)
    return edge_list, all_valid_polys, valid_centers, invalid_centers

"""Converts each polygon to a merged polygon class, and returns the resulting list of merged polygons."""
def construct_merged_polygons(polygons):    
    merged_polygons = []
    for polygon in polygons:
        merged_polygon = pp.MergedPolygon([polygon], len(merged_polygons))
        merged_polygons.append(merged_polygon)
    return merged_polygons

"""Joins 'merged' polygons recursively. If two merged polygons are adjacent and have similar average heights, they are combined."""
def get_merged_polygons(merged_polygons):
    final_polygons = []
    has_merged_in_iter = True
    while has_merged_in_iter == True:
        has_merged_in_iter = False
        for merged in merged_polygons:
            if merged.to_destroy:
                continue 
            for other_merged in merged_polygons:
                if other_merged == merged or other_merged.to_destroy:
                    continue
                b_merged = merged.absorb_merged_polygon(other_merged)
                has_merged_in_iters = has_merged_in_iter or b_merged
    for merged in merged_polygons:
        if not merged.to_destroy:
            merged_edges = merged.get_outer_edges()
            for edge in merged_edges:
                for other_edge in merged_edges:
                    edge.check_and_set_neighbor(other_edge, dist=0.8)
            final_polygons.append(merged)
    return final_polygons


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default="Data")
    parser.add_argument('--point_cloud_file', type=str, default="points.txt")
    parser.add_argument('--density_file', type=str, default="density.txt")
    args = parser.parse_args()
    output_dir = args.data_folder
    density_filename = args.density_filename
    if output_dir[-1] != "/":
        output_dir = output_dir + "/"
    point_cloud_file = output_dir + args.point_cloud_file
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    with open(output_dir + density_filename, 'r') as f:
        density = float(f.readline())

    display_grid, points, offset = construct_grid(point_cloud_file)
    new_segs = load_lines(output_dir + "line_segments.txt")

    edges = []
    for i in range(len(new_segs)):
        seg = new_segs[i]
        edge = pe.Edge(seg.get_start(), seg.get_end(), i)
        edges.append(edge) 
    edges, co_p, valid_centers, invalid_centers = iterative_polygon_detection(edges, density, display_grid, 8, dist=0.8)
    polygons = deepcopy_polygons(co_p)
    extract_internal_poly_points(polygons, display_grid, dist=0.8)

    for polygon in polygons:
        polygon.merged = False
    merged_polygons = construct_merged_polygons(polygons)
    merged_polygons = get_merged_polygons(merged_polygons)
    merged_polygons = get_merged_polygons(merged_polygons)

    out_filename = output_dir + "merged_polygons.txt"
    fw.save_merged_polygon_points(merged_polygons, offset, filename=out_filename)
    
if __name__ == "__main__":
    main()

