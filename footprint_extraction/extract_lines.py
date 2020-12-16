import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from line_extraction_primitives.cell import Cell
from line_extraction_primitives.line import Line, load_lines, save_lines
from line_extraction_primitives.grid import Grid
import line_extraction_primitives.extraction_helper_methods as ehm

import plotting
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage import exposure

import argparse

minimum_wall_height = 2.0
minimum_wall_length = 2.0
line_cell_width = 0.3
line_cell_height = 0.3
density_coeff = 0.2

"""Constructs lines from hough parameters (angles and distances)"""
def construct_lines(angles, dists):
    lines = []
    for i in range(len(angles)):
        next_line = Line()
        next_line.set_hough_params(angles[i], dists[i])
        lines.append(next_line)
    return lines

"""Removes the points around a line segment from the grid."""
def remove_line_points_from_grid(grid, line_segment):
    line_seg_points = grid.remove_points_from_line_segment(line_segment)
    new_counts = grid.get_counts_image()
    return line_seg_points

"""Returns the input points split into two arrays, the points surrounding the line segments (within a small distance), and the remaining points."""
def remove_line_seg_points(points, line_segments):
    line_segment_points = []
    grid = Grid(points)
    for line_seg in line_segments:
        next_point_list = remove_line_points_from_grid(grid, line_seg)
        line_segment_points.append(next_point_list)
    remaining_points = grid.get_all_points()
    return line_segment_points, remaining_points



"""Transforms a set of line segments into grid space from world space."""
def transform_line_segments(grid, line_segments):
    transformed_line_segments = []
    for each in line_segments:
        transformed_line_segments.append(grid.transform_line_to_grid_space(each))
    return transformed_line_segments

"""Constructs a grid from a set of points, determines lines in the grid, and filters the lines into line segments."""
def get_lines_from_points(iter_points, density, min_wall_height, min_wall_length, verbose=False):
    points = iter_points.copy()

    #Creates a new grid from the current set of points
    grid = Grid(points)

    #Removes all points below a certain height from consideration
    grid.filter_cell_points_by_height(min_wall_height)
    points = grid.get_all_points()
    
    #If density has not yet been determined, use a hard-coded threshold
    if density < 0.0:
        thresh = 15.0
        use_density = False
    else:
        min_wall_points = density * minimum_wall_height
        thresh_height_ratio = min_wall_points / np.amax(grid.get_counts())
        thresh = get_thresh_from_height_ratio(thresh_height_ratio, grid.cell_width, grid.cell_height)
        use_density = True
    
    #Compute the hough transform from the thresh. Thresh has been multiplied by 3 since it is preferable to avoid spurious lines,
    #and detect any missed lines in the polygonization step
    counts, out, theta, rho = grid.hough_from_grid(thresh=thresh * 2.0)
    angles, dists = ehm.get_hough_peaks(out, theta, rho)
    lines = construct_lines(angles, dists)

    if verbose:
        plotting.draw_lines(grid.get_counts_image(), lines)
    #Orders the detected lines by the number of points that corresponds to them. This ensures that if the Hough extraction yielded
    # two lines that are very similar, we only add the one that is better supported
    ordered_lines, lines_points = grid.get_best_line_ordering(lines)
    filtered_lines, filtered_points = grid.filter_lines(ordered_lines, lines_points)

    #Extracts the line segments from the detected lines
    line_segments, lines = get_filtered_segments(filtered_points, density, grid, min_wall_height=min_wall_height, min_wall_length=min_wall_length, use_density=use_density)
    return grid, line_segments, lines

"""Determines a pixel brightness threshold from the thresh_height_ratio."""
def get_thresh_from_height_ratio(height_ratio, cell_width=0.1, cell_height=0.1):
    thresh = (cell_width * cell_height * height_ratio * 255.0)
    return thresh

"""Computes a density for the lines that are present in a set of points."""
def compute_line_density(points, min_wall_height=2.0, min_wall_length=2.0):
    grid, line_segs, lines = get_lines_from_points(points, density=-1.0, min_wall_height=min_wall_height, min_wall_length=min_wall_length, verbose=True)
    transformed_line_segments = transform_line_segments(grid, line_segs)

    #Calculates density from the line segment with the greatest amount of support
    line_seg_points, remaining_points = remove_line_seg_points(points, [transformed_line_segments[-1]])
    
    density = ehm.get_density(line_seg_points)
    return density

"""Iteratively detects lines from a set of points. Uses the hough line transform to first detect lines,
then converts these lines into line segments, based on the density of different sections of the line.
Points corresponding to the line are then removed, and the process is repeated, until no new lines are detected."""
def iterative_line_detection(points, density, min_wall_height=1.0, min_wall_length=1.0, iterations=20, verbose=False):
    line_segments = []
    remaining_points = points
    grid = None
    continue_running = True
    
    for _ in range(iterations):
        print("Iteration: " + str(_))
        if not continue_running:
            break
#         minimum_wall_height = min_wall_height * (2.0 + iterations - i)
        grid, line_segs, lines = get_lines_from_points(remaining_points, density=density, min_wall_height=minimum_wall_height, min_wall_length=min_wall_length, verbose=verbose)
        transformed_lines = transform_line_segments(grid, line_segs)
        line_seg_points, remaining_points = remove_line_seg_points(remaining_points, transformed_lines)
        
        if (len(line_segs) == 0):
            continue_running = False
        
        line_segments += line_segs

        #Plots the detected lines and line segments on the grid image
        if verbose:
            plotting.plot_line_segments_on_grid(grid, line_segs)
            plotting.plot_line_segments_on_grid(grid, line_segments)
            print(len(line_segs))
            
    for i in range(len(line_segments)):
        line_segments[i].id = i
        
    return grid, line_segments, line_seg_points, remaining_points


"""Constructs line segments from lines based on the underlying support/density of their points. This function calculates the median filter size, before running get_filtered_line_segments."""
def get_filtered_segments(line_points, density, orig_grid, min_wall_height, min_wall_length, cell_width=0.1, cell_height=0.1, use_density=False):
    line_segments = []
    lines = []
    for points in line_points:
        if len(points) == 0:
            continue
        min_x, min_y, min_z = np.amin(points, axis=0)
        max_x, max_y, max_z = np.amax(points, axis=0)
        height = max(max_z - min_z, minimum_wall_height)
        
        if use_density:
            #Calculates a value to threshold the line pixels, based on the minimum number of wall points required per line, based on the density and minimum wall height
            thresh_height_ratio = minimum_wall_height * density * density_coeff
        else:
            #When no density is available (i.e. no lines have been extracted yet), uses a set height ratio
            thresh_height_ratio=100.0
        med_filter_size = math.floor(min(min_wall_length / cell_width, min_wall_height / cell_height) * 2) + 1
        line_segs, line = get_filtered_line_segments(points, orig_grid, thresh_height_ratio, med_filter_size)
        line_segments += line_segs
        lines.append(line)
    return line_segments, lines

"""Applies a 1D median filter to the line. Assumes the line has been rotated to be vertical."""
def median_filter_line(line_grid, line, med_filter_size=11, thresh=500.0):

    max_points = max(np.amax(line_grid.get_counts()), thresh)
    im_thresh = (thresh / max_points) * 255.0
    counts_image = line_grid.get_counts_image()

    vertical_med_image = ehm.median_filter_image(counts_image, med_filter_size)
    thresholded_im = ehm.threshold_image(vertical_med_image, im_thresh)
#     vertical_med_image = hm.threshold_image(vertical_med_image, thresh=30)
    next_lines = line.extract_segments(thresholded_im)
    return next_lines

"""Constructs line segments from lines based on the underlying support/density of their points, and ensures that sections are above the minimum wall length and height."""
def get_filtered_line_segments(line_points, orig_grid, thresh_height_ratio, med_filter_size):    
    line = Line()


    #(Marginally) improves how well the line fits the points, by applying RANSAC
    line.line_from_ransac(line_points)
    
    #Rotates the line to the vertical, in order to apply the 1D median filter
    theta = line.find_rotation_angle()
    rotation_point = line.get_start()
    points_to_rotate = np.delete(line_points, 2, 1).tolist()
    rot_points = ehm.rotate_about_point(points_to_rotate, theta, rotation_point)
    rot_line = line.rotate_line(theta, rotation_point)
    
    
    #project points onto the rotated line
    projected_points = rot_line.project_points_onto_line(rot_points)

    #Create a grid from the rotated line, with 1 cell width and a height of
    # the length of the line / line_cell_height  
    grid = Grid(projected_points, line_cell_width, line_cell_height)
    
    #Transforms the rotated line to grid space
    converted_line = grid.transform_line_to_grid_space(rot_line)

    #Applies a threshold to the grid, to determine which cells have a sufficient number of points
    thresh = (line_cell_width * line_cell_height * thresh_height_ratio)
    
    #Applies a '1D' median filter to the line to filter out cells, and extracts contiguous segments 
    t_rot_line_segments = median_filter_line(grid, converted_line, med_filter_size, thresh)
    line_segments = []
    for t_rot_line_segment in t_rot_line_segments:
        rot_line_segment = grid.transform_line_to_world_space(t_rot_line_segment)
        line_segment = rot_line_segment.rotate_line(-1.0 * theta, rotation_point)
        line_segments.append(line_segment)
    
    return line_segments, line



"""Constructs the grid from a set of points."""
def construct_grid(filename):
    points = ehm.get_points_from_file(filename)
    # points = hm.project_points(points)
    points, offset = ehm.subtract_point_min(points)
    display_grid = Grid(points)
    return display_grid, points, offset

"""Post process the line segments. This moves the endpoints of different line segments that are a short distance away to the same location.
Additionally, if a line segment can be extended so that it intersects another line segment, this is performed, and the other (i.e. non-extended)
line segment is split into two pieces. Finally, small line segments (i.e. <1m) are removed."""

def post_process_segments(line_segments):
    processed_line_segments = line_segments.copy()
    i = 0
    #Iteratively process line segments, including those that are created as a result of previous iterations 
    while i < len(processed_line_segments):
        line_segment = processed_line_segments[i]
        new_line_segments = []
        other_counter = 0
            
        for j in range(len(processed_line_segments)):
            other_segment = processed_line_segments[j]
            if other_segment == line_segment:
                continue 
            
            #Checks whether the current line segment can be extended to intersect another line segment
            new_lines = line_segment.check_line_proximity(other_segment)
            for line in new_lines:
                new_line_segments.append(line)
            other_counter += 1
            
        processed_line_segments += new_line_segments
        i += 1
                
    for line_segment in processed_line_segments:
        for other_segment in processed_line_segments:
            if other_segment == line_segment:
                continue

            #Attracts the endpoints of the line segments together, within a specified range
            int_point = line_segment.attract_line_endpoints(other_segment, 1.5)
    
    for i in reversed(range(len(processed_line_segments))):
        if processed_line_segments[i].get_length() < 1.0:
            del processed_line_segments[i]
            
    return processed_line_segments

"""Attracts the endpoints of a pair of line segments together."""
def attract_endpoints(segments, dist=1.0):
    for line_segment in segments:
        line_segment.set_end_movable(True)
        line_segment.set_start_movable(True)
    for line_segment in segments:
        int_point = line_segment.attract_line_endpoints2(segments, dist)

"""Iterates through all line segments and removes duplicates (i.e. if both of their endpoints are close together)."""
def delete_duplicate_segments(segments):
    new_segment_list = []
    for segment in segments:
        to_add = True
        start = segment.get_start()
        end = segment.get_end()
        for other_segment in new_segment_list:
            duplicate = False

            #Checks if the endpoints match
            if ehm.compare_points(other_segment.get_start(), start, 0.4) and \
                ehm.compare_points(other_segment.get_end(), end, 0.4):
                duplicate = True
            elif ehm.compare_points(other_segment.get_start(), end, 0.4) and \
                ehm.compare_points(other_segment.get_end(), start, 0.4):
                duplicate = True
            
            if duplicate:
                to_add = False
        if to_add:
            new_segment_list.append(segment)
    #Attracts the endpoints of the line segments together after removing duplicates, since duplicates interfere with this process
    attract_endpoints(new_segment_list)
    return new_segment_list

"""If a segment engulfs another segment, the other segment is removed."""
def delete_overlapping_segments(segments, minimum_wall_length=1.0):
    new_segment_list = []
    skip_indices = []
    for i in range(len(segments)):
        to_add = True
        segment = segments[i]
        start = segment.get_start()
        end = segment.get_end()

        for j in range(len(segments)):
            #All edges in the skip_indices list are deleted
            if i == j or j in skip_indices or i in skip_indices:
                continue
            other_segment = segments[j]
            
            #Compares the endpoints of the two segments
            start_shared = ehm.compare_points(other_segment.get_start(), start, minimum_wall_length) or \
                ehm.compare_points(other_segment.get_end(), start, minimum_wall_length)
            end_shared = ehm.compare_points(other_segment.get_start(), end, minimum_wall_length) or \
                ehm.compare_points(other_segment.get_end(), end, minimum_wall_length)
            
            #Checks to see if the current edge is engulfed by the other edge
            if start_shared and other_segment.on_line(end, epsilon=minimum_wall_length):
                to_add = False
            elif end_shared and other_segment.on_line(start, epsilon=minimum_wall_length):
                to_add = False
            elif other_segment.on_line(start, epsilon=minimum_wall_length) and \
                other_segment.on_line(end, epsilon=minimum_wall_length):
                to_add = False
        if to_add:
            new_segment_list.append(segment)
        else:
            skip_indices.append(i)
    attract_endpoints(new_segment_list)
    return new_segment_list

#TODO: Adjust filenames, savelines,
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default="Data")
    parser.add_argument('--point_cloud_file', type=str, default="points.txt")
    args = parser.parse_args()
    output_dir = args.data_folder
    if output_dir[-1] != "/":
        output_dir = output_dir + "/"

    point_cloud_file = output_dir + args.point_cloud_file
    density_filename = "density.txt"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    display_grid, points, offset = construct_grid(point_cloud_file)
    density = compute_line_density(points, minimum_wall_height, minimum_wall_length)
    with open(output_dir + density_filename, 'w') as f:
        f.write(str(density))

    np.savetxt(output_dir + 'polygon_offset.txt', offset.reshape((1,3)), delimiter=',')
    
    print("Density is: " + str(density))
    grid, all_segs, line_seg_points, remaining_points = iterative_line_detection(points, density, minimum_wall_height, minimum_wall_length, iterations=20)    
    im = display_grid.get_thresholded_counts()
    new_segs = post_process_segments(all_segs)
    new_segs = delete_duplicate_segments(new_segs)
    new_segs = delete_overlapping_segments(new_segs) 
    save_lines(new_segs, output_dir + "line_segments.txt")

if __name__ == "__main__":
    main()

