import matplotlib.pyplot as plt

from line_extraction_primitives.grid import Grid
from line_extraction_primitives.line import Line
import line_extraction_primitives.extraction_helper_methods as hm 

from skimage.transform import (hough_line, hough_line_peaks)
import seaborn as sns
import numpy as np


def draw_lines(image, lines):
    image = (image / np.amax(image))
    image = np.array(image * 255, dtype = np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(image, cmap="gray")
    for line in lines:
        angle, dist = line.get_hough_params()
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
        ax[1].plot((0, image.shape[1]), (y0, y1), '-r')
    ax[1].set_xlim((0, max(image.shape[0], image.shape[1])))
    ax[1].set_ylim((max(image.shape[0], image.shape[1]), 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected lines')

    plt.tight_layout()
    plt.show()

def plot_line_segments_on_grid(grid, line_segments, thresh=0.0):
    if thresh > 0.0:
        image = grid.get_thresholded_counts(thresh)
    else:
        image = grid.log_transform_counts_image()
        
    transformed_line_segments = []
    for each in line_segments:
        transformed_line_segments.append(grid.transform_line_to_grid_space(each))
    draw_boundaries(image, transformed_line_segments)

def construct_heatmap_heights(grid):
    plt.figure(num=None, figsize=(15, 6), facecolor='w', edgecolor='k')

    dims = grid.get_dimensions()
    height_vals = np.zeros(dims)

    image = grid.log_transform_counts_image()

    for i in range(dims[0]):
        for j in range(dims[1]):
            cell = grid.get_cell_at_location(i, j)
            if cell.get_count() == 0:
                height_vals[i, j] = 0.0
                continue
                            
            height_vals[i,j] = cell.percentile_95 

    plt.imshow(height_vals, cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def construct_heatmap_polygons(polygons, grid):
    dims = grid.get_dimensions()
    height_vals = np.zeros(dims)
    plt.figure(num=None, figsize=(15, 6), facecolor='w', edgecolor='k')
    
    image = grid.log_transform_counts_image()
    for polygon in polygons:
        polygon.set_internal_polygon()
    for i in range(dims[0]):
        for j in range(dims[1]):
            cell = grid.get_cell_at_location(i, j)
            if cell.get_count() == 0:
                height_vals[i, j] = 0.0
                continue
                            
            comp_point = [cell.get_comp_point()[0], cell.get_comp_point()[1]]
            
            in_poly = False
            for polygon in polygons:
                
                in_boundary = polygon.is_point_within_boundary(comp_point)
                if in_boundary:
                    height_vals[i,j] = polygon.average_point_height
                    # print(polygon.average_point_height)
                    in_poly = True
                    break
            if not in_poly:
                height_vals[i, j] = 0.0

    plt.imshow(height_vals, cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def plot_polygons(grid, polygons):
    edges = []
    for polygon in polygons:
        edges += polygon.get_edges()
    plot_line_segments_on_grid(grid, edges)
    
def construct_heatmap_merged_polygons(polygons, grid):
    dims = grid.get_dimensions()
    height_vals = np.zeros(dims)
    plt.figure(num=None, figsize=(15, 6), facecolor='w', edgecolor='k')
    
    image = grid.log_transform_counts_image()
    
    for i in range(dims[0]):
        for j in range(dims[1]):
            cell = grid.get_cell_at_location(i, j)
            if cell.get_count() == 0:
                height_vals[i, j] = 0.0
                continue
                            
            comp_point = [cell.get_comp_point()[0], cell.get_comp_point()[1]]
            
            in_poly = False
            for polygon in polygons:
                
                in_boundary = polygon.point_in_merged_polygon(comp_point)
                if in_boundary:
                    height_vals[i,j] = polygon.average_height
                    # print(polygon.average_point_height)
                    in_poly = True
                    break
            if not in_poly:
                height_vals[i, j] = 0.0

    plt.imshow(height_vals, cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def plot_heatmap(grid):
    grid_dim = grid.get_dimensions()
    counts = np.zeros(grid_dim)
    for i in range(grid_dim[0]):
        for j in range(grid_dim[1]):
            count = grid.get_count(j, i)
            counts[i, j] = count
    # ax = sns.heatmap(counts, linewidth=0.05)
    plt.imshow(counts, cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.show()

def plot_polygon_height(polygon, grid):
    dims = grid.get_dimensions()
    heights = []

    image = grid.log_transform_counts_image()
    transformed_line_segments = []
    for each in polygon.get_edges():
        transformed_line_segments.append(grid.transform_line_to_grid_space(each))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].imshow(image, cmap="gray")
    if len(lines) > 0:
        for line_segment in lines:
            y0 = line_segment.get_start()[1]
            y1 = line_segment.get_end()[1]
            ax[0].plot((line_segment.get_start()[0], line_segment.get_end()[0]), (y0, y1), '-r')
        ax[0].set_xlim((0, image.shape[1]))
        ax[0].set_ylim((image.shape[0], 0))
    ax[0].set_axis_off()
    ax[0].set_title('Detected lines')

    

    polygon.set_internal_polygon()

        
    for i in range(dims[0]):
        for j in range(dims[1]):
            cell = grid.get_cell_at_location(i, j)
            if cell.get_count() == 0:
                continue
                            
            comp_point = [cell.get_comp_point()[0], cell.get_comp_point()[1]]
            
            in_boundary = polygon.point_within_boundary(comp_point)
            if in_boundary:
                heights.append(cell.percentile_95)

    ax[1].hist(heights)

    plt.tight_layout()
    plt.show()

def plot_height_along_edge(edge, grid):
    image = grid.log_transform_counts_image()
    

    line = fp.Line()
    line.set_points(edge.get_start(), edge.get_end())
    transformed_line = grid.transform_line_to_grid_space(line)

    edge_length = edge.get_length()

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
        rotated_points = hm.rotate_line_about_point(points, theta, rotation_point)

        line_grid = fp.Grid(rotated_points, width * 2, height * 2)    

        heights = fp.get_heights(line_grid.get_cells()[3:-3])
    else:
        heights = [0.0]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].imshow(image, cmap="gray")
    y0 = transformed_line.get_start()[1]
    y1 = transformed_line.get_end()[1]
    ax[0].plot((transformed_line.get_start()[0], transformed_line.get_end()[0]), (y0, y1), '-r')
    ax[0].set_xlim((0, image.shape[1]))
    ax[0].set_ylim((image.shape[0], 0))
    ax[0].set_axis_off()
    ax[0].set_title('Input lines')
    

    ax[1].plot([0.5*i for i in range(len(heights))], heights)
    ax[1].set_ylim(ymin=0)
    plt.tight_layout()
    plt.show()

def plot_grid_3d(grid):
    grid_dim = grid.get_dimensions()
    point_arr = []
    for j in range(grid_dim[0]):
        for i in range(grid_dim[1]):
            points = grid.get_points(i, j)
            for each in points:
                p = np.array([each[0], each[1], 0.0])
                point_arr.append(p)
    point_arr = np.array(point_arr)
    v = pptk.viewer(point_arr)

def draw_line_ordering(img, lines):
    for i in range(len(lines)):
        to_draw = [lines[j] for j in range(len(lines)) if j <= i]
        draw_lines(img, to_draw)

def draw_boundaries(image, lines_start, lines_end):
    image = (image / np.amax(image))
    image = np.array(image * 255, dtype = np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(image, cmap="gray")
    for i in range(len(lines_start)):
        y0 = lines_start[i][1]
        y1 = lines_end[i][1]
        ax[1].plot((lines_start[i][0], lines_end[i][0]), (y0, y1), '-r')
    ax[1].set_xlim((0, image.shape[1]))
    ax[1].set_ylim((image.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected lines')

    plt.tight_layout()
    plt.show()

def draw_boundaries(image, lines):

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(image, cmap="gray")
    if len(lines) > 0:
        for line_segment in lines:
            y0 = line_segment.get_start()[1]
            y1 = line_segment.get_end()[1]
            ax[1].plot((line_segment.get_start()[0], line_segment.get_end()[0]), (y0, y1), '-r')
        ax[1].set_xlim((0, image.shape[1]))
        ax[1].set_ylim((image.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected lines')

    plt.tight_layout()
    plt.show()

def draw_boundary(image, line_segment):
    # image = (image / np.amax(image))
    # image = np.array(image * 255, dtype = np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(image, cmap="gray")
    y0 = line_segment.get_start()[1]
    y1 = line_segment.get_end()[1]
    ax[1].plot((line_segment.get_start()[0], line_segment.get_end()[0]), (y0, y1), '-r')
    ax[1].set_xlim((0, image.shape[1]))
    ax[1].set_ylim((image.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected lines')

    plt.tight_layout()
    plt.show()

def plot_hough_peaks(counts, h, theta, d):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(counts, cmap="gray")
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(np.log(1 + h),
                 extent=[np.rad2deg(theta[0]), np.rad2deg(theta[-1]), d[-1], d[0]],
                 cmap="gray", aspect=1/1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')
    for angle, dist in zip(*hm.get_hough_peaks(h, theta, d)):
        ax[1].plot(np.rad2deg(angle), dist, 'ro')
    
    ax[2].imshow(counts, cmap="gray")
    for angle, dist in zip(*hm.get_hough_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - counts.shape[1] * np.cos(angle)) / np.sin(angle)
        ax[2].plot((0, counts.shape[1]), (y0, y1), '-r')
    ax[2].set_xlim((0, counts.shape[1]))
    ax[2].set_ylim((counts.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    plt.tight_layout()
    plt.show()
    
def plot_image(image):

    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.show()

def show_median_filtering(image1, image2):
    # image1 = (image1 / np.amax(image1))
    # image1 = np.array(image * 255, dtype = np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(image1, cmap="gray")
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(image2, cmap="gray")
    ax[1].set_axis_off()
    ax[1].set_title('After Median Filtering:')

    plt.tight_layout()
    plt.show()