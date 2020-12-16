import line_extraction_primitives.extraction_helper_methods as hm
from scipy import ndimage
from scipy.signal import medfilt
from line_extraction_primitives.cell import Cell
from line_extraction_primitives.line import Line
import numpy as np
import math
        
class Grid:
    """Initialize the grid from a set of points, with a specified cell width and height. Grid is square and extent of the grid is calculated from the 
    minimimum and maximum x,y coordinates of the points within the grid."""
    def __init__(self, points, cell_width=0.1, cell_height=0.1):
        #determine resolution of the grid

        #Cell width measured in m
        self.cell_width = cell_width
        self.cell_height = cell_height

        if len(points[0]) == 3:
            min_x, min_y, min_z = np.amin(points, axis=0)
            max_x, max_y, max_z = np.amax(points, axis=0)
        else:
            min_x, min_y = np.amin(points, axis=0)
            max_x, max_y = np.amax(points, axis=0)

        height = max_y - min_y
        width = max_x - min_x

        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

        self.grid_height = max(math.ceil(height / self.cell_height), 1)
        self.grid_width = max(math.ceil(width / self.cell_width), 1)
        self.x_offset = min_x
        self.y_offset = min_y
        self.cells = [[Cell(i, j) for i in range(self.grid_width)] for j in range(self.grid_height)]

        for p in points:
            self.add_point_to_cell(p)

        for i in range(self.grid_height):
            for j in range(self.grid_width):
                cell = self.get_cell_at_location(i, j)
                cell.set_average_height()

    """Deletes points in the cells that are below a specified minimum height."""
    def filter_cell_points_by_height(self, min_height):
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                cell = self.get_cell_at_location(i, j)
                if cell.get_height_diff() < min_height:
                    cell.delete_points()

    """Gets the dimensions/extent of the grid."""
    def get_dimensions(self):
        return (self.grid_height, self.grid_width)

    """Adds a point to the cell it corresponds to in the grid."""
    def add_point_to_cell(self, point):
        x = point[0]
        y = point[1]
        index_w = math.floor((x - self.x_offset) / self.cell_width)
        index_h = math.floor((y - self.y_offset) / self.cell_height)

        self.cells[index_h][index_w].add_point(point)

    """Converts a point in grid space to world space."""
    def convert_point_to_world_space(self, point):
        x = point[0] * self.cell_width + self.x_offset
        y = point[1] * self.cell_height + self.y_offset
        return [x, y]

    """Calculates the cell indices from the point's x, y coordinates."""
    def interp_cell_location_from_point(self, point):
        cell_z = 0
        x, y = point[0], point[1]
        if len(point) == 3:
            cell_z = point[2]
        cell_x = math.floor((x - self.x_offset) / self.cell_width)
        cell_y = math.floor((y - self.y_offset) / self.cell_height)
        return cell_x, cell_y, cell_z
    
    """Returns the number of points at a particular cell index."""
    def get_count(self, index_w, index_h):
        return self.cells[index_h][index_w].get_count()

    """Returns the points at a particular cell index."""
    def get_points(self, index_w, index_h):
        return self.cells[index_h][index_w].get_points()

    """Returns all points in the grid."""
    def get_all_points(self):
        dim = self.get_dimensions()
        points = []
        for i in range(dim[0]):
            for j in range(dim[1]):
                cell_points = self.get_points(j, i)
                points += cell_points 
        return points

    """Returns the projected points of a cell at a specified index."""
    def get_projected_points(self, index_w, index_h):
        return self.cells[index_h][index_w].get_projected_points()

    """Returns all projected points in the grid."""
    def get_all_projected_points(self):
        points = []
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                next_points = self.get_projected_points(j, i)
                points += next_points
        return points

    """Returns all the cells in the grid."""
    def get_cells(self):
        cells = []
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                cells.append(self.get_cell_at_location(i,j))
        return cells

    """Retuns the cell at index i,j."""
    def get_cell_at_location(self, i, j):
        if i < 0 or i >= self.grid_height or j < 0 or j >= self.grid_width:
            return None
        return self.cells[i][j]

    """Returns the points along the line, by checking grid cells. Line uses hough parameters."""
    def get_points_from_line(self, line, width=0.8):
        theta, rho = line.get_hough_params()
        num_cells_line_width = round(width / self.cell_width)
        num_cells_line_height = round(width / self.cell_height)
        entered_grid = False
        points = []
        found_cells = set()
        for j in range(self.grid_width):
            #Check cells up to num_cells_line_width around the line
            for rho_width_index in range(-num_cells_line_width, num_cells_line_width):
                y = (rho + rho_width_index - j * np.cos(theta)) / np.sin(theta)
                i = math.floor(y)
                cell = self.get_cell_at_location(i, j)
                if cell is not None and not cell.is_visited():
                    found_cells.add(cell)
        
        for i in range(self.grid_height):
            #Check cells up to num_cells_line_width around the line
            for rho_height_index in range(-num_cells_line_height, num_cells_line_height):
                x = (rho + rho_height_index - i * np.sin(theta)) / np.cos(theta)
                j = math.floor(x)
                cell = self.get_cell_at_location(i, j)
                if cell is not None and not cell.is_visited():
                    found_cells.add(cell)

        range_x = [self.min_x, self.max_x]
        range_y = [self.min_y, self.max_y]
        for cell in found_cells:
            #For each potential cell, add the points within a certain distance to the line
            for point in cell.get_points():
                x, y, z = self.interp_cell_location_from_point(point)
                if line.get_point_proximity([x, y], range_x, range_y) < width:
                    points += [point]
            # points = points + cell.get_projected_points()
        return points

    """Gets the cells around a line segment, using x,y coordinates."""
    def get_cells_around_line_segment(self, line_segment, width):
        if width == float("inf") or line_segment is None or line_segment.get_start() is None or line_segment.get_end() is None:
            return []
        num_cells_line_width = int(width / self.cell_width)
        found_cells = set()
        start_point = line_segment.get_start()
        end_point = line_segment.get_end()

        #Determines the change in x and y coordinates for the two endpoints
        delta_x = end_point[0] - start_point[0]
        delta_y = end_point[1] - start_point[1]

        #Gets the x and y slopes for the line segment
        if abs(delta_x) > 0.0:
            y_slope = delta_y / delta_x
        else:
            y_slope = 0.0

        if abs(delta_y) > 0.0:
            x_slope = delta_x / delta_y
        else:
            x_slope = 0.0

        if start_point[0] > end_point[0]:
            y_start = end_point[1]
            j_start = math.floor(end_point[0])
            j_end = math.floor(start_point[0])
        else: 
            y_start = start_point[1]
            j_start = math.floor(start_point[0])
            j_end = math.floor(end_point[0])

        for j in range(j_start, j_end):
            #Check cells up to num_cells_line_width around the line
            #TODO: Instead of looking within a certain x or y range, should choose a range perpendicular to the line
            for grid_x in range(-num_cells_line_width, num_cells_line_width):
                if j + grid_x < 0:
                    continue
                
                y = (j - j_start) * y_slope + grid_x + y_start
                i = math.floor(y)
                cell = self.get_cell_at_location(i, j)
                if cell is not None and not cell.is_visited():
                    found_cells.add(cell)
        
        if start_point[1] > end_point[1]:
            x_start = end_point[0]
            i_start = math.floor(end_point[1])
            i_end = math.floor(start_point[1])
        else: 
            x_start = start_point[0]
            i_start = math.floor(start_point[1])
            i_end = math.floor(end_point[1])

        for i in range(i_start, i_end):
            #Check cells up to num_cells_line_width around the line
            for grid_y in range(-num_cells_line_width, num_cells_line_width):
                if i + grid_y < 0:
                    continue
                
                x = (i - i_start) * x_slope + grid_y + x_start
                j = math.floor(x)
                cell = self.get_cell_at_location(i, j)
                if cell is not None and not cell.is_visited():
                    found_cells.add(cell)
        return found_cells

    """Gets the points around a line segment, using x,y coordinates, at a particular perpendicular distance from the line."""
    def get_points_from_line_segment(self, line_segment, width=1.0):
        found_cells = self.get_cells_around_line_segment(line_segment, width)
        points = []
        for cell in found_cells:
            points = points + cell.get_points()
        return np.array(points)

    """Removes points in the grid that correspond to a line segment, at a particular perpendicular distance from the line."""
    def remove_points_from_line_segment(self, line_segment, width=1.0):
        found_cells = self.get_cells_around_line_segment(line_segment, width)
        points = []
        num_points = 0
        for cell in found_cells:
            #For each potential cell, add the points within a certain distance to the line
            cell_points = cell.get_points()
            num_points += len(cell_points)
            for i in reversed(range(len(cell_points))):
                x, y, z = self.interp_cell_location_from_point(cell_points[i])
                # if line_segment.get_point_proximity([x, y]) < width:
                points += [cell.pop_point(i)]
            # points = points + cell.get_points()
        return np.array(points)

    """Flags a cell as 'visited'. Used for traversal."""
    def set_line_cells_visited(self, line, width=0.7):
        theta, rho = line.get_hough_params()
        num_cells_line_width = round(width / self.cell_width)
        entered_grid = False
        points = []
        found_cells = set()

        for j in range(self.grid_width):
            #Check cells up to num_cells_line_width around the line
            for rho_width_index in range(-num_cells_line_width, num_cells_line_width):
                y = (rho + rho_width_index - j * np.cos(theta)) / np.sin(theta)
                i = math.floor(y)
                cell = self.get_cell_at_location(i, j)
                if cell is not None:
                    cell.set_visited(True)
        
        for i in range(self.grid_height):
            #Check cells up to num_cells_line_width around the line
            for rho_width_index in range(-num_cells_line_width, num_cells_line_width):
                x = (rho + rho_width_index - i * np.sin(theta)) / np.cos(theta)
                j = math.floor(x)
                cell = self.get_cell_at_location(i, j)
                if cell is not None:
                    cell.set_visited(True)

    """Clears the 'visited' flag for each of the cells in the grid."""
    def clear_visited(self):
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                self.get_cell_at_location(i,j).set_visited(False)

    """Retuns an array containing the number of points for each of the cells. Has the same dimension as the grid."""
    def get_counts(self):
        dim = self.get_dimensions()
        counts = np.zeros(dim)
        for i in range(dim[0]):
            for j in range(dim[1]):
                count = self.get_count(j, i)
                counts[i, j] = count        
        return counts

    """Retuns an array containing the number of points for each of the cells, as an image normalized to [0, 255]. Has the same dimension as the grid."""
    def get_counts_image(self, min_wall_points=200.0):
        counts = self.get_counts()
        max_points = max(np.amax(counts), min_wall_points)

        counts = counts / max_points

        counts = np.array(counts * 255, dtype = np.uint8)
        # counts = np.log(1 + counts)
        return counts

    """Retuns the counts_image above, after being log transformed. Used for visualization purposes."""
    def log_transform_counts_image(self):
        counts = self.get_counts()
        counts = (counts / np.amax(counts))
        counts = np.array(counts * 255, dtype = np.uint8) + 1.0
        counts = np.log(counts)
        return counts

    """Retuns a thresholded version of the counts_image."""
    def get_thresholded_counts(self, thresh=10):
        counts = self.get_counts()

        counts = counts / np.amax(counts)

        counts = np.array(counts * 255, dtype = np.uint8)

        below_threshold_indices = counts < thresh
        counts[below_threshold_indices] = 0

        above_threshold_indices = counts >= thresh
        counts[above_threshold_indices] = 255
        return counts

    """Applies the hough transform to the thresholded counts array."""
    def hough_from_grid(self, thresh=10, bThresh=True):
        if bThresh:
            counts = self.get_thresholded_counts(thresh)
        else:
            counts = self.get_counts()
        return hm.hough_transform(counts)

    """Retuns the points corresponding to a set of lines."""
    def get_line_points(self, lines):
        points_list = []
        for i in range(len(lines)):
            # angle, dist = lines[i].get_params()
            points = self.get_points_from_line(lines[i])
            self.set_line_cells_visited(lines[i])
            points_list.append(points)
        self.clear_visited()
        return points_list

    """Returns the index for the line with the highest number of corresponding points, helper method for get_best_line_ordering."""
    def get_best_line_index(self, lines, line_points, already_visited=[]):
        best_index = 0
        max_length = -1
        for i in range(len(lines)):
            # angle, dist = lines[i].get_params()
            if i in already_visited:
                continue

            points = line_points[i]
            if len(points) > max_length:
                best_index = i
                max_length = len(points)

        return best_index

    """Sorts the lines based on the number of corresponding points they have, from highest to lowest. Brute-force approach."""
    def get_best_line_ordering(self, lines):
        line_order = []
        line_points = []
        visited = []
        preliminary_point_list = self.get_line_points(lines)

        for _ in range(len(lines)):
            index = self.get_best_line_index(lines, preliminary_point_list, already_visited=visited)
            points = preliminary_point_list[index]
            line_order.append(lines[index])
            line_points.append(points)
            visited.append(index)
        return line_order, line_points

    """Filters the lines, so two very similar lines are not both added."""
    def filter_lines(self, lines, points):
        dim = self.get_dimensions()
        filtered_lines = lines.copy()
        filtered_points = points.copy()
        for i in reversed(range(len(filtered_lines))):
            bIntersect = False
            check_y = False
            max_val = dim[1]
            angle, dist = filtered_lines[i].get_hough_params()
            
            if abs(np.rad2deg(angle)) < 45.0:
                check_y = True
                max_val = dim[0]

            for j in range(len(filtered_lines)):
                if i == j:
                    continue
                other_line = filtered_lines[j]
                other_angle, other_dist = other_line.get_hough_params()

                if abs(np.rad2deg(angle) - np.rad2deg(other_angle)) < 25.0:
                    x, y = filtered_lines[i].intersect_lines(other_line)
                    if (check_y):
                        check_val = y
                    else:
                        check_val = x
                        
                    if check_val >= 0 and check_val <= max_val:
                        bIntersect = True
                        break
            if bIntersect:
                filtered_lines.pop(i)
                filtered_points.pop(i)
            else:
                break

        return filtered_lines, np.array(filtered_points)

    """Transforms a line in world space to grid space. Non-destructive."""
    def transform_line_to_grid_space(self, line):
        start_point = line.get_start()
        end_point = line.get_end()
        transformed_line = Line()
        transformed_line.set_points(self.interp_cell_location_from_point(start_point), self.interp_cell_location_from_point(end_point))
        return transformed_line

    """Transforms a line from grid space to world space. Non-destructive."""
    def transform_line_to_world_space(self, line):
        start_point = line.get_start()
        end_point = line.get_end()
        transformed_line = Line()
        transformed_line.set_points(self.convert_point_to_world_space(start_point), self.convert_point_to_world_space(end_point))
        return transformed_line