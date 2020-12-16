import numpy as np
from line_extraction_primitives.extraction_helper_methods import get_heights


#Definition for a Grid cell
class Cell:
    def __init__(self, i, j):
        self.points = []
        self.index = [i, j]
        self.visited = False
        self.polygon = None
        self.average_height = 0.0
        self.num_points = 0
        self.average_point = None
        self.median_height = 0.0
        self.percentile_95 = 0.0

    #Adds a point to the end of the points list, and increments the number of points
    def add_point(self, point):
        self.points.append(point)
        self.num_points = self.num_points + 1

    def pop_point(self, index):
        point = self.points.pop(index)
        return point

    #Gets the average point in the cell, averaged over all axes
    def get_comp_point(self):
        average_point = np.array([0.0, 0.0, 0.0])
        if self.average_point is None and self.num_points > 0:
            points = np.array(self.points)
            average_point = np.sum(points, axis=0) / self.num_points
            self.average_point = average_point
        return self.average_point

    #Returns the total number of points in the cell
    def get_count(self):
        return len(self.points)

    #Returns the average height of the points in the cell, also stores the median and 95 percentiles for future use
    def set_average_height(self):
        points = np.array(self.points)
        if points.shape[0] == 0:
            self.average_height = 0.0
            return

        allz = (np.delete(points, [0, 1], 1)).flatten()
        if allz.shape[0] == 0:
            self.average_height = 0
            return
        avg_height = np.sum(allz) / allz.shape[0]

        self.median_height = np.median(allz)
        if len(allz) > 1:
            self.percentile_95 = np.percentile(allz, 95)
        else:
            self.percentile_95 = avg_height
        self.average_height = avg_height

    def get_points(self):
        return self.points

    def get_num_points(self):
        return self.num_points

    def delete_points(self):
        self.points = []

    #Gets the difference between the maximum and minimum heights in the cell
    def get_height_diff(self):
        if len(self.points) == 0 or len(self.points[0]) < 2:
            return 0.0
        else:
            min_x, min_y, min_z = np.amin(self.points, axis=0)
            max_x, max_y, max_z = np.amax(self.points, axis=0)

            return max_z - min_z

    #Projects points (i.e. removes Z-axis)
    def get_projected_points(self):
        if not self.points:
            return []
        return np.delete(self.points, 2, 1).tolist()

    #Assigns boolean visited value, used when traversing the cells
    def set_visited(self, visited):
        self.visited = visited

    def is_visited(self):
        return self.visited