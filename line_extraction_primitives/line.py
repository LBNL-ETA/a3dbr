import line_extraction_primitives.extraction_helper_methods as hm
import numpy as np
import math 

#Saves a set of lines to a file
def save_lines(lines, filename="lines.txt"):
    with open(filename, 'w') as f:
        for line in lines:
            string = str(line.get_start()[0]) + "," + str(line.get_start()[1]) + " " + str(line.get_end()[0]) + "," + str(line.get_end()[1]) + "\n"
            f.write(string)
        f.close()

#Loads a set of lines from a file
def load_lines(filename="lines.txt"):
    lines = []
    with open(filename, 'r') as f:
        counter = 0
        for next_line in f.readlines():
            start, end = next_line.split()
            line = Line()
            start_split = start.split(',')
            end_split = end.split(',')
            start_point = np.array([float(start_split[0]), float(start_split[1])])
            end_point = np.array([float(end_split[0]), float(end_split[1])])

            line.set_points(start_point, end_point)
            line.set_id(counter)
            counter += 1
            lines.append(line)
        f.close()
    return lines
    
#Definition for a Line, which can be used either with a start and end point or as a rho and theta parameter (hough parameters)
class Line:
    def __init__(self):
        self.theta = None
        self.rho = None
        self.start = None
        self.end = None
        self.id = 0

        """Whether the start or endpoints can still be moved, i.e. when bringing it to a corner. Should be false for discovered corners or
        points that lie on the intersection of two line segments"""
        self.is_end_movable = True
        self.is_start_movable = True

    def set_hough_params(self, theta, rho):
        self.rho = rho
        self.theta = theta

    def is_vertical(self):
        angle = np.rad2deg(self.theta)
        if abs(angle) < 45.0:
            return True
        else:
            return False

    def set_id(self, line_id):
        self.id = line_id

    def set_start(self, start):
        self.start = start

    def set_end(self, end):
        self.end = end

    """Booleans that determine if the endpoints are fixed or can be moved. Used for segment post-processing logic, to make sure that if a line segment has split
    another line segment, corners that are close together remain separate (i.e. are not merged/brought together)"""

    def set_start_movable(self, movable):
        self.is_start_movable = movable

    def set_end_movable(self, movable):
        self.is_end_movable = movable

    def get_start_movable(self):
        return self.is_start_movable

    def get_end_movable(self):
        return self.is_end_movable

    #Returns the minimum distance between the endpoints of a line and another point
    def get_distance_to_point(self, point):
        start_dist = hm.distance(self.get_start(), point)
        end_dist = hm.distance(self.get_end(), point)

        return min(start_dist, end_dist)

    #Sets the endpoints of a line, and enforces a consistent ordering
    def set_points(self, point1, point2):
        if point1[1] > point2[1]:
            self.start = point2
            self.end = point1
        else:
            self.start = point1
            self.end = point2

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def get_hough_params(self):
        return self.theta, self.rho

    #Returns the direction vector of a line, and the start point
    def get_line_equation(self):
        start_point = np.array(self.get_start())
        end_point = np.array(self.get_end())
        line_vector = start_point - end_point
        return start_point, line_vector

    #Projects a set of points onto the line
    def project_points_onto_line(self, points):
        projected_points = []
        start_point, s = self.get_line_equation()
        for each in points:
            coords = np.array(each)
            v = coords - start_point
            proj = start_point + ((np.dot(s, v) / np.dot(s, s)) * s)
            projected_points.append(proj)
        return projected_points

    #Determines the rotation angle between the vertical axis and the line
    def find_rotation_angle(self):
        start_point = self.get_start()
        end_point = self.get_end()
        v1 = np.linalg.norm(end_point - start_point)
        v2 = np.linalg.norm(np.array([start_point[0], end_point[1]]) - start_point)
        if v1 == 0.0:
            return 0.0
        else:
            theta = np.arccos(v2 / v1)

        if start_point[0] > end_point[0]:
            theta = -1.0 * theta
        return theta

    #Finds the intersection point between this line and the other, if it exists, and otherwise returns infinity (float)
    def intersect_lines(self, other):
        theta1, rho1 = self.get_hough_params()
        theta2, rho2 = other.get_hough_params()
        
        y_denom = (np.cos(theta2)* np.sin(theta1) - np.cos(theta1)*np.sin(theta2))
        x_denom = (np.sin(theta2)* np.cos(theta1) - np.sin(theta1)*np.cos(theta2))
        if y_denom == 0 or x_denom ==0:
            return float("inf"), float("inf")
        else:
            y = (rho1*np.cos(theta2) - rho2*np.cos(theta1)) / y_denom
            x = (rho1*np.sin(theta2) - rho2*np.sin(theta1)) / x_denom
            return x, y

    """Enumerates the cases for how two lines intersect at a corner. I.e. whether the two start points line up, 
    the two end points, not at all, or one start point with one end point"""
    def get_corner_case(self, other_line, width=2.0):
        start_point = np.array(self.get_start())
        end_point = np.array(self.get_end())

        other_start = np.array(other_line.get_start())
        other_end = np.array(other_line.get_end())

        if np.linalg.norm(start_point - other_start) < width and self.get_start_movable() and other_line.get_start_movable():
            return 1
        elif np.linalg.norm(start_point - other_end) < width and self.get_start_movable() and other_line.get_end_movable():
            return 2
        elif np.linalg.norm(end_point - other_end) < width and self.get_end_movable() and other_line.get_end_movable():
            return 3
        elif np.linalg.norm(end_point - other_start) < width and self.get_end_movable() and other_line.get_start_movable():
            return 4
        else:
            return 0

    """Determines if a point lies on the line, within an allowable distance epsilon"""
    def on_line(self, point, epsilon=0.1):
        start_point = np.array(self.get_start())
        end_point = np.array(self.get_end())

        line_vec = end_point - start_point
        orig = point - start_point
        if line_vec[0] == 0:
            return False
        elif line_vec[0] < 0:
            line_vec = start_point - end_point
            orig = point - end_point
            start_point, end_point = end_point, start_point

        check_point = start_point + line_vec * orig[0] / line_vec[0]
        if hm.distance(check_point, point) < epsilon and orig[0] / line_vec[0] <= 1.0 and orig[0] / line_vec[0] > 0:
            # print(start_point, end_point, point)
            return True
        return False

    """Function that is used to split the lines, and attract separate line endpoints together. 
    This is the main method for the line segment post processing."""
    def check_line_proximity(self, other_line, percent_proximity=0.2, epsilon_proximity=0.1):
        intersection_point = self.get_euclidean_intersection_point(other_line)
        start_point = np.array(self.get_start())
        end_point = np.array(self.get_end())

        if intersection_point is None:
            return []

        new_lines = []

        start_distance = hm.distance(intersection_point, start_point)
        end_distance = hm.distance(intersection_point, end_point)

        #Attracts line endpoints together, with the distance increasing proportional to the length of the line segment
        attr_proximity = max(epsilon_proximity * self.get_length(), 2.0)
        b_attracted = self.attract_line_endpoints(other_line, dist=attr_proximity)

        #Distance for which to extend lines, if doing so would cause it to intersect another line
        snip_proximity = max(percent_proximity * self.get_length(), 3.0)
        
        #Only extend the line if it's endpoints did not get 'attracted' to another line's endpoints, and the intersection point lies on the other line
        if not b_attracted and other_line.on_line(intersection_point):
            #If the intersection point lies on the current line segment as well, split both the line segments at the intersection point, to create 2 new line segments 
            if self.on_line(intersection_point):

                
                line1 = self.snip_line(intersection_point)
                line2 = other_line.snip_line(intersection_point)

                if line1:
                    new_lines.append(line1)
                if line2:
                    new_lines.append(line2)

            #If the distance of either endpoint to the intersection point is less than the snip_proximity, split the other edge into two, and extend the current line segment
            #to the endpoint
            elif (start_distance < snip_proximity or end_distance < snip_proximity):
                
                line = other_line.snip_line(intersection_point)
                if line:
                    new_lines.append(line)

                self.extend_line(intersection_point)
            
        return new_lines

    """Splits the line into two pieces. Takes as input a point along the line, which is used as the splitting point. 
    Modifies the input line, and returns a new line."""
    def snip_line(self, intersection_point):
        start_point = np.array(self.get_start())
        end_point = np.array(self.get_end())
        start_distance = hm.distance(intersection_point, start_point)
        end_distance = hm.distance(intersection_point, end_point)
        ret_val = None
        if start_distance > 0.05 and end_distance > 0.05:
            line_length = self.get_length()
            new_line = Line()
            new_line.set_points(intersection_point, self.get_end())
            new_line.set_start_movable(False)
            if new_line.get_length() > 0.2:
                ret_val = new_line
                self.set_points(self.get_start(), intersection_point)
                self.set_end_movable(False)
        return ret_val

    """Extends the line so that one of the endpoints is the intersection point of another line 
    """
    def extend_line(self, intersection_point):
        start_point = np.array(self.get_start())
        end_point = np.array(self.get_end())
        start_distance = hm.distance(intersection_point, start_point)
        end_distance = hm.distance(intersection_point, end_point)

        #Chooses the start point that results in a longer line
        if start_distance < end_distance and self.get_start_movable():
            self.set_points(intersection_point, end_point)
            self.set_start_movable(False)
        elif self.get_end_movable() and end_distance < start_distance:
            self.set_points(start_point, intersection_point)
            self.set_end_movable(False)

    """Finds the intersection point of the line with another line, if it exists"""
    def get_euclidean_intersection_point(self, other_line):

        start_point = np.array(self.get_start())
        end_point = np.array(self.get_end())

        other_start = np.array(other_line.get_start())
        other_end = np.array(other_line.get_end())

        v1 = hm.normalize(start_point - end_point)
        v2 = hm.normalize(other_start - other_end)

        if v2[1] == 0.0 or (v1[0] - v1[1] * v2[0] / v2[1]) == 0.0:
            return
        denom_a = (v1[0] - v1[1] * v2[0] / v2[1])
        a = ((start_point[1] - other_start[1]) * v2[0] / v2[1] + (other_start[0] - start_point[0])) / denom_a

        intersection_point = start_point + a * v1
        return intersection_point

    """Attracts line endpoints together if they are within a specified distance (dist), by moving the endpoint of another line to the closest endpoint on this line."""
    def attract_to_line(self, other_line, dist):
        corner_case = self.get_corner_case(other_line, dist)

        #If the corners do not line up at all, do nothing
        if corner_case == 0:
            return

        start_point = np.array(self.get_start())
        end_point = np.array(self.get_end())

        other_start = np.array(other_line.get_start())
        other_end = np.array(other_line.get_end())

        v1 = hm.normalize(start_point - end_point)
        v2 = hm.normalize(other_start - other_end)

        intersection_point = self.get_euclidean_intersection_point(other_line)
        if intersection_point is None:
            return False

        if corner_case == 1 and (hm.distance(start_point, intersection_point) < dist
            or hm.distance(other_start, intersection_point) < dist):

            self.set_start(intersection_point)
            other_line.set_start(intersection_point)

        elif corner_case == 2 and (hm.distance(start_point, intersection_point) < dist
            or hm.distance(other_end, intersection_point) < dist):

            self.set_start(intersection_point)
            other_line.set_end(intersection_point)

        elif corner_case == 3 and (hm.distance(end_point, intersection_point) < dist
            or hm.distance(other_end, intersection_point) < dist):

            self.set_end(intersection_point)
            other_line.set_end(intersection_point)

        elif corner_case == 4 and (hm.distance(end_point, intersection_point) < dist
            or hm.distance(other_start, intersection_point) < dist):

            self.set_end(intersection_point)
            other_line.set_start(intersection_point)

        else:
            return False
        return True

    """ Attraction works as follows: All lines with endpoints within a search radius (e.g. 5 meters) are chosen as potential corner candidates. 
        The intersection between the current line and the potential candidate is found. Then the intersection point is checked 
        to make sure that the intersection point is very close to at least one of the line segments (e.g. within 1 meter).
    """
    def attract_line_endpoints2(self, other_lines, dist=1.0):

        min_dist_start = float("inf")
        min_dist_end = float("inf")
        closest_start = None
        closest_end = None

        for line in other_lines:
            if line == self:
                continue
            start_dist = line.get_distance_to_point(self.get_start())
            end_dist = line.get_distance_to_point(self.get_end())

            if start_dist < min_dist_start:
                closest_start = line
                min_dist_start = start_dist
            if end_dist < min_dist_end:
                closest_end = line
                min_dist_end = end_dist

        has_attracted = False
        if min_dist_start < dist:
            start_attracted = self.attract_to_line(closest_start, dist)
            has_attracted = has_attracted or start_attracted
        if min_dist_end < dist:
            end_attracted = self.attract_to_line(closest_end, dist)
            has_attracted = has_attracted or end_attracted

        return has_attracted

    """ Attraction works as follows: All lines with endpoints within a search radius (e.g. 5 meters) are chosen as potential corner candidates. 
        The intersection between the current line and the potential candidate is found. Then the intersection point is checked 
        to make sure that the intersection point is very close to at least one of the line segments (e.g. within 1 meter).
    """
    def attract_line_endpoints(self, other_line, dist=1.0):
        corner_case = self.get_corner_case(other_line, dist)

        if corner_case == 0:
            return

        start_point = np.array(self.get_start())
        end_point = np.array(self.get_end())

        other_start = np.array(other_line.get_start())
        other_end = np.array(other_line.get_end())

        v1 = hm.normalize(start_point - end_point)
        v2 = hm.normalize(other_start - other_end)

        if np.rad2deg(hm.angle_between(v1, v2)) < 25.0:
            return False

        intersection_point = self.get_euclidean_intersection_point(other_line)
        if intersection_point is None:
            return False

        if corner_case == 1 and (hm.distance(start_point, intersection_point) < dist
            or hm.distance(other_start, intersection_point) < dist):

            self.set_start(intersection_point)
            other_line.set_start(intersection_point)
            self.set_start_movable(False)
            other_line.set_start_movable(False)
           

        elif corner_case == 2 and (hm.distance(start_point, intersection_point) < dist
            or hm.distance(other_end, intersection_point) < dist):

            self.set_start(intersection_point)
            other_line.set_end(intersection_point)
            self.set_start_movable(False)
            other_line.set_end_movable(False)
           

        elif corner_case == 3 and (hm.distance(end_point, intersection_point) < dist
            or hm.distance(other_end, intersection_point) < dist):

            self.set_end(intersection_point)
            other_line.set_end(intersection_point)
            self.set_end_movable(False)
            other_line.set_end_movable(False)
           
        elif corner_case == 4 and (hm.distance(end_point, intersection_point) < dist
            or hm.distance(other_start, intersection_point) < dist):

            self.set_end(intersection_point)
            other_line.set_start(intersection_point)
            self.set_end_movable(False)
            other_line.set_start_movable(False)
        else:
            return False
        return True

    """Gets the x coordinate of the point on the line at a specified y coordinate"""
    def get_x_on_line(self, y):
        theta, rho = self.get_hough_params()
        x = (rho - y * np.sin(theta)) / np.cos(theta)
        return x

    """Gets the y coordinate of the point on the line at a specified x coordinate"""
    def get_y_on_line(self, x):
        theta, rho = self.get_hough_params()
        y = (rho - x * np.cos(theta)) / np.sin(theta)
        return y

    """Extracts line segments along the line, requires line and points to be rotated to the vertical axis and median filtered"""
    def extract_segments(self, image, pixel_gap=5, wall_length=5):
        start = self.get_start()
        end = self.get_end()

        x = start[0]
        j = 0
        y_start = float("inf")
        y_end = 0

        iteration_start = max(math.floor(start[1]), 0)
        iteration_end = min(math.ceil(end[1]), image.shape[0])

        current_gap = 0
        lines = []

        for i in range(iteration_start, iteration_end):
            if j >= 0 and j < image.shape[1] and image[i][j] > 1.0:
                current_gap = 0
                if i < y_start:
                    y_start = i
                y_end = i
            elif y_start <= y_end:
                current_gap += 1
                if current_gap > pixel_gap:
                    if y_end > y_start + wall_length:
                        line = Line()
                        line.set_points([x, y_start], [x, y_end])
                        lines.append(line)
                    current_gap = 0
                    y_start = float("inf")
                    y_end = 0   

        if y_end > y_start + wall_length:
            line = Line()
            line.set_points([x, y_start], [x, y_end])
            lines.append(line)

        return lines

    """Gets the perpendicular distance of a point to the line"""
    def get_point_proximity(self, point, range_x=[0, 100], range_y=[0, 100], eps=0.0001):
        theta, rho = self.get_hough_params()
        if theta and rho:
            x_min = range_x[0]
            x_max = range_x[1]
            y_min = range_y[0]
            y_max = range_y[1]

            if x_max - x_min < y_max - y_min:
                x_min = (rho - range_y[0] * np.sin(theta)) / np.cos(theta)
                x_max = (rho - range_y[1] * np.sin(theta)) / np.cos(theta)
            else:
                y_min = (rho - range_x[0] * np.cos(theta)) / np.sin(theta)
                y_max = (rho - range_x[1] * np.cos(theta)) / np.sin(theta)

            point2 = np.array([x_min, y_min])
            point3 = np.array([x_max, y_max])

            d = np.linalg.norm(np.cross(point2-point, point2-point3))/(np.linalg.norm(point2-point3) + eps)
            return d
        else:
            point2 = np.array(self.get_start())
            point3 = np.array(self.get_end())

            d = np.linalg.norm(np.cross(point2-point, point2-point3))/(np.linalg.norm(point2-point3) + eps)
            return d

    """Generates a line from a set of points using RANSAC"""
    def line_from_ransac(self, points):
        proj_points = np.delete(points, 2, 1)
        point1, point2 = hm.apply_ransac(proj_points)
        self.set_points(point1, point2)

    """Rotates a line around a rotation point, by a specified angle theta. Non-destructive."""
    def rotate_line(self, theta, rotation_point):
        start_point = self.get_start()
        end_point = self.get_end()

        rotated_start_point, rotated_end_point = hm.rotate_about_point([start_point, end_point], theta, rotation_point)
        rotated_line = Line()
        rotated_line.set_points(rotated_start_point, rotated_end_point)
        return rotated_line

    """Get the length of a line, treated as a line segment"""
    def get_length(self):
        return hm.distance(self.get_start(), self.get_end())