import polygon_primitives.helper_methods as hm
import numpy as np
from line_extraction_primitives.line import Line
import plotting

"""Definition for the edge class. Extends the Line class from line_extraction_primitives."""
class Edge(Line):
    def __init__(self, point1, point2, edge_id=-1, temp_edge=False, order_points=True):
        self.end_points = [point1, point2]
        self.start_neighbors = []
        self.end_neighbors = []
        self.polygon_edge_candidate = True
        self.edge_id = edge_id
        self.left_poly = None
        self.right_poly = None
        self.potential_start = None
        self.potential_end = None
        self.temp_edge = temp_edge
        self.set_points(point1, point2, order_points)


    def get_start(self):
        return self.start_point

    def get_end(self):
        return self.end_point

    def set_edge_id(self, edge_id):
        self.edge_id = edge_id
        self.temp_edge = False

    def set_start(self, point):
        self.start_point = point

    def set_end(self, point):
        self.end_point = point

    def get_edge_vector(self):
        return self.end_point - self.start_point

    """Determines the density of the edge in the grid, by finding the points near the edge and projecting them onto the line."""
    def get_support(self, grid):
        edge = Edge(self.get_start(), self.get_end())
        line.set_points(self.get_start(), self.get_end())
        transformed_line = grid.transform_line_to_grid_space(line)
        points = grid.get_points_from_line_segment(transformed_line, width=0.4)
        density = hm.get_density([points], width=0.4)
        return density

    """Reverse the direction of an edge. This switches the start and end neighbors as well."""
    def reverse_direction(self):
        start_point = self.get_end()
        end_point = self.get_start()
        new_start_neighbors = self.get_end_neighbors()
        new_end_neighbors = self.get_start_neighbors()

        self.set_start(start_point)
        self.set_end(end_point)
        self.start_neighbors = new_start_neighbors
        self.end_neighbors = new_end_neighbors

    """Put the edge points in an absolute ordering. Sorted on the y direction, so that the end point has a higher y."""
    def set_points_absolute_order(self):
        point1 = self.get_start()
        point2 = self.get_end()

        if point1[1] > point2[1]:
            self.reverse_direction()

    def set_points(self, point1, point2, order_points=False):
        self.set_start(point1)
        self.set_end(point2)
        if order_points and point1[1] > point2[1]:
            self.reverse_direction()
            
    """Resets all the edge components."""
    def reset_edge(self):
        self.start_neighbors = []
        self.end_neighbors = []
        self.polygon_edge_candidate = True
        self.left_poly = None
        self.right_poly = None
        self.temp_edge = False

    def get_neighbors(self):
        return self.start_neighbors + self.end_neighbors

    def remove_neighbor(self, neighbor):
        self.start_neighbors.remove(neighbor)
        self.end_neighbors.remove(neighbor)

    """Removes the current edge as a neighbor from the neighboring edges."""
    def delete(self):
        for each in self.start_neighbors:
            each.remove_neighbor(self)
        for each in self.end_neighbors:
            each.remove_neighbor(self)

    def get_start_neighbors(self):
        return self.start_neighbors

    def set_edge_candidate(self, b_candidate):
        self.polygon_edge_candidate = b_candidate

    def get_end_neighbors(self):
        return self.end_neighbors

    def is_polygon_candidate(self):
        return self.polygon_edge_candidate

    def is_neighbor(self, other_edge):
        return other_edge in self.start_neighbors or other_edge in self.end_neighbors

    def check_if_neighbors_in_list(self, edges):
        for edge in self.get_neighbors():
            if edge in edges:
                return True
        return False

    """Shortens the edge by the specified amount, from either the start or the end or both."""
    def shorten_edge_by_amount(self, amount, is_start, is_end):
        start = self.get_start()
        end = self.get_end()
        vec = hm.normalize(end - start)
        if is_start:
            start = start + amount * vec
        if is_end:
            end = end - amount * vec
        
        self.set_points(start, end)        

    """Intersects a ray with the current edge, and returns the intersection point if one exists."""
    def get_euclidean_intersection_with_ray(self, ray_start, ray_direction):
        start_point = np.array(self.get_start())
        end_point = np.array(self.get_end())

        v1 = hm.normalize(start_point - end_point)
        v2 = hm.normalize(ray_direction)

        return hm.get_intersection_point(v1, v2, start_point, ray_start)

    """Gets the intersection point of the current edge with another edge, if one exists."""
    def get_euclidean_intersection_point(self, other_edge):

        start_point = np.array(self.get_start())
        end_point = np.array(self.get_end())

        other_start = np.array(other_edge.get_start())
        other_end = np.array(other_edge.get_end())

        v1 = hm.normalize(start_point - end_point)
        v2 = hm.normalize(other_start - other_end)
        return hm.get_intersection_point(v1, v2, start_point, other_start)

    """Compares the current edge with another edge, and determines whether the endpoints are close enough to consider them neighbors."""
    def check_and_set_neighbor(self, edge, dist=0.5):
        if self == edge or edge in self.start_neighbors or edge in self.end_neighbors:
            return
        if hm.compare_points(edge.get_start(), self.get_start(), dist):
            self.start_neighbors.append(edge)
            edge.start_neighbors.append(self)

        elif hm.compare_points(edge.get_start(), self.get_end(), dist):
            self.end_neighbors.append(edge)
            edge.start_neighbors.append(self)

        elif hm.compare_points(edge.get_end(), self.get_start(), dist):
            self.start_neighbors.append(edge)
            edge.end_neighbors.append(self)

        elif hm.compare_points(edge.get_end(), self.get_end(), dist):
            self.end_neighbors.append(edge)
            edge.end_neighbors.append(self)

    """Walk along the edge, in the direction specified by following the incoming edge to the current edge."""
    def walk_along_edge(self, incoming_edge):
        if incoming_edge in self.start_neighbors:
            return self.end_neighbors
        else:
            return self.start_neighbors

    """Gets the endpoint shared by two edges, if one exists."""
    def get_shared_point(self, other_edge, dist=0.8):
        start_point = self.get_start()
        end_point = self.get_end()
        other_start = other_edge.get_start()
        other_end = other_edge.get_end()

        if hm.compare_points(start_point, other_start, dist) or hm.compare_points(start_point, other_end, dist):
            return start_point
        elif hm.compare_points(end_point, other_start, dist) or hm.compare_points(end_point, other_end, dist):
            return end_point
        else:
            return None

    """Checks if the current edge is left of the input edge."""
    def is_edge_left(self, edge):
        start_point = None
        matching_point = None
        end_point = None
        if hm.compare_points(edge.get_start(), self.get_start()):
            start_point = self.get_end()
            matching_point = self.get_start()
            end_point = edge.get_end()
        elif hm.compare_points(edge.get_end(), self.get_start()):
            start_point = self.get_end()
            matching_point = self.get_start()
            end_point = edge.get_start()
        elif hm.compare_points(edge.get_end(), self.get_end()):
            start_point = self.get_start()
            matching_point = self.get_end()
            end_point = edge.get_start()
        elif hm.compare_points(edge.get_start(), self.get_end()):
            start_point = self.get_start()
            matching_point = self.get_end()
            end_point = edge.get_end()
        if matching_point is None:
            return False

        return hm.is_point_left(start_point, matching_point, end_point)

    """Gets the angle between two neighboring edges. Checks for neighborhood by checking whether endpoints are within 'dist' of each other."""
    def get_angle_between_edges(self, edge, dist=0.5):
        #Find matching endpoints, note that the current edge (self) is always treated as the start
        start_point = None
        matching_point = None
        end_point = None

        if hm.compare_points(edge.get_start(), self.get_start(), dist):
            start_point = self.get_end()
            matching_point = self.get_start()
            end_point = edge.get_end()
        elif hm.compare_points(edge.get_end(), self.get_start(), dist):
            start_point = self.get_end()
            matching_point = self.get_start()
            end_point = edge.get_start()
        elif hm.compare_points(edge.get_end(), self.get_end(), dist):
            start_point = self.get_start()
            matching_point = self.get_end()
            end_point = edge.get_start()
        elif hm.compare_points(edge.get_start(), self.get_end(), dist):
            start_point = self.get_start()
            matching_point = self.get_end()
            end_point = edge.get_end()

        if matching_point is None:
            return float("inf")

        v1 = np.array(matching_point) - np.array(start_point)
        v2 = np.array(end_point) - np.array(matching_point)
        theta = hm.signed_angle_between(v1, v2)
        if theta > 140:
            theta = theta - 180.0
        elif theta < -140:
            theta = theta + 180.0

        return theta

    """Gets the angle between two neighboring edges."""
    def get_angle(self, edge):
        #Find matching endpoints, note that the current edge (self) is always treated as the start

        shared_point = self.get_shared_point(edge)
        if shared_point is not None:
            if hm.compare_points(shared_point, self.get_start()):
                self_start = self.get_start()
                self_end = self.get_end()
            else:
                self_start = self.get_end()
                self_end = self.get_start()

            if hm.compare_points(shared_point, edge.get_start()):
                other_start = edge.get_start()
                other_end = edge.get_end()
            else:
                other_start = edge.get_end()
                other_end = edge.get_start()

            v1 = np.array(other_end) - np.array(other_start)
            v2 = np.array(self_end) - np.array(self_start)

            theta = hm.angle_between(v1, v2)
            return np.rad2deg(theta)
        else:
            return float("inf")

    """Finds the normal vector to the edge"""
    def find_normal(self):
        v1 = np.array(self.get_start()) - np.array(self.get_end())
        normal_dir = [-1.0 * v1[1], v1[0]]
        return hm.normalize(normal_dir)

    """Returns whether the point is below the edge, above the edge, or on the edge"""
    def check_point_below_edge(self, point):
        normal = self.find_normal()
        signed_dist = np.dot(normal, (np.array(point) - np.array(self.get_start())))
        if signed_dist > 0:
            return False
        else:
            return True

    """Returns whether a vector with starting point comp_point intersects the edge"""
    def check_edge_intersection(self, comp_point, comp_vector, epsilon=0.01):
        start_point = np.array(self.get_start())
        end_point = np.array(self.get_end())

        edge_vec = end_point - start_point

        if comp_vector[1] == 0.0 or (edge_vec[0] - edge_vec[1] * comp_vector[0] / comp_vector[1]) == 0.0:
            return
        denom_a = (edge_vec[0] - edge_vec[1] * comp_vector[0] / comp_vector[1])
        a = ((start_point[1] - comp_point[1]) * comp_vector[0] / comp_vector[1] + (comp_point[0] - start_point[0])) / denom_a
        b = (start_point[0] - comp_point[0] + a * edge_vec[0]) / comp_vector[0]
        if a >= -epsilon and a <= 1.0 + epsilon and b > -epsilon:
            return True
        else:
            return False

    def get_length(self):
        return hm.distance(self.get_start(), self.get_end())

    """Returns whether the current edge is collinear with another edge."""
    def is_collinear(self, other_edge):
        start = self.get_start()
        end = self.get_end()
        other_start = other_edge.get_start()
        other_end = other_edge.get_end()

        return hm.collinear(start, end, other_start) and hm.collinear(start, end, other_end)

    """Returns the intersection point that would occur by extending one of the current edge or the input edge, if it exists."""
    def get_extended_intersection(self, other_edge):
        start = self.get_start()
        end = self.get_end()
        other_start = other_edge.get_start()
        other_end = other_edge.get_end()
        edge = None 

        if hm.collinear(start, end, other_start) and hm.collinear(start, end, other_end):
            dist1 = hm.distance(end, other_end)
            dist2 = hm.distance(start, other_end)
            dist3 = hm.distance(end, other_start)
            dist4 = hm.distance(start, other_start)
            min_dist = min(dist1, dist2, dist3, dist4)

            if min_dist == dist1:
                edge_start, edge_end = end, other_end
            elif min_dist == dist2:
                edge_start, edge_end = start, other_end
            elif min_dist == dist3:
                edge_start, edge_end = end, other_start
            elif min_dist == dist4:
                edge_start, edge_end = start, other_start
            edge = Edge(edge_start, edge_end)
        elif np.rad2deg(hm.angle_between(end - start, other_end - other_start)) > 10.0:
            intersection_point = self.get_euclidean_intersection_point(other_edge)
            if intersection_point is None:
                return None
            dist1 = hm.distance(end, intersection_point)
            dist2 = hm.distance(start, intersection_point)
            min_dist = min(dist1, dist2)

            if min_dist == dist1:
                edge_start, edge_end = end, intersection_point
            elif min_dist == dist2:
                edge_start, edge_end = start, intersection_point
            edge = Edge(edge_start, edge_end)

        return edge

    """If an endpoint is located at shared_point, moves it to move_point."""
    def move_to_point(self, shared_point, move_point):
        start = self.get_start()
        end = self.get_end()
        if hm.compare_points(start, shared_point):
            self.set_start(move_point)
        else:
            self.set_end(move_point)

    """Checks whether two edges have endpoints that are close together, and if so, moves them to the exact same location."""
    def move_edges_to_shared_point(self, other_edge, epsilon):
        shared_point = self.get_shared_point(other_edge, epsilon)
        has_moved = True

        if shared_point is not None:    
            start = self.get_start()
            end = self.get_end()
            other_start = other_edge.get_start()
            other_end = other_edge.get_end()

            angle = self.get_angle(other_edge)
            if abs(angle) < 10 or abs(angle - 180.0) < 10:
                if hm.compare_points(shared_point, start, epsilon) and self.get_length() < other_edge.get_length():
                    other_edge.move_to_point(shared_point, start)
                    
                elif hm.compare_points(shared_point, end, epsilon) and self.get_length() < other_edge.get_length():
                    other_edge.move_to_point(shared_point, end)
                elif hm.compare_points(shared_point, other_start, epsilon) and other_edge.get_length() < self.get_length():
                    self.move_to_point(shared_point, other_start)
                elif hm.compare_points(shared_point, other_end, epsilon) and other_edge.get_length() < self.get_length():
                    self.move_to_point(shared_point, other_end)
                else:
                    
                    has_moved = False
            else:
                has_moved = False

        return has_moved

    """If one edge engulfs another edge, this removes the overlap between them by moving one endpoint of the larger edge to the start of the
    smaller edge, and constructs a third edge connecting to the end of the smaller edge."""
    def remove_edge_overlap(self, other_edge):
        start = self.get_start()
        end = self.get_end()
        other_start = other_edge.get_start()
        other_end = other_edge.get_end()

        shared_point = self.get_shared_point(other_edge)

        start_dist = hm.distance(start, other_start)
        end_dist = hm.distance(end, other_start)

        if hm.compare_points(shared_point, other_start):
            if hm.compare_points(shared_point, start):
                other_edge.set_points(end, other_end)
            else:
                other_edge.set_points(start, other_end)
        else:
            if hm.compare_points(shared_point, start):
                other_edge.set_points(other_start, end)
            else:
                other_edge.set_points(other_start, start)

    """Removes edge overlap by moving edge endpoints. Can also create a third edge if one edge engulfs the other."""
    def split_edge_overlap(self, other_edge, percent_proximity=0.05):
        start = self.get_start()
        end = self.get_end()
        other_start = other_edge.get_start()
        other_end = other_edge.get_end()

        new_edge = None

        on_line_dist = percent_proximity * max(other_edge.get_length(), self.get_length())
        

        shared_point = self.get_shared_point(other_edge)
        start_point_on_other = other_edge.on_line(start, dist=on_line_dist)
        end_point_on_other = other_edge.on_line(end, dist=on_line_dist)
        other_start_on_self = self.on_line(other_start, dist=on_line_dist)
        other_end_on_self = self.on_line(other_end, dist=on_line_dist)

        #Case 1: One edge engulfs the other
        if shared_point is not None and ((start_point_on_other and hm.compare_points(shared_point, end)) or 
            (end_point_on_other and hm.compare_points(shared_point, start))):
            self.remove_edge_overlap(other_edge)

        elif shared_point is not None and ((other_start_on_self and hm.compare_points(shared_point, other_end)) or 
            (other_end_on_self and hm.compare_points(shared_point, other_start))):
            other_edge.remove_edge_overlap(self)

        #Case 2: The two edges overlap
        elif (start_point_on_other or end_point_on_other) and self.is_collinear(other_edge):
            if start_point_on_other:
                new_edge_end = start
            else:
                new_edge_end = end

            if other_start_on_self:
                new_edge = Edge(other_start, new_edge_end)
                self.move_to_point(new_edge_end, other_start)
                other_edge.move_to_point(other_start, new_edge_end)
            else:
                new_edge = Edge(other_end, new_edge_end)
                self.move_to_point(new_edge_end, other_end)
                other_edge.move_to_point(other_end, new_edge_end)
        return new_edge

    """Checks whether an edge intersects any of the edges (within a small proximity), and if so, splits both edges at the intersection point."""
    def split_edges(self, edges, dist=0.5, percent_proximity=0.05):
        new_edges = []
        for edge in edges:

            if edge == self:
                continue
         
            moved_shared = self.move_edges_to_shared_point(edge, dist)
            
            new_edge = self.split_edge_overlap(edge, percent_proximity=percent_proximity)
            if new_edge is not None:
                new_edges.append(new_edge)
            
        return new_edges

    """Checks whether a point lies on the edge."""
    def on_line(self, point, dist=0.1, epsilon=0.1, debug=False):
        start_point = np.array(self.get_start())
        end_point = np.array(self.get_end())

        v1 = end_point - start_point
        orig = point - start_point


        #Chooses the shorter line segment between the test point and the two endpoints
        checkpoint_start = start_point
        if hm.distance(point, start_point) > hm.distance(point, end_point):
            orig = point - end_point
            checkpoint_start = end_point
            v1 = start_point - end_point

        index = 0
        if abs(v1[0]) < abs(v1[1]):
            index = 1

        #TODO (Maybe): choose the starting point that minimizes checkpoint distance?
        checkpoint = checkpoint_start + v1 * orig[index] / v1[index]

        if hm.distance(checkpoint, point) < dist and orig[index] / v1[index] <= 1.0 + epsilon and orig[index] / v1[index] > 0:
            # print(start_point, end_point, point)
            return True
        return False

    def check_new_edges(self, edges, dist=1.0):
        for edge in edges:
            if self.equals(edge, dist=dist):
                #Ensures edge directions match before returning
                if not self.ordered_equals(edge, dist=dist):
                    self.reverse_direction()
                return edge
        return None

    #Attract endpoints of edges together
    def join_with_nearby_edge(self, edges, polygon, dist=1.0):
        corners = polygon.get_corners()
        poly_edges = polygon.get_edges()

        new_start = None
        new_end = None
        start_point = self.get_start()
        end_point = self.get_end()

        for edge in edges:
            if self.equals(edge, dist=dist) or edge in poly_edges or edge.get_length() < dist:
                continue
            edge_start = edge.get_start()
            edge_end = edge.get_end()
            
            for corner in corners:
                if hm.compare_points(edge_start, corner, dist):
                    if hm.compare_points(self.get_start(), edge_start, dist):
                        new_start = edge_start
                    elif hm.compare_points(self.get_end(), edge_start, dist):
                        new_end = edge_start 
            
                if hm.compare_points(edge_end, corner, dist):
                    if hm.compare_points(self.get_start(), edge_end, dist):
                        new_start = edge_end       
                    elif hm.compare_points(self.get_end(), edge_end, dist):
                        new_end = edge_end 

        # print(new_start, new_end)
        if new_end is not None:
            # vec = self.get_start() - self.get_end()
            # new_start = new_end + vec
            polygon_edges = [self] + poly_edges
            # move_edge_points(polygon_edges, start_point, new_start)
            
            hm.move_edge_points(polygon_edges, end_point, new_end)
        if new_start is not None:
            # vec = self.get_end() - self.get_start()
            # new_end = new_start + vec
            polygon_edges = [self] + poly_edges
            hm.move_edge_points(polygon_edges, start_point, new_start)
            # move_edge_points(polygon_edges, end_point, new_end)

    """Checks whether two edges have the same endpoints in the same order."""
    def ordered_equals(self, other, dist=1.0):
        is_same = hm.compare_points(self.get_start(), other.get_start(), dist) and hm.compare_points(self.get_end(), other.get_end(), dist)
        return is_same

    """Checks whether two edges have the same endpoints regardless of ordering (i.e. one edge can be the reverse of the other)."""
    def equals(self, other, dist=1.0):
        is_same = hm.compare_points(self.get_start(), other.get_start(), dist) and hm.compare_points(self.get_end(), other.get_end(), dist)
        is_reversed = hm.compare_points(self.get_start(), other.get_end(), dist) and hm.compare_points(self.get_end(), other.get_start(), dist)
        return is_same or is_reversed

    "Splits an edge into two at a point."
    def split_edge_at_point(self, point):
        new_edge = Edge(point, self.get_end())
        split_edge = Edge(self.get_start(), point)
        return new_edge, split_edge

    """Moves the endpoint of an edge to the intersection point, and splits the edge the intersection point lies on into two."""
    def snip_edge(self, intersection_point, snip_dist=1.0):
        start_point = np.array(self.get_start())
        end_point = np.array(self.get_end())

        start_distance = hm.distance(intersection_point, start_point)
        end_distance = hm.distance(intersection_point, end_point) 
        new_edge = None 

        if start_distance > 0.05 and end_distance > 0.05:
            line_length = self.get_length()

            created_edge, split_edge = self.split_edge_at_point(intersection_point)
            if split_edge.get_length() > snip_dist and created_edge.get_length() > snip_dist:
                new_edge = created_edge
                self.set_points(split_edge.get_start(), split_edge.get_end())

                
        return new_edge
            
    """Splits an edge into two if it intersects another edge or is within a reasonable distance from another edge 
    (i.e. if extending it a small distance based on the length of the edge would cause it to intersect another edge)."""
    def split_edge_intersect(self, other_edge, min_wall_length=3.0, percent_proximity=0.2, epsilon_proximity=0.01, debug=False):
        intersection_point = self.get_euclidean_intersection_point(other_edge)
        start_point = np.array(self.get_start())
        end_point = np.array(self.get_end())

        new_edges = []

        if intersection_point is not None:
            start_distance = hm.distance(intersection_point, start_point)
            end_distance = hm.distance(intersection_point, end_point)
            other_start_dist = hm.distance(intersection_point, other_edge.get_start())
            other_end_dist = hm.distance(intersection_point, other_edge.get_end())

            #TODO: Make this min_wall_length
            snip_proximity = max(percent_proximity * self.get_length(), min_wall_length)
            min_dist = min(start_distance, end_distance)
            min_other_dist = min(other_start_dist, other_end_dist)
            if other_edge.on_line(intersection_point) and min_dist < snip_proximity and min_other_dist < snip_proximity and min_dist > epsilon_proximity:
                edge = other_edge.snip_edge(intersection_point)
                if edge is not None:
                    new_edges.append(edge)
                    start_dist = hm.distance(intersection_point, self.get_start())
                    end_dist = hm.distance(intersection_point, self.get_end())
                    if start_dist > end_dist:
                        self.set_points(self.get_start(), intersection_point)
                    else:
                        self.set_points(self.get_end(), intersection_point)
            
        return new_edges
    
    def __eq__(self, other):
        return self.equals(other, dist=0.8)
        # return self.edge_id == other.edge_id

    def __repr__(self):
        start_neighbor_ids = []
        end_neighbor_ids = []
        for each in self.start_neighbors:
            start_neighbor_ids.append(each.edge_id)
        for each in self.end_neighbors:
            end_neighbor_ids.append(each.edge_id)

        repr_string = "ID: " + str(self.edge_id) + ", Start Neighbor IDs: " + str(start_neighbor_ids) + ", End Neighbor IDs: " + str(end_neighbor_ids)
        return repr_string

    def __hash__(self):
        return self.edge_id