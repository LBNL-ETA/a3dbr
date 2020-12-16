import polygon_primitives.helper_methods as hm
import numpy as np
import shapely.geometry as shp
import polygon_primitives.primitive_edge as pe
import polygon_primitives.file_writer as fw

"""Class definition for a Polygon."""
class Polygon:

    def __init__(self, uid=0, edges=None):
        self.id = uid
        self.dir = -1
        self.completed = False
        self.points = []
        self.adjusted_edges = []
        self.shapely_polygon = None
        self.full_polygon = None
        self.average_point_height = 0.0
        self.total_num_points = 0
        self.polygon_center = None
        self.corners = None
        self.merged = False
        self.medians = []
        self.median = 0.0
        if edges is None:
            self.edges = []
        else:
            self.edges = edges

    def get_edges(self):
        return self.edges

    def get_adjusted_edges(self):
        return self.adjusted_edges

    def get_points(self):
        return self.points

    #Replaces an edge with a different one, does not adjust its neighbors
    def replace_edge(self, current_edge, edge):
        for i in range(len(self.edges)):
            if current_edge == self.edges[i]:
                self.edges[i] = edge
                return

    def add_point(self, point):
        self.points.append(point)

    def add_points(self, points):
        self.points = self.points + points

    def get_edge_ids(self):
        ids = []
        for each in self.edges:
            ids.append(each.edge_id)
        return ids

    """For all edges in the polygon, any of their endpoints corresponding to 'current_endpoint' to 'new_endpoint'."""
    def move_edge_endpoint(self, current_endpoint, new_endpoint):
        for edge in self.edges:
            if hm.compare_points(edge.get_start(), current_endpoint, epsilon=0.3):
                edge.set_start(new_endpoint)
            elif hm.compare_points(edge.get_end(), current_endpoint, epsilon=0.3):
                edge.set_end(new_endpoint)
            
    """Offsets the endpoints of the edges of the polygon by a specified amount."""
    def offset_polygon(self, offset):
        for edge in self.get_edges():
            start = edge.get_start() - offset
            end = edge.get_end() - offset
            edge.set_points(start, end)

    """Compares a polygon with another polygon, by checking whether the edge ids of the edges in both polygons are the same."""
    def compare_polygon(self, polygon):
        ids = self.get_edge_ids()
        other_ids = polygon.get_edge_ids()

        for each in other_ids:
            if not each in ids:
                return False
        return True
    
    """Checks whether the edge is a valid addition to the edges in the polygon,  by checking whether the edge is a 'left' edge or a
    'right' edge. Polygons must have all left edges or all right edges. A traversal with only left edges takes only left turns, and vice-versa."""
    def edge_orientation_valid(self, edge, dist=0.1):
        orientation = self.get_edge_orientation(edge, dist=dist)
        if (orientation == -1) or (orientation == 0 and edge.left_poly is None) or (orientation == 1 and edge.right_poly is None):
            return True
        else:
            return False

    """Chooses which edge if any, from a list of edges, to add to the polygon, such that a traversal makes either all left turns or
    all right turns."""
    def choose_member_edge(self, edges, last_edge, dist=0.5):
        if len(edges) == 0:
            return None

        for edge in edges:
            for other_edge in self.edges:
                if edge == other_edge:
                    return edge

        if len(self.edges) == 0:
            return edges[0]

        min_angle, max_angle = float("inf"), -float("inf")
        index1, index2 = 0, 0
        # If a polygon exists for an edge, and the polygon matches that orientation, we should not add the edge
        for i in range(len(edges)):
            edge = edges[i]

            #Somewhat hack-y
            if last_edge.on_line(edge.get_start()) and last_edge.on_line(edge.get_end()):
                continue
            angle_between = last_edge.get_angle_between_edges(edge, dist)

            if angle_between < min_angle and angle_between != float("inf"):
                min_angle = angle_between
                index1 = i
            if angle_between > max_angle and angle_between != float("inf"):
                max_angle = angle_between
                index2 = i
        if self.dir == 0:
            if min_angle > 10.0:
                return None
            potential_edge = edges[index1]

            is_edge_valid = self.edge_orientation_valid(potential_edge, dist)
            if is_edge_valid:
                return potential_edge
        elif self.dir == 1:
            if max_angle < -10.0:
                return None
            potential_edge = edges[index2]

            is_edge_valid = self.edge_orientation_valid(potential_edge, dist)
            if is_edge_valid:
                return potential_edge
        else:
            potential_edge = edges[index1]
            is_edge_valid = self.edge_orientation_valid(potential_edge, dist)
            if is_edge_valid:
                self.dir = 0
                return potential_edge
        return None
    
    """Inserts the edge into the polygon's edge list, at either the front or the back."""
    def add_edge(self, edge, prepend=False):
        if prepend:
            self.edges.insert(0, edge)
        else:
            self.edges.append(edge)

    """Removes an edge from the polygon."""
    def remove_edge(self, edge):
        self.edges.remove(edge)

    """Returns whether an edge is present in the polygon."""
    def check_edge_membership(self, edge):
        if len(self.edges) <= 1:
            return True

    """Returns the orientation of an edge, relative to the edges of the polygon, by checking whether the center of the polygon is above 
    or below the edge. Note the edge is first assigned its absolute order."""
    def get_edge_orientation(self, edge, dist=0.1):
        corners = self.search_for_corners(dist=dist)

        #Note: search_for_corners() changes the point order, so we need to make this consistent with set_points_absolute_order()
        edge.set_points_absolute_order()

        points = hm.get_unique_points(self.edges)
        non_edge_points = [point for point in points if not hm.compare_points(point, edge.get_start()) and not hm.compare_points(point, edge.get_end())]
        if len(points) > 0 and len(non_edge_points) > 0:
            if len(corners) >= 4:
                comp_point = self.get_point_average()
                is_left = edge.check_point_below_edge(comp_point)
            else:
                comp_point = self.get_point_average()
                is_left = edge.check_point_below_edge(comp_point)
            return is_left
        return -1

    """Iterates through the polygon's edges, and assigns the polygon to either its left or right""" 
    def set_polygon_edges_direction(self):
        for each in self.edges:
            orientation = self.get_edge_orientation(each)
            if orientation == 0 and each.left_poly is None:
                each.left_poly = self
            elif orientation == 1 and each.right_poly is None:
                each.right_poly = self

    """Get the center point of the polygon."""
    def get_polygon_center(self):
        if self.polygon_center is None:
            self.set_polygon_center()
        return self.polygon_center

    """Returns the average point in the polygon."""
    def get_point_average(self):
        points = hm.get_unique_points(self.edges)
        average_point = np.array([0.0, 0.0])
        for point in points:
            average_point += np.array(point) / len(points)
        return average_point

    """Computes the point at the center of the polygon and saves it for future use.""" 
    def set_polygon_center(self):
        corners = self.search_for_corners()
        center_val = np.sum(corners, axis=0) / len(corners)
        self.polygon_center = center_val
        return center_val

    """Moves an edge in the perpendicular direction towards the center. Used to shrink a polygon."""
    def move_edge_toward_center(self, edge, movement_amount):
        center_dir = self.get_polygon_center() - edge.get_start()
        normal = edge.find_normal()
        if np.dot(normal, center_dir) < 0.0:
            normal = -1.0 * normal

        if np.linalg.norm(normal) == 0.0:
            return edge
        adjusted_edge_start = edge.get_start() + np.array(normal) * movement_amount
        adjusted_edge_end = edge.get_end() + np.array(normal) * movement_amount
        adjusted_edge = pe.Edge(adjusted_edge_start, adjusted_edge_end, -1, temp_edge=edge.temp_edge, order_points=False)
        return adjusted_edge

    """Checks if a point is in the polygons corners."""
    def point_in_corners(self, point, dist):
        corners = self.search_for_corners(dist)
        for corner in corners:
            if hm.compare_points(corner, point, dist):
                return True
        return False

    """Shrinks a polygon by a specified amount, by moving the edges in the direction of the center. Used when computing the points
    on the inside of the polygon, in order to exclude points on the boundary (which have variable heights). Assigns a shapely polygon
    to check for internal points."""
    def set_internal_polygon(self, boundary_exclusion_width=0.5, dist=1.0):
        right_angled = True
        corners = self.search_for_corners(dist)
        if len(corners) == 0:
            return
        
        self.set_polygon_center()
        self.adjusted_edges = []

        boundary_edges = []
        lengths = []
        for i in range(len(corners)):
            j = (i + 1) % len(corners)
            boundary_edge = pe.Edge(corners[i], corners[j])
            boundary_edges.append(boundary_edge)
            lengths.append(boundary_edge.get_length())

        if min(lengths) < 1.5:
            boundary_exclusion_width = 0.2

        for i in range(len(boundary_edges)):
            edge = boundary_edges[i]
            next_edge = boundary_edges[(i + 1) % len(boundary_edges)]
            angle = edge.get_angle_between_edges(next_edge)
            if abs(angle - 90.0) > 15 and abs(angle + 90) > 15:
                right_angled = False
            adjusted_boundary = self.move_edge_toward_center(edge, boundary_exclusion_width)
            is_start, is_end = False, False
            if self.point_in_corners(edge.get_start(), dist): 
                is_start = True
            if self.point_in_corners(edge.get_end(), dist):
                is_end = True
            adjusted_boundary.shorten_edge_by_amount(boundary_exclusion_width, is_start=is_start, is_end=is_end)
            self.adjusted_edges.append(adjusted_boundary)
        is_poly = self.create_shapely_polygon(self.adjusted_edges) and right_angled
        return is_poly

    """Returns whether a set of edges is a valid polygon. Creates a polygon from the shapely library from the current polygon. 
    Used for the point in polygon test."""
    def create_shapely_polygon(self, adjusted_edges):
        corners = hm.get_corner_ordering(adjusted_edges)
        if len(corners) == 0:
            return False

        shapely_polygon = shp.Polygon(corners)
        self.shapely_polygon = shapely_polygon
        return True

    """Checks if point is inside the polygon. Requires that the polygons shapely_polygon has been assigned."""
    def is_point_within_boundary(self, point):
        p1 = shp.Point(point)
        return self.shapely_polygon.contains(p1)

    """Returns the endpoints of any polygon edge that does not connect to any other edge in the polygon."""
    def get_open_points(self, dist=1.0):
        if self.is_closed():
            return []
        ordered_edges = self.order_edges()
        start_point = ordered_edges[0].get_start()
        end_point = ordered_edges[-1].get_end()
        if not hm.compare_points(start_point, end_point, dist):  
            return [start_point, end_point]
        return []

    """Returns any polygon edge that has an open endpoint, i.e. one that connects to only one edge."""
    def get_open_edge_segments(self):
        open_points = self.get_open_points()
        edges = []
        if len(open_points) < 2:
            return []

        ordered_edges = self.order_edges()
        return [ordered_edges[0], ordered_edges[-1]]

    #DEPRECATED
    """Returns any polygon edge that has an open endpoint, i.e. one that connects to only one edge."""
    def get_open_edges(self, corners, open_points, dist):
        open_segments = self.get_open_edge_segments()
        open_edges = []

        for segment in open_segments:
            for corner in corners:
                for point in open_points:
                    start = segment.get_start()
                    end = segment.get_end()
                    # if hm.compare_vector()

                    if hm.collinear(corner, point, start) and hm.collinear(corner, point, end):
                        edge = pe.Edge(corner, point)
                        open_edges.append(edge)
                
        return open_edges

    """Returns whether a polygon is rectangular."""
    def is_rectangular(self):
        ordered_edges = self.order_edges()
        num_turns = 0
        for i in range(len(self.edges)):
            edge = self.edges[i]
            next_index = (i + 1) % len(self.edges)
            other_edge = self.edges[next_index]
            shared_point = edge.get_shared_point(other_edge)
            if shared_point is not None:
                angle = edge.get_angle_between_edges(other_edge)
                if (angle > 60 and angle < 120) or (angle < -60 and angle > -120):
                    num_turns += 1
        if num_turns == 4:
            return True
        return False

    """Returns a traversal of the polygon's edges. Direction can be reversed."""
    def order_edges(self, reverse=False, edges=None):
        if edges is None:
            edges = self.get_edges()
            if len(edges) == 0:
                return []
        next_edge = edges[0]
        edge_order = []
        edge_order.append(next_edge)

        for _ in range(len(edges) + 2):
            if next_edge is None:
                break
            # neighbors = next_edge.get_end_neighbors()
            edge_to_add = None
            for edge in next_edge.get_neighbors():
                if edge in edge_order:
                    continue
                in_start = edge in next_edge.get_start_neighbors()
                in_end = edge in next_edge.get_end_neighbors()
                
                if edge in edges:
                    if in_end and hm.distance(next_edge.get_end(), edge.get_end()) < hm.distance(next_edge.get_end(), edge.get_start()):
                        edge.reverse_direction()
                    elif in_start and hm.distance(next_edge.get_start(), edge.get_start()) < hm.distance(next_edge.get_start(), edge.get_end()):
                        edge.reverse_direction()
                    edge_to_add = edge
                    break
            if edge_to_add is not None:
                if in_end:
                    edge_order.append(edge_to_add)
                else:
                    edge_order.insert(0, edge_to_add)
                next_edge = edge_to_add
            else:
                next_edge = edges[0]
        if reverse:
            edge_order = hm.reverse_edges(edges)
        return edge_order

    """Returns the endpoints of the polygon's edges that are corners."""
    def search_for_valid_corners(self, dist=0.8):
        corners = []
        corner_edges = []
        angles = []
        self.edges = self.order_edges()
        for i in range(len(self.edges)):
            edge = self.edges[i]
            next_index = (i + 1) % len(self.edges)
            other_edge = self.edges[next_index]
            shared_point = edge.get_shared_point(other_edge, dist)
            if shared_point is not None:
                angle = edge.get_angle_between_edges(other_edge, dist)
                if (angle > 60 and angle < 120) or (angle < -60 and angle > -120):
                    in_corners = False
                    for corner in corners:
                        if hm.compare_points(corner, shared_point, dist):
                            in_corners = True
                    if not in_corners:
                        corners.append(shared_point)
                        angles.append(angle)
        median_angle = np.median(angles)
        valid_corners = []
        for i in range(len(angles)):
            if abs(angles[i] - median_angle) < 60:
                valid_corners.append(corners[i])
        return valid_corners

    """Returns all edges between two points in a traversal of the polygon's edges."""
    def get_edges_between_points(self, start_point, end_point):
        edge_order = self.order_edges()
        edges = []
        edges = hm.search_along_line(edge_order, start_point, end_point=end_point)
        if len(edges) == 0:
            edge_order = hm.reverse_edges(edge_order)
            edges = hm.search_along_line(edge_order, start_point, end_point=end_point)
        return edges

    """Returns the first collinear edge in an ordered traversal that is collinear with the input edge."""
    def get_first_collinear_edge(self, edge, edge_order):
        for each in edge_order:
            if edge.is_collinear(each):
                return each
        return None

    #Takes one of the two open endpoints of the polygon, and draws an edge parallel to the neighboring corners, to form a new polygon
    def close_closest_edge_to_corners(self, corners, closest_edges, dist):
        if len(closest_edges) == 0:
            return [], []
        edge_to_close = closest_edges[0]

        start = edge_to_close.get_start()
        end = edge_to_close.get_end()

        end_corner = corners[0]
        ordered = self.order_edges()
        # return [pe.Edge(end, corners[1])]

        collinearity_dir1 = hm.get_collinearity(start, end, corners[0])
        collinearity_dir2 = hm.get_collinearity(start, end, corners[1])

        if collinearity_dir1 < collinearity_dir2:
            
            end_corner = corners[1]
            vec = np.array(corners[1]) - np.array(corners[0])

            if hm.compare_points(start, corners[0], dist):
                start_point = end
            else:
                start_point = start
            between_edges = hm.find_edges_between_points(ordered, start_point, corners[1])

        else:
            vec = np.array(corners[0]) - np.array(corners[1])
            
            if hm.compare_points(start, corners[1], dist):
                start_point = end
            else:
                start_point = start

            between_edges = hm.find_edges_between_points(ordered, start_point, corners[0])

        if len(between_edges) == 0:
            return [], []

        end_point = start_point + vec

        edge1 = pe.Edge(start_point, end_point, temp_edge=True)
        edge2 = pe.Edge(end_corner, end_point)

        return [edge1], between_edges + [edge2]

    """Determines the first edge connecting to a corner that does not lie between two corners."""
    def closest_edge_from_corner(self, corner, dist):
        corner_edges = []
        for edge in self.get_edges():
            if hm.compare_points(edge.get_start(), corner, epsilon=dist) or hm.compare_points(edge.get_end(), corner, epsilon=dist):
                corner_edges.append(edge)

        open_points = self.get_open_points()
        if hm.distance(open_points[0], corner) < hm.distance(open_points[1], corner):
            between_edges = hm.search_between_corners(self.get_edges(), corner, open_points[0])
            collinear_point = open_points[1]
        else:
            between_edges = hm.search_between_corners(self.get_edges(), corner, open_points[1])
            collinear_point = open_points[0]

        closest_edge = None
        for edge in corner_edges:
            if hm.collinear(corner, collinear_point, edge.get_start()) and hm.collinear(corner, collinear_point, edge.get_end()):
                closest_edge = edge

        return closest_edge, between_edges

    """Completes a polygon that has a single corner and two open points (i.e. an L shape.)."""
    def close_from_single_corner(self, corner, open_points, dist=0.8):
        corner_edges = []
        for edge in self.get_edges():
            if hm.compare_points(edge.get_start(), corner, epsilon=dist) or hm.compare_points(edge.get_end(), corner, epsilon=dist):
                corner_edges.append(edge)
        if len(corner_edges) == 0:
            raise Exception("Polygon does not have corners!")
        edge_points = []
        for edge in corner_edges:
            if hm.distance(edge.get_start(), corner) < hm.distance(edge.get_end(), corner):
                edge_points.append(edge.get_end())
            else:
                edge_points.append(edge.get_start())

        start_point1 = edge_points[0]
        start_point2 = edge_points[1]

        vec = np.array(start_point1) - corner
        end_point = start_point2 + vec
        edge1 = pe.Edge(start_point2, end_point, temp_edge=True)
        edge2 = pe.Edge(start_point1, end_point, temp_edge=True)
        return [edge1, edge2], corner_edges

    #TODO: Add comments here
    def get_segments_closest_to_corner(self, corners, dist, min_length=1.5):
        open_segments = self.get_open_edge_segments()
        if len(open_segments) < 2:
            return []

        closest_open_segments = []        
        edge_order = self.order_edges()
        edges1 = []
        edges2 = []
        for i in range(len(edge_order)):
            edge = edge_order[i]
            
            start = edge.get_start()
            end = edge.get_end()

            if hm.compare_points(end, corners[0], dist) or (len(corners) > 1 and hm.compare_points(end, corners[1], dist)):
                if len(edges1) == 0:
                    #Note: May run into issues if min length is greater than the distance to a corner (but probably should not be resolved here)
                    edge_candidates = edge_order[:i+1]
                    edge_candidates.reverse()
                    edges1 = hm.edge_list_above_min_length(edge_candidates, min_length)


                elif i + 1 < len(edge_order):
                    edge_candidates = edge_order[i+1:]
                    edges2 = hm.edge_list_above_min_length(edge_candidates, min_length)

        if len(edges1) > 0 and len(edges2) > 0:
            corner_edge = pe.Edge(corners[0], corners[1])
            angle1 = edges1[0].get_angle_between_edges(corner_edge)
            angle2 = edges2[0].get_angle_between_edges(corner_edge)

            is_perp1 = abs(angle1) > 88 and abs(angle1) < 92
            is_perp2 = abs(angle2) > 88 and abs(angle2) < 92

            if not is_perp1 and is_perp2:
                closest_open_segments = edges2
            elif is_perp1 and not is_perp2:
                closest_open_segments = edges1


            if hm.get_edges_length(edges1) < hm.get_edges_length(edges2):
                closest_open_segments = edges1
            else:
                closest_open_segments = edges2
    
        return closest_open_segments

    """Splits a completed polygon into two smaller polygons, by adding an edge."""
    #TODO: Check search for corners.
    def split_completed_polygon(self, dist):
        corners = self.search_for_valid_corners(dist)
        # corners = self.search_for_corners(dist)
        corners = hm.choose_min_distance_points(corners)
        closest_edges = self.get_segments_closest_to_corner(corners, dist)

        new_edge, old_edges = self.close_closest_edge_to_corners(corners, closest_edges, dist)
        return new_edge, old_edges

    """Returns a list of corners in the polygon."""
    def search_for_corners(self, dist=0.8):
        edges = self.order_edges()
        self.edges = edges
        corners = hm.search_for_corners(edges, dist)
        if len(corners) >= 4:
            self.corners = corners
        return corners

    """Calculates the distance between each pair of ordered corners by traversing along the edges."""
    def get_min_corner_distance(self, corners):
        if len(corners) < 2:
            return []

        min_dist = -1
        corner1 = corners[0]
        corner2 = corners[1]
        for i in range(len(corners)):
            for j in range(len(corners)):
                if i == j:
                    continue
                edges = self.get_edges_between_points(corners[i], corners[j])
                dist = hm.get_edges_length(edges)
                if dist < min_dist or min_dist < 0:
                    min_dist = dist
                    corner1 = corners[i]
                    corner2 = corners[j]
        return corner1, corner2
    
    """Completes a polygon by adding an edge connecting to the closest open segment, and going across the partial polygon."""
    def complete_polygon_close_edges(self, dist):
        corners = self.get_corners(dist)
        if len(corners) > 2:
            corners = self.get_min_corner_distance(corners)
        closest_edges = self.get_segments_closest_to_corner(corners, dist)
        if len(closest_edges) == 0:
            return [], self.get_edges()
        new_edge, old_edges = self.close_closest_edge_to_corners(corners, closest_edges, dist)
        return new_edge, old_edges

    #TODO: Check whether this is necessary.
    def complete_polygon_extend_shortest(self, open_edges):
        new_edges = []

        shortest_edge = None
        longest_edge = None
        for edge in open_edges:
            if shortest_edge is None or edge.get_length() < shortest_edge.get_length():
                shortest_edge = edge
            if longest_edge is None or edge.get_length() > longest_edge.get_length():
                longest_edge = edge

        if shortest_edge is not None:
            open_points = self.get_open_points()
            end_open = False
            for point in open_points:
                if hm.compare_points(longest_edge.get_end(), point):
                    end_open = True

            if end_open:
                vec = longest_edge.get_end() - longest_edge.get_start()
                edge1 = pe.Edge(shortest_edge.get_end(), shortest_edge.get_start() + vec, temp_edge=True)
                edge2 = pe.Edge(shortest_edge.get_start() + vec, longest_edge.get_end(), temp_edge=True)
            else:
                vec = longest_edge.get_start() - longest_edge.get_end()
                edge1 = pe.Edge(shortest_edge.get_start(), shortest_edge.get_end() + vec, temp_edge=True)
                edge2 = pe.Edge(shortest_edge.get_end() + vec, longest_edge.get_start(), temp_edge=True)

            new_edges.append(edge1)
            new_edges.append(edge2)
        return new_edges, self.get_edges()

    """Resets the polygon's height to 0, and the number of points used in its height averaging."""
    def reset_height(self):
        self.average_point_height = 0.0
        self.total_num_points = 0

    """Updates the height of a polygon by taking a weighted average of the current height and input height."""
    def update_height(self, height, num_points):
        update_amount = ((height - self.average_point_height) * num_points) / (self.total_num_points + num_points)
        self.average_point_height = self.average_point_height + update_amount
        self.total_num_points = self.total_num_points + num_points

    def get_corners(self, dist=0.5):
        pot_corners = self.search_for_corners(dist)

        open_points = self.get_open_points()

        corners = []
        for corner in pot_corners:
            to_add = True
            for point in open_points:
                if hm.compare_points(corner, point):
                    to_add = False
            if to_add:
                corners.append(corner)

        return corners

    """Completes a polygon based on the number of corners it has, and the number of open points (see the definition for get_open_points)."""
    def complete_polygon(self, dist=0.5):

        if self.completed:

            is_polygon = self.set_internal_polygon(dist=dist)
            if is_polygon:
                return [], self.get_edges()

        new_edges = []
        old_edges = []
        

        open_points = self.get_open_points()
        corners = self.get_corners(dist)

        open_edges = self.get_open_edges(corners, open_points, dist)

        #Complete polygon based on the number of corners it currently has
        if len(corners) == 1 and len(open_points) == 2:
            new_edges, old_edges = self.close_from_single_corner(corners[0], open_points, dist)

        elif len(corners) == 2 and len(open_points) == 2:
            new_edges, old_edges = self.complete_polygon_close_edges(dist)
            if len(new_edges) == 0:
                new_edges, old_edges = self.complete_polygon_extend_shortest(open_edges)
        elif len(corners) > 4 and len(open_points) == 0:
            new_edges, old_edges = self.split_completed_polygon(dist)

        elif len(corners) == 4 and len(open_points) == 0:
            return [], self.get_edges()
        elif len(corners) > 2 and len(corners) + len(open_points) >= 4:
            new_edges, old_edges = self.complete_polygon_close_edges(dist)
        return new_edges, old_edges

    """Returns the edges shared between the current polygon and another polygon."""
    def get_shared_edges(self, other_polygon):
        shared_edges = []
        for edge in self.get_edges():
            for other_edge in other_polygon.get_edges():
                if edge == other_edge:
                    shared_edges.append(edge)
        return shared_edges

    #TODO: Move this to helper methods
    def merge_polygons(self, polygons, shared_edges):
        merged_polygon = Polygon(uid=-1, edges=[])
        merged_total_points = self.total_num_points
        merged_total_height = self.average_point_height * self.total_num_points

        self.merged = True
        for edge in self.get_edges():
            if edge not in shared_edges:
                merged_polygon.add_edge(edge)

        for polygon in polygons:
            polygon.merged = True
            
            for edge in polygon.get_edges():
                if edge not in shared_edges and edge not in merged_polygon.get_edges():

                    merged_polygon.add_edge(edge)

            merged_total_points = merged_total_points + polygon.total_num_points
            merged_total_height = merged_total_height + (polygon.average_point_height * polygon.total_num_points)

        if merged_total_points > 0.0:
            merged_polygon.average_point_height = merged_total_height / merged_total_points
        merged_polygon.total_num_points = merged_total_points

        return merged_polygon

    #Move this to helper methods
    def check_and_merge_polygons(self, polygons, max_height_diff=1.0):
        if self.merged:
            return None

        polygons_to_merge = []
        polygons_to_merge.append(self)
        height = self.average_point_height
        all_shared_edges = []
        for other_polygon in polygons:
            if other_polygon in polygons_to_merge or other_polygon.merged:
                continue

            shared_edges = []
            for polygon in polygons_to_merge:
                shared_edges = shared_edges + polygon.get_shared_edges(other_polygon)

            if len(shared_edges) > 0:
                other_height = other_polygon.average_point_height
                if abs(height - other_height) < max_height_diff:
                    polygons_to_merge.append(other_polygon)
                    all_shared_edges = all_shared_edges + shared_edges

        ids = []
        for polygon in polygons_to_merge:
            ids.append(polygon.id)
        merged_polygon = self.merge_polygons(polygons_to_merge, all_shared_edges)
        return merged_polygon

    """Returns whether a polygon is closed, i.e. has at least 4 corners and no open points."""
    def is_closed(self, dist=1.0):
        is_polygon = self.set_internal_polygon(dist=dist)
        ordered_edges = self.order_edges()
        start_edge = ordered_edges[0]
        end_edge = ordered_edges[-1]

        if is_polygon and (hm.compare_points(start_edge.get_start(), end_edge.get_end(), dist) or \
            hm.compare_points(start_edge.get_end(), end_edge.get_start(), dist)):
            return True
        return False

    """Equality is determined by comparing the ids of the polygons."""
    def __eq__(self, other):
        if self.id == other.id:
            return True
        return False

"""Class definition for a merged polygon."""
class MergedPolygon():
    def __init__(self, polygons, uid=0):
        self.polygons = polygons
        self.uid = uid
        self.to_destroy = False
        self.outer_edges = []
        self.compute_average_height()

    """Computes the average height of the merged polygon."""
    def compute_average_height(self):
        self.average_height = 0.0
        self.total_points = 0

        merged_total_height = 0.0        
        for polygon in self.polygons:
            self.total_points = self.total_points + polygon.total_num_points
            merged_total_height = merged_total_height + (polygon.average_point_height * polygon.total_num_points)

        if self.total_points > 0:
            self.average_height = merged_total_height / self.total_points

    """Gets all edges of the merged polygon."""
    def get_edges(self):
        edges = []
        for polygon in self.polygons:
            for edge in polygon.get_edges():
                if edge not in edges:
                    edges.append(edge)
        return edges

    """Returns only the outer edges of the polygon."""
    def get_outer_edges(self):

        edge_counts = dict()
        for polygon in self.polygons:
            for edge in polygon.get_edges():
                if edge in edge_counts:
                    edge_counts[edge] += 1
                else:
                    edge_counts[edge] = 1
        edges = [edge for edge in edge_counts.keys() if edge_counts[edge] == 1]
        self.outer_edges = edges
        return edges

    """Traverses the edges and orders them."""
    def order_edges(self, reverse=False, edges=None):
        if edges is None:
            edges = self.get_edges()
            if len(edges) == 0:
                return []
        next_edge = edges[0]
        edge_order = []
        edge_order.append(next_edge)

        for _ in range(len(edges) + 2):
            if next_edge is None:
                break
            # neighbors = next_edge.get_end_neighbors()
            edge_to_add = None
            for edge in next_edge.get_neighbors():
                if edge in edge_order:
                    continue
                in_start = edge in next_edge.get_start_neighbors()
                in_end = edge in next_edge.get_end_neighbors()
                
                if edge in edges:
                    if in_end and hm.distance(next_edge.get_end(), edge.get_end()) < hm.distance(next_edge.get_end(), edge.get_start()):
                        edge.reverse_direction()
                    elif in_start and hm.distance(next_edge.get_start(), edge.get_start()) < hm.distance(next_edge.get_start(), edge.get_end()):
                        edge.reverse_direction()
                    edge_to_add = edge
                    break
            if edge_to_add is not None:
                if in_end:
                    edge_order.append(edge_to_add)
                else:
                    edge_order.insert(0, edge_to_add)
                next_edge = edge_to_add
            else:
                next_edge = edges[0]
        if reverse:
            edge_order = hm.reverse_edges(edges)
        return edge_order

    """Determines whether a point lies within the merged polygon."""
    def point_in_merged_polygon(self, point):
        for polygon in self.polygons:
            if polygon.is_point_within_boundary(point):
                return True
        return False

    """Absorbs another merged polygon with the current one, if they are adjacent and have a height differential less than max_height_diff."""
    def absorb_merged_polygon(self, other_merged_polygon, max_height_diff=1.7):
        if other_merged_polygon.total_points == 0:
            # print("No points...")
            return False

        edges = self.get_edges()
        other_edges = other_merged_polygon.get_edges()
        b_shared_edge = False
        for other_edge in other_edges:
            if other_edge in edges:
                b_shared_edge = True

        if b_shared_edge and abs(self.average_height - other_merged_polygon.average_height) < max_height_diff:
            self.polygons = self.polygons + other_merged_polygon.polygons
            total_num_points = self.total_points + other_merged_polygon.total_points
            self.average_height = ((self.average_height * self.total_points) + 
                (other_merged_polygon.average_height * other_merged_polygon.total_points)) / total_num_points
            self.total_points = total_num_points

            other_merged_polygon.to_destroy = True
            return True
        return False

    """Equality is determined by comparing merged polygon ids."""
    def __eq__(self, other):
        if self.uid == other.uid:
            return True
        return False