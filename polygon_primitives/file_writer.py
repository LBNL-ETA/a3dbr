import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polygon_primitives.helper_methods as hm


#Saves the points for each merged polygon in a list to a file. Note that the first corner is repeated at the end.
def save_merged_polygon_points(merged_polygons, offset, filename="merged1.txt"):
    with open(filename, 'w') as f:
        f.write(str(len(merged_polygons)) + " UTM\n")
        for polygon in merged_polygons:
            height = polygon.average_height
            edges = polygon.get_outer_edges()
            print("Next polygon...")
            print(len(edges))
            edges = polygon.order_edges(edges=edges)
            corners = hm.search_for_corners(edges)
            
            print(len(corners))
            print(len(edges))
            
            f.write(str(len(corners) + 1) + " " + str(height) + "\n")
            for corner in corners:
                point = corner + offset[0:2]
                string = str(point[0]) + " " + str(point[1]) + "\n"
                f.write(string)

            last_point = corners[0]
            last_point = last_point + offset[0:2]
            string = str(last_point[0]) + " " + str(last_point[1]) + "\n"
            f.write(string)
        f.close()

def load_merged_polygons(filename="merged1.txt"):
    polygons = []
    heights = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        num_polys, point_format = lines[0].split()
        point_format = point_format.rstrip()
        lines = lines[1:]

        polygon = []
        curr_polygon_len = 0
        counter = 0
        for next_line in lines:
            if curr_polygon_len == counter:
                if len(polygon) > 0:
                    polygons.append(polygon)
                    heights.append(height)
                    polygon = []
                vals = next_line.rstrip().split()
                curr_polygon_len, height = int(vals[0]), float(vals[1])
                counter = 0
                continue
            x, y = [float(val) for val in next_line.rstrip().split()]
            polygon.append([x,y])
            counter += 1
        polygons.append(polygon)
        heights.append(height)
        f.close()
    return polygons, heights, point_format

#Loads 3D polygon facades, including the top and the bottom of the polygons. Output is a list of polygons, where a polygon is a list of facades, and a facade is a list of points.
def load_merged_polygon_facades(filename="merged1.txt"):
    polygons, heights, point_format = load_merged_polygons(filename)
    polygon_list = []
    polygon_facade_type_list = []
    for poly_index in range(len(polygons)):
        polygon = polygons[poly_index]
        polygon_facades = []
        polygon_facade_types = []

        bottom_facade_points = []
        top_facade_points = []

        for i in range(len(polygon)):
            point = polygon[i]
            bottom_point = [point[0], point[1], 0.0]
            top_point = [point[0], point[1], heights[poly_index]]
            bottom_facade_points.append(bottom_point)
            top_facade_points.append(top_point)

            if i < len(polygon) - 1:
                

                next_point = polygon[i + 1]
                next_top = [next_point[0], next_point[1], heights[poly_index]]
                next_bottom = [next_point[0], next_point[1], 0.0]
                facade = [bottom_point, top_point, next_top, next_bottom]
                polygon_facades.append(facade)
                polygon_facade_types.append(0)
        polygon_facades.append(bottom_facade_points)
        polygon_facades.append(top_facade_points)

        polygon_facade_types.append(1)
        polygon_facade_types.append(2)

        polygon_list.append(polygon_facades)
        polygon_facade_type_list.append(polygon_facade_types)

        
    return polygon_list, polygon_facade_type_list, point_format