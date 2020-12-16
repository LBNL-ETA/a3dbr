import argparse
import polygon_primitives.polygon as pp
import polygon_primitives.edge as pe
import utm
from shapely.geometry import Polygon
import geopandas
import numpy as np
import cv2

_projections = {}

def compute_centroid(points):
    centroid = np.mean(points)
    return centroid

def parse_polygon_file(filename, offset=[0.0, 0.0]):
    edges = []
    shapely_polys = []
    with open(filename, "r") as f:
        lines = f.readlines()
        file_type = lines[0].rstrip()
        edge_lines = lines[1:]
        next_polygon = pp.Polygon()
        polygon_points = []
        for line in edge_lines:
            if line == "\n":
                for i in range(len(polygon_points) - 1):
                    next_index = i + 1
                    edge = pe.Edge(polygon_points[i], polygon_points[next_index])

                    if edge.get_start() == edge.get_end():
                        print(i, next_index)
                    edges.append(edge)

                shap_pol = Polygon(polygon_points)
                shapely_polys.append(shap_pol)
                polygon_points = []

            else:
                points = [float(i) for i in line.split()]
                if file_type != "UTM":
                    x, y, z, l = utm.from_latlon(points[0], points[1])
                else:
                    x = points[0]
                    y = points[1]

                x = x + offset[0]
                y = y + offset[1]
                polygon_points.append((x, y))
                
        if line[-1] != "\n":
            # for i in range(len(polygon_points)):
            #     next_index = (i + 1) % len(polygon_points)
            #     edge = pp.Edge(polygon_points[i], polygon_points[next_index])
            #     next_polygon.add_edge(edge)
            # polygons.append(next_polygon)
            shap_pol = Polygon(polygon_points)
            shapely_polys.append(shap_pol)
            
    return shapely_polys, edges

def get_least_squares_approx(points, ref_points):
    points = np.array(points)
    ref_points = np.array(ref_points)
    point_centroid = compute_centroid(points)
    ref_centroid = compute_centroid(ref_points)
    points = points - point_centroid
    ref_points = ref_points - ref_centroid
    H = np.dot(points, np.transpose(ref_points))
    u,s,v = np.linalg.svd(H)
    rot = np.dot(v, np.transpose(u))
    t = np.dot(rot, -point_centroid) + ref_centroid
    return np.dot(rot, points) + t

def get_fund_mat(points, ref_points):
    return cv2.findFundamentalMat(points, ref_points)

def create_shapely_polygon(merged_polygons, points):
    poly = Polygon(points)
    return poly

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    kp_pairs = zip(mkp1, mkp2)
    return kp_pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--polygon_file", type=str)
    parser.add_argument("--comparison_file", type=str)
    args = parser.parse_args()

    comp_file = args.comparison_file
    poly_file = args.polygon_file
    comp_polygons, edges = parse_polygon_file(poly_file)

# if __name__ == "__main__":
#     main()