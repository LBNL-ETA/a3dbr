import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import json
import polygon_primitives as pp
import utm
from shapely.geometry import Polygon
import numpy as np



#Accepts polygon_coords as a 2D list, and polygon_heights as a 1D array
def save_merged_polygon_geojson(polygon_coords, polygon_heights, filename="merged.json"):
	data = {}
	feature_list = []

	for i in range(len(polygon_coords)):
		coords = []
		polygon_json = {}
		properties = {}
		geometry = {}
		height = polygon_heights[i]
		polygon = polygon_coords[i]
		
		properties["height"] = height
		properties["index"] = i
		for j in range(len(polygon)):
			coords.append(polygon[j])

		geometry["type"] = "Polygon"
		geometry["coordinates"] = [coords]
		
		polygon_json["type"] = "Feature"
		polygon_json["geometry"] = geometry
		polygon_json["properties"] = properties
		polygon_json["id"] = str(i)
		

		feature_list.append(polygon_json)

	data["type"] = "FeatureCollection"
	data["features"] = feature_list
	

	with open(filename, 'w') as outfile:
		json.dump(data, outfile, indent=4, separators=(',', ': '))

def parse_polygon_file(filename, offset=[0.0, 0.0]):
	edges = []
	shapely_polys = []
	heights = []
	with open(filename, "r") as f:
		lines = f.readlines()
		file_type = lines[0].rstrip()
		edge_lines = lines[1:]
		polygon_points = []
		height = 0.0

		for line in edge_lines:
			if line == "\n":
				for i in range(len(polygon_points) - 1):
					next_index = i + 1
					edge = pp.Edge(polygon_points[i], polygon_points[next_index])

					if edge.get_start() == edge.get_end():
						print(i, next_index)
					edges.append(edge)

				shap_pol = Polygon(polygon_points)
				shapely_polys.append(shap_pol)
				heights.append(height)
				polygon_points = []

			else:
				values = [float(i) for i in line.split()]
				height = values[-1]
				if file_type != "UTM":
					x, y, z, l = utm.from_latlon(values[0], values[1])
				else:
					x = values[0]
					y = values[1]

				x = x - offset[0]
				y = y - offset[1]
				polygon_points.append((x, y))
				
		if line[-1] != "\n":
			# for i in range(len(polygon_points)):
			#     next_index = (i + 1) % len(polygon_points)
			#     edge = pp.Edge(polygon_points[i], polygon_points[next_index])
			#     next_polygon.add_edge(edge)
			# polygons.append(next_polygon)
			shap_pol = Polygon(polygon_points)
			shapely_polys.append(shap_pol)
			
	return shapely_polys, edges, heights

def parse_json_file(filename, offset=[0.0, 0.0]):
	polygon_coords = []
	heights = []
	shapely_polys = []
	with open(filename, 'r') as f:
		data = json.load(f)
		feature_list = data['features']
		for i in range(len(feature_list)):
			polygon = []

			polygon_json = feature_list[i]
			coords = polygon_json['geometry']['coordinates'][0]
			for j in range(len(coords)):
				point = coords[j]
				x, y, z, l = utm.from_latlon(point[1], point[0])
				x = x - offset[0]
				y = y - offset[1]
				new_coord = [x,y]
				polygon.append(new_coord)
			polygon_coords.append(polygon)
			heights.append(polygon_json['properties']['height'])

			shap_pol = Polygon(polygon)
			shapely_polys.append(shap_pol)

	return polygon_coords, heights, shapely_polys

def convert_polygons_to_coords(polygons):
	coords = []
	for polygon in polygons:
		coord_list = [list(xy_coord) for xy_coord in list(polygon.exterior.coords)]
		coords.append(coord_list)
	return coords

def get_test_json():
	comp_file_name = "merged2.json"
	offset = np.loadtxt("../params/offset2.txt", dtype=np.float32, usecols=range(2))
	polygon_coords, heights = parse_json_file(comp_file_name)

	polygon_coords = [[[coord[0] - offset[0], coord[1] - offset[1]] for coord in polygon] for polygon in polygon_coords]
	save_merged_polygon_geojson(polygon_coords, heights, filename="merged3.json")

def main():
	# offset = np.loadtxt("../params/offset.txt", dtype=np.float32, usecols=range(2))
	# polygons, edges, heights = parse_polygon_file("../merged1.txt", offset)
	# heights = [4.0, 3.82, 9.1, 7.84, 3.63, 12.6, 3.3, 11.0, 9.5, 3.93, 4.05]
	# coords = convert_polygons_to_coords(polygons)
	# save_merged_polygon_geojson(coords, heights, filename="merged1.json")
	get_test_json()
if __name__ == "__main__":
	main()


