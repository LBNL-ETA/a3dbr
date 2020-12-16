from matplotlib.path import Path
import numpy as np
from PIL import Image
import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polygon_primitives.file_writer as fw
import image_processing.contour_extraction as ce
import image_processing.utils as utils

from scipy.spatial import Delaunay    

"""Returns the index of the facade containing the specified windows. Takes as input a list
of facades, as well as a list of their projective distances to a camera image. """
def get_facade_index_with_window(facades, projective_distances, window):
    
    facade_index = -1
    for i in range(len(facades)):
        facade = facades[i]
        proj_distance = projective_distances[i]
        
        if utils.point_within_polygon(window.centroid, facade):
            if facade_index == -1:
                facade_index = i
            elif projective_distances[facade_index] > proj_distance:
                facade_index = i
    if facade_index >= 0:
        return facade_index
    
    return -1

"""Adds a specified polygon in 2D to the image."""
def add_poly_to_image(polygon, image, color=(255, 50, 50, 255)):
    im_arr = np.array(image)
    poly_points = get_shapely_polygon_points(polygon)
    image_height, image_width = im_arr.shape[:2]
    y, x = np.indices((image_height, image_width)) # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T 

    p = Path(poly_points) # make a polygon
    grid = p.contains_points(points)
    mask = grid.reshape(image_height,image_width) # now you have a mask with points inside a polygon

    im_arr[np.where(mask)] = color
    image = Image.fromarray(im_arr)
    return image

"""Plots a list of polygons on an image and displays it to the screen."""
def plot_polygons_on_image(polygons, image):
    for polygon in polygons:
        image = add_poly_to_image(polygon, image)
    image.show()

"""Plots a single polygon on an image."""
def plot_polygon_on_image(polygon, image):
    plot_polygons_on_image([polygon], image)

"""Plots a list of facades and a list of windows on an image."""
def plot_windows_facade(projected_facade, windows, image):
    image = add_poly_to_image(projected_facade, image)
    
    for window in windows:
        image = add_poly_to_image(window, image, color=(50, 50, 255, 255))
    image.show()

"""Projects a list of merged polygons onto a camera image, requires the camera's offset and the pmatrix."""
def project_merged_polygons(merged_polygons, offset, pmatrix):
    projected_facades = []
    projective_distances = []
    for i in range(len(merged_polygons)):
        polygon = merged_polygons[i]
        proj_facades, proj_distances = utils.get_projected_facades(polygon, pmatrix, offset=offset)
        facade_polygons = utils.convert_polygons_shapely(proj_facades)
        projected_facades = projected_facades + facade_polygons
        projective_distances = projective_distances + proj_distances
    return projected_facades, projective_distances

"""Returns a dictionary where the keys are indices into the facade list, and the values are the window polygons."""
def get_facade_window_map(window_polygons, projected_facades, projective_distances):
    facade_window_map = dict()
    for window in window_polygons:
        facade_index_for_window = get_facade_index_with_window(projected_facades, projective_distances, window)
        if facade_index_for_window != -1:
            if facade_index_for_window in facade_window_map.keys():
                facade_window_map[facade_index_for_window].append(window)
            else:
                facade_window_map[facade_index_for_window] = [window]
    return facade_window_map

"""Retrieves the points in a shapely polygon as a numpy array."""
def get_shapely_polygon_points(shapely_polygon):
    poly_points = shapely_polygon.exterior.xy
    poly_points = np.vstack((poly_points[0], poly_points[1])).transpose()
    return poly_points

"""Plots the triangulation of a set of points."""
def plot_triangulation(points, tri):
    import matplotlib.pyplot as plt
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()

"""Normalizes a non-zero vector."""
def normalize(v):
    return v / np.linalg.norm(v)

"""Computes the area of a polygon using the input points. This computes the cross product between subsequent point vectors in the list. 
More information can be found here: http://geomalgorithms.com/a01-_area.html"""
def get_poly_area_from_points(points):
    points = np.array(points)
    cross_sum = np.zeros(3)
    unit_norm = normalize(np.cross(points[1] - points[0], points[2] - points[0]))
    for i in range(points.shape[0] - 1):
        next_index = (i + 1) % (points.shape[0])
        cross = np.cross(points[i], points[next_index])
        cross_sum += cross
    area = abs(np.dot(unit_norm, cross_sum) / 2)
    return area

"""Calculates 3D coordinates for the window, from its 2D points. Requires the facade as well as its projection in the camera image. We first triangulate the projected
facade points. Then, we compute barycentric coordinates (i.e. a linear interpolation of the triangle points) for each of the window points. Finally, for each window point we use 
its barycentric coefficients along with the corresponding facade point coordinates in 3D to compute a 3D window coordinate."""
def get_window_coordinates(projected_facade, facade, window):
    
    #Gets the coordinates for each point in the projected facade and window
    proj_points = get_shapely_polygon_points(projected_facade)
    window_points = get_shapely_polygon_points(window)

    #Triangulates the projected points using Delaunay triangulation (though any triangulation scheme should work in theory)
    tri = Delaunay(proj_points)
    
    #Gets the indices of the triangles that each of the window points lies in
    triangle_index = tri.find_simplex(window_points)

    barycentric_coefficients = np.zeros((window_points.shape[0], 3))

    #tri.simplices is a list of triangles, where each entry contains a list of three indices to the three points that make up the triangle.
    triangle_indices = tri.simplices.shape[0]

    """Computes the barycentric coefficients for each of the window points. This is done by iterating through each triangle and computing barycentric coordinates for all points,
    then removing ones with negative coefficients (i.e. points that are not in the particular trianngle."""
    for i in range(triangle_indices):
        b = tri.transform[i,:2].dot(np.transpose(window_points - tri.transform[i,2]))
        coords = np.c_[np.transpose(b), 1 - b.sum(axis=0)]
        barycentric_coefficients[np.where(triangle_index == i)] = coords[np.where(triangle_index == i)]
    
    """Instead of each triangle being composed of three indices, this converts them into a list of three actual facade points."""
    facade_simplices = np.take(facade, tri.simplices, axis=0)

    absolute_window_coords = np.zeros((window_points.shape[0], 3))

    #Computes the 3D coordinates for each of the window points using the barycentric coordinates
    for i in range(absolute_window_coords.shape[0]):
        b_coeff = barycentric_coefficients[i,:]
        tri_index = triangle_index[i]
        facade_points = facade_simplices[tri_index]
        for j in range(facade_points.shape[0]):
            absolute_window_coords[i] += facade_points[j] * b_coeff[j]
    
    return absolute_window_coords    

"""Calculates the window to wall ratio for a facade. Requires the projected facade, the facade polygon in 3D, and a list of the window polygons contained in the facade, with points
in 3D. """
def get_window_wall_ratio(projected_facade, facade, windows):
    # facade_points = get_shapely_polygon_points(facade)
    facade_area = get_poly_area_from_points(facade)
    print(print("facade_area: " + str(facade_area)))
    window_area = 0
    for i in range(len(windows)):
        absolute_window_coords = get_window_coordinates(projected_facade, facade, windows[i])
        window_area += get_poly_area_from_points(absolute_window_coords)
    print(print("window_area: " + str(window_area)))

    if window_area > 0 and facade_area > 0:
        return window_area / facade_area
    return 0

def main():
    #Sets the input and output directories
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="data/Drone_Flight_Samir/")
    parser.add_argument('--facade_file', type=str, default="data/Drone_Flight_Samir/merged.txt")
    parser.add_argument('--skip', type=int, default=10)

    #Move the window predictions into this folder.
    parser.add_argument('--window_predict_dir', type=str, default="data/predictions/")

    #Filenames 
    parser.add_argument('--file_ext', type=str, nargs='+')
    args = parser.parse_args()
    directory = args.dir
    facade_file = args.facade_file
    window_dir = args.window_predict_dir

    if directory[-1] != "/":
        directory += "/"
    image_dir = directory + "RGB/"
    param_dir = directory + "params/"

    offset = np.loadtxt(param_dir + "offset.txt",usecols=range(3))
    polygon_offset = np.loadtxt(directory + "polygon_offset.txt", delimiter=",")

    #Initializes a dictionary of Camera classes. See utils.py for more information.
    camera_dict = utils.create_camera_dict(param_dir, offset)

    #Loads image filenamees
    filenames = np.genfromtxt(param_dir + 'pmatrix.txt', usecols=range(1), dtype=str)

    #Loads the merged polygons, as well as a list of facade types (i.e. roof, wall, or floor)
    merged_polygons, facade_type_list, file_format = fw.load_merged_polygon_facades(filename=facade_file)

    #Offset adjustment parameter
    height_adj = -polygon_offset[2]
    offset_adj = np.array([0.0, 0.0, height_adj])
    offset = offset + offset_adj

    #Loads all files in the param directory with a specified extension
    ext = args.file_ext
    filenames = [f for f in filenames if f.split('.')[1] in ext]

    #Create a list of indices of filenames, skipping indices by skip_amount
    skip_amount = args.skip
    file_indices = [i for i in range(0, len(filenames), skip_amount)]

    #Iterate through the file indices, and compute the window to wall ratio for each facade in the image
    for i in file_indices:
        filename = filenames[i]
        camera = camera_dict[filename]
        pmatrix = camera.calc_pmatrix()

        #TODO: Adjust this to whatever format the predictions take
        window_file = window_dir + filename.split('.')[0] + "_pred" + filename.split('.')[1]
        
        #Extract the contours of the window file
        contours = ce.extract_contours(window_file)

        #Create polygons from the window contours
        window_polygons = utils.convert_polygons_shapely(contours)

        #Projects the merged polygon facades onto the camera image
        projected_facades, projective_distances = project_merged_polygons(merged_polygons, offset, pmatrix)

        #Creates a dictionary mapping the facade to the windows contained within them, keyed by facade index
        facade_window_map = get_facade_window_map(window_polygons, projected_facades, projective_distances)
    
        #Creates a list of all the facades in the merged polygon
        facades = []
        for poly in merged_polygons:
            facades = facades + poly

        facade_indices = list(facade_window_map.keys())
        for i in facade_indices:
            #Computes window to wall ratio
            win_wall_ratio = get_window_wall_ratio(projected_facades[i], facades[i], facade_window_map[i])
            
            #Output printing:
            print("Facade index: " + facades[i])
            print("Window-to-wall ratio: " + win_wall_ratio)

            #Display the windows and facades on the image
            image_file = utils.load_image(image_dir + filename)
            plot_windows_facade(projected_facades[i], facade_window_map[i], image_file)

if __name__ == "__main__":
    main()