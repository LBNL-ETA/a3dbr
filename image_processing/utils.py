import numpy as np
from shapely import geometry
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import image_processing.camera_processing as cp
from PIL import Image

"""Loads an image using Pillow. Will return an RGBA version of the image."""
def load_image(filename):
    im = Image.open(filename)
    im = im.convert('RGBA')
    return im

"""Saves an image to the specified output directory, with a specified filename. Will add a .png extension regardless of the given extension."""
def save_image(image, out_dir, filename):
    print("Saving image...")
    out_filename = filename.split('.')[0] + ".png"
    image.save(out_dir + out_filename)

"""Checks if a given point is located within a given polygon."""
def point_within_polygon(point, polygon):
    return polygon.contains(point)

"""Calculates the average projective distance of a facade. Used to compare the distance of facades to the camera."""
def compute_average_projective_distance(points, pmatrix, offset):
    points = points - offset
    R = pmatrix[:, :3]
    t = pmatrix[:, 3]

    #Compute the location of the camera in world coordinates
    camera_loc = np.append(-np.dot(np.linalg.pinv(R), t), 1)[:3]

    #Compute the projection distance of the points
    projected_distances = np.linalg.norm(points - camera_loc, axis=1)
    return np.mean(projected_distances)


"""Projects a set of points onto a camera image."""
def project_points(points, pmatrix, offset):
    pmatrix = pmatrix.reshape((3,4))
    pmatrix = np.vstack((pmatrix, [0,0,0,1]))
    ones = np.ones((len(points), 1))
    centered_points = np.array(points).reshape(-1, 3) - np.array(offset)
    point_list = np.hstack((centered_points, ones))

    projected_points = np.transpose(np.matmul(pmatrix, np.transpose(point_list)))
    z = projected_points[:,2]
    projected_points = projected_points / z[:, None]
    projected_points = np.rint(projected_points).astype(int)[:,:2]
    return projected_points

"""Projects a set of facades onto a camera image."""
def get_projected_facades(facades, pmatrix, offset=[0.0, 0.0, 0.0]):
    projected_facades = []
    projective_distances = []
    for facade in facades:
        projected_points = project_points(facade, pmatrix, offset)
        projective_distance = compute_average_projective_distance(facade, pmatrix, offset)
        projected_facades.append(projected_points)
        projective_distances.append(projective_distance)
    return projected_facades, projective_distances

"""Creates a list of shapely polygons from a list of lists of points."""
def convert_polygons_shapely(polygons):
    shapely_polys = []
    for polygon in polygons:
        shp_polygon = geometry.Polygon([p for p in polygon])
        shapely_polys.append(shp_polygon)
    return shapely_polys

"""Creates a dictionary mapping filenames to Camera classes (see camera_processing.py)"""
def create_camera_dict(param_dir, filename="calibrated_camera_parameters.txt", offset=[0.0, 0.0, 0.0]):
    file_path = param_dir + filename
    cameras = cp.initialize_cameras(file_path, offset)
    camera_dict = dict()
    for camera in cameras:
        camera_dict[camera.file_name] = camera
    return camera_dict