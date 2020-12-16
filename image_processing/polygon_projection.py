from matplotlib.path import Path
import numpy as np
import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polygon_primitives.file_writer as fw
import image_processing.camera_processing as cp
import image_processing.utils as utils
from PIL import Image

"""Returns a mask of the pixels in the image that correspond to the projected facade."""
def get_facade_mask(im_shape, projected_facade):
    image_height, image_width = im_shape[:2]
    y, x = np.indices((image_height, image_width)) # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T 

    p = Path(projected_facade) # make a polygon
    grid = p.contains_points(points)
    mask = grid.reshape(image_height,image_width) # now you have a mask with points inside a polygon
    return mask

"""Projects a given set of points using the pmatrix and offset."""
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

"""Projects a list of facades (where each facade is a list of points), and returns a list of projected points."""
def get_projected_facades(facades, pmatrix, offset=[0.0, 0.0, 0.0]):
    projected_facades = []
    for facade in facades:
        projected_points = project_points(facade, pmatrix, offset)
        projected_facades.append(projected_points)
    return projected_facades    

"""Creates a mask for the given polygon."""
def get_polygon_mask(im_shape, polygon):
    poly_mask = np.zeros(im_shape[:2], dtype=np.uint8)
    for i in range(len(polygon)):
        facade = polygon[i]
        mask = get_facade_mask(im_shape, facade)
        poly_mask[np.where(mask)] = 1
    return poly_mask

"""Creates a mask for the given polygon, split into roof_mask and facades_mask, depending on the facade type ptype."""
def get_polygon_facades_roof_mask(im_shape, polygon, ptype):
    facades_mask = np.zeros(im_shape[:2], dtype=np.uint8)
    roof_mask = np.zeros(im_shape[:2], dtype=np.uint8)

    for i in range(len(polygon)):
        facade = polygon[i]
        facade_type = ptype[i]
        if facade_type != 1:
            mask = get_facade_mask(im_shape, facade)

            #Facade type == 0 corresponds to a wall, facade type == 1 corresponds to a roof
            if facade_type == 0:
                facades_mask[np.where(mask)] = 1
            else:
                roof_mask[np.where(mask)] = 1
    return facades_mask, roof_mask

"""Crops the image by assigning all points outside of the given facade an RGBA color of (0.0, 0.0, 0.0, 0.0)."""
def crop_image_outside_facade(image, facade, offset, pmatrix):
    cropped_image = np.array(image)

    crop_color = (0.0, 0.0, 0.0, 0.0)
    im_shape = np.array(image).shape
    projected_facade = project_points(facade, pmatrix, offset)
    mask = get_facade_mask(im_shape, projected_facade, 0)

    cropped_image[np.where(np.logical_not(mask))] = crop_color
    cropped_image = Image.fromarray(cropped_image, 'RGBA')
    return cropped_image

"""Crops the image by assigning all points that do not lie within the merged polygons an RGBA color of (0.0, 0.0, 0.0, 0.0)."""
def crop_image_outside_building(image, merged_polygons, offset, pmatrix):
    crop_color = (0.0, 0.0, 0.0, 0.0)
    im_shape = np.array(image).shape
    cropped_image = np.array(image)
    mask = np.zeros(im_shape[:2], dtype=np.uint8)

    for i in range(len(merged_polygons)):
        polygon = merged_polygons[i]
        projected_facades = get_projected_facades(polygon, pmatrix, offset=offset)
        poly_mask = get_polygon_mask(im_shape, projected_facades)
        mask[np.where(poly_mask)] = 1

    cropped_image[np.where(np.logical_not(mask))] = crop_color
    cropped_image = Image.fromarray(cropped_image, 'RGBA')
    return cropped_image

"""Iterates through the list of merged polygons and projects them onto the image. The wall and roof colors can optionally be specified. Note that the
alpha channel is ignored."""
def mark_image_file(image, merged_polygons, facade_type_list, offset, pmatrix, wall_color=(255, 50, 50, 255), roof_color = (50, 50, 255, 255)):
    # marked_image = image.copy()
    im_shape = np.array(image).shape
    facades_image = np.zeros(im_shape, dtype=np.uint8)
    facades_mask = np.zeros(im_shape[:2], dtype=np.uint8)
    roof_mask = np.zeros(im_shape[:2], dtype=np.uint8)

    for i in range(len(merged_polygons)):
        polygon = merged_polygons[i]
        projected_facades = get_projected_facades(polygon, pmatrix, offset=offset)
        poly_facade_mask, poly_roof_mask = get_polygon_facades_roof_mask(im_shape, projected_facades, facade_type_list[i])
        facades_mask[np.where(poly_facade_mask)] = 1
        roof_mask[np.where(poly_roof_mask)] = 1
    
    facades_image[np.where(facades_mask)] = wall_color
    facades_alpha_mask = np.pad(np.expand_dims(facades_mask, axis=2), ((0,0), (0,0), (3, 0)))
    facades_image[np.where(facades_alpha_mask)] = np.ones(facades_mask[facades_mask > 0].shape) * 100

    facades_image[np.where(roof_mask)] = roof_color
    roof_alpha_mask = np.pad(np.expand_dims(roof_mask, axis=2), ((0,0), (0,0), (3, 0)))
    facades_image[np.where(roof_alpha_mask)] = np.ones(roof_mask[roof_mask > 0].shape) * 100

    facades_image = Image.fromarray(facades_image, 'RGBA')
    image = Image.alpha_composite(image, facades_image)
    return image


def create_camera_dict(param_dir, filename, offset): #(param_dir, offset):
    #file_name = param_dir + "calibrated_camera_parameters.txt"
    file_name = param_dir + filename
    cameras = cp.initialize_cameras(file_name, offset)
    camera_dict = dict()
    for camera in cameras:
        camera_dict[camera.file_name] = camera
    return camera_dict

"""Example code for projecting the merged polygons onto a list of thermal images and coloring the facades based on their type (i.e. roof or wall)."""
def mark_thermal_image():
    #Arg parser has been placed here for illustration purposes
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="data/Drone_Flight/")
    parser.add_argument('--facade_file', type=str, default="data/Drone_Flight/merged_thermal.txt")
    parser.add_argument('--out', type=str, default='data/Drone_Flight/output_thermal/')
    parser.add_argument('--height_adj', type=float, default=0.0, help="Offset applied to the height of the camera images. Used to adjust for camera image with erroneous sea level data.")
    parser.add_argument('--skip_amount', type=int, default=10, help="Starting at the first camera image, number of subsequent camera images to skip.")
    parser.add_argument('--image_path', type=str, help="Image file to process. Will only process this image if a filename is provided.")

    args = parser.parse_args()
    directory = args.dir
    out_dir = args.out
    facade_file = args.facade_file

    #Load the relevant files, and assign directory paths
    image_dir = directory + "Thermal/"
    param_dir = directory + "params_thermal/"
    offset = np.loadtxt(param_dir + "offset.txt",usecols=range(3))
    p_matrices = np.loadtxt(param_dir + 'pmatrix.txt', usecols=range(1,13))
    filenames = np.genfromtxt(param_dir + 'pmatrix.txt', usecols=range(1), dtype=str)
    merged_polygons, facade_type_list, file_format = fw.load_merged_polygon_facades(filename=facade_file)

    #Create a dictionary mapping the camera filename to a Camera object
    camera_dict = create_camera_dict(param_dir, offset)
    
    #Adjust height if necessary for the camera images
    height_adj = args.height_adj
    offset_adj = np.array([0.0, 0.0, height_adj])
    offset = offset + offset_adj

    filenames = [f for f in filenames if f.split('.')[1] == 'tif']
    
    #If an image file path has been provided, use it instead of iterating through a list of filenames
    image_filename = args.image_path
    if image_filename is not None and len(image_filename) > 0:
        filenames = [image_filename]
        file_indices = [0]
    else:
        #Select from the list of filenames, skipping every skip_amount of images.
        skip_amount = args.skip_amount
        file_indices = [i for i in range(0, len(filenames), skip_amount)]

    #Project the merged polygons onto the image for each file
    for i in file_indices:
        filename = filenames[i]
        camera = camera_dict[filename]
        pmatrix = camera.calc_pmatrix()

        #Necessary to replace the .tif extension used by Pix4D in the pmatrix file
        filename = filename.split('.')[0] + '.jpg'

        image = utils.load_image(image_dir + filename)
        out_image = mark_image_file(image, merged_polygons, facade_type_list, offset, pmatrix)
        utils.save_image(out_image, out_dir, filename)

"""Example code for projecting the merged polygons onto a list of RGB images and coloring the facades based on their type (i.e. roof or wall)."""
def mark_rgb_image():
    #Arg parser has been placed here for illustration purposes
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="data/Drone_Flight/", help='Parent directory for drone data and Pix4D output.')
    parser.add_argument('--facade_file', type=str, default="data/Drone_Flight/merged.txt", help="Filepath for the merged polygons from the building footprint.")
    parser.add_argument('--out', type=str, default='data/Drone_Flight/output/', help="Output directory.")
    parser.add_argument('--height_adj', type=float, default=0.0, help="Offset applied to the height of the camera images. Used to adjust for camera image with erroneous sea level data.")
    parser.add_argument('--skip_amount', type=int, default=10, help="Starting at the first camera image, number of subsequent camera images to skip.")
    parser.add_argument('--image_path', type=str, help="Image file to process. Will only process this image if a filename is provided.")

    args = parser.parse_args()
    directory = args.dir
    out_dir = args.out
    facade_file = args.facade_file

    #Load the relevant files, and assign directory paths
    image_dir = directory + "RGB/"
    param_dir = directory + "params/"
    offset = np.loadtxt(param_dir + "offset.txt",usecols=range(3))
    p_matrices = np.loadtxt(param_dir + 'pmatrix.txt', usecols=range(1,13))
    filenames = np.genfromtxt(param_dir + 'pmatrix.txt', usecols=range(1), dtype=str)
    polygon_offset = np.loadtxt(directory + "polygon_offset.txt", delimiter=",")
    merged_polygons, facade_type_list, file_format = fw.load_merged_polygon_facades(filename=facade_file)

    #Create a dictionary mapping the camera filename to a Camera object
    camera_dict = create_camera_dict(param_dir, offset)
    
    #Adjust height if necessary for the camera images
    height_adj = -polygon_offset[2]
    offset_adj = np.array([0.0, 0.0, height_adj])
    offset = offset + offset_adj

    #Load the filenames
    filenames = [f for f in filenames if f.split('.')[1] != 'tif']

    #If an image file path has been provided, use it instead of iterating through a list of filenames
    image_filename = args.image_path
    if image_filename is not None and len(image_filename) > 0:
        filenames = [image_filename]
        file_indices = [0]
    else:
        #Select from the list of filenames, skipping every skip_amount of images.
        skip_amount = args.skip_amount
        file_indices = [i for i in range(0, len(filenames), skip_amount)]

    #Project the merged polygons onto the image for each file
    for i in file_indices:
        filename = filenames[i]
        camera = camera_dict[filename]
        pmatrix = camera.calc_pmatrix()

        image = utils.load_image(image_dir + filename)
        out_image = mark_image_file(image, merged_polygons, facade_type_list, offset, pmatrix)
        utils.save_image(out_image, out_dir, filename)

def main():
    mark_rgb_image()

    # Uncomment to run the example code with thermal images
    # mark_thermal_image()

if __name__ == "__main__":
    main()