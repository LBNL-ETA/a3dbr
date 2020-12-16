import math
import numpy as np
from skimage.transform import (hough_line, hough_line_peaks)
from scipy import ndimage
from scipy.signal import medfilt
from sklearn import linear_model
import matplotlib.pyplot as plt
import laspy

#Get the 95th percentile of the heights of the points in the cell
def get_heights(cells):
    heights = []
    if len(cells) == 0:
        heights.append(0.0)
    for cell in cells:
        if cell.num_points < 2:
            continue

        heights.append(cell.percentile_95)
    return heights
    
#Calculates the density from a list (the cells) of a list of points (the points within the cells)
def get_density(point_lists, width=1.0):
    #Note: This could also be changed to median
    avg_density = 0.0
    for points in point_lists: 
        if points.size > 0:
            min_x, min_y, min_z = np.amin(points, axis=0)
            max_x, max_y, max_z = np.amax(points, axis=0)


            dist_x = max_x - min_x
            dist_y = max_y - min_y

            length = math.sqrt(dist_x ** 2 + dist_y ** 2)
            dist_z = max_z - min_z

            density = points.size / (length * width * dist_z)
            avg_density += density / len(point_lists)
    return avg_density

#Centers the points by subtracting their mean along X, Y and Z directions
def subtract_point_min(points):
    min_x, min_y, min_z = np.amin(points, axis=0)
    ground_z = np.percentile(points, 5, axis=0)[2]
    offset = np.array([min_x, min_y, ground_z])
    return points - offset, offset

#Gets the peaks in Hough Space
def get_hough_peaks(h, theta, d):
    accum, angles, dists = hough_line_peaks(h, theta, d)
    return angles, dists

#Performs the hough transform on an image, returns the image, the transformed image, the hough line angles and the hough line distances (d)
def hough_transform(img):
    out, angles, d = hough_line(img)
    return img, out, angles, d

#Rotates a set of points around a particular point
def rotate_about_point(points, theta, rot_point=[0.0, 0.0, 0.0]):
    translated_points = np.subtract(points, rot_point)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, s], [-1.0 * s, c]])
    rotated_points = np.dot(translated_points, rot)
    return np.add(rotated_points, rot_point)

#Rotates a line about a particular point
def rotate_line_about_point(points, theta, rot_point=[0.0, 0.0, 0.0]):
    if len(points) == 0:
        return []
    translated_points = np.subtract(points, rot_point)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, s, 0.0], [-1.0 * s, c, 0.0], [0.0, 0.0, 1.0]])
    rotated_points = np.dot(translated_points, rot)
    return np.add(rotated_points, rot_point)

#Performs a median filter on an image (1D)
def median_filter_image(image, med_filter_size):
    out = ndimage.filters.median_filter(image, size=(med_filter_size,1))
    return out

#Binarizes an image with a given threshold
def threshold_image(image, thresh=30):
    below_threshold_indices = image < thresh
    image[below_threshold_indices] = 0

    above_threshold_indices = image >= thresh
    image[above_threshold_indices] = 255
    return image

#Normalizes a vector v
def normalize(v):
    norm = np.linalg.norm(v)

    if norm == 0: 
       return v

    return v / norm

#Checks whether the direction between two vectors is approximately equal
def compare_vector(v1, v2):
    normv1 = normalize(v1)
    normv2 = normalize(v2)

    if min(normv1 - normv2, normv1 + normv2) < 0.01:
        return True
    return False

#Gets the Euclidean distance between two points
def distance(point1, point2):
    return np.linalg.norm(point1 - point2)

#Gets the angle between two vectors
def angle_between(v1, v2):
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#Applies the Ransac method to a set of points, and returns the line corresponding to it
def apply_ransac(points):
    X, y = np.split(points,[-1],axis=1)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    #For vertical lines, X values can be infinitesimal, so we should use the Y values
    #for the line equation instead
    if max_y - min_y > max_x - min_x:
        reverse = True
        X, y = y, X
    else:
        reverse = False

    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    line_X = np.array([X.min(), X.max()]).reshape((2, 1))
    line_y_ransac = ransac.predict(line_X)
    
    if reverse:
        return np.array([line_y_ransac[0][0], line_X[0][0]]), np.array([line_y_ransac[-1][0], line_X[-1][0]])
    return np.array([line_X[0][0], line_y_ransac[0][0]]), np.array([line_X[-1][0], line_y_ransac[-1][0]])

#Checks if too points are within a distance epsilon from one another
def compare_points(point1, point2, epsilon=0.8):
    if distance(point1, point2) < epsilon:
        return True
    return False

#Loads points from a preprocessed .txt file
def get_points_from_file(filename):
    points = np.loadtxt(filename, delimiter=",")
    return points

