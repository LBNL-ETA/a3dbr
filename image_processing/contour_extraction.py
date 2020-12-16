import cv2
import numpy as np
from shapely import geometry
import image_processing.utils
import matplotlib.pyplot as plt

"""Loads a list of contours from a file."""
def load_contours(contour_file):
    with open(contour_file, 'r') as f:
        lines = f.readlines()
        num_contours = int(lines[0].strip('\n'))
        contours = []
        curr_contour = []
        contour_length = None
        counter = 0
        for line in lines[1:]:
            line = line.strip('\n')
            if contour_length is None:
                contour_length = int(line)
                counter = 0
                if len(curr_contour) > 0:
                    contours.append(curr_contour)
                curr_contour = []
            else:
                pixel_index = [int(i) for i in line.split()]
                curr_contour.append(pixel_index)
                counter += 1
                if counter == contour_length:
                    contour_length = None
    return contours

"""Computes a list of contours and writes to a file."""
def extract_contours_to_file(image_file, out_filename):
    image = cv2.imread(image_file)
    im_bw = 255 - cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (thresh, binarized_im) = cv2.threshold(im_bw, 40.0, 255.0, 0)
    
    contours, hierarchy = cv2.findContours(binarized_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    with open(out_filename, 'w') as filehandle:
        filehandle.write('%s\n' % len(contours))
        for contour in contours:
            filehandle.write('%s\n' % len(contour))
            for pixel in contour:
                filehandle.write('%s %s\n' % (pixel[0][0], pixel[0][1]))

    simplified_contours = []
    for contour in contours:
        simplified_contours.append(simplify(contour, 0.1))
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
    cv2.imshow('Contours', image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

"""Extracts a set of contours from an image file, as a list."""
def extract_contours(image_file):
    image = cv2.imread(image_file)
    im_bw = 255 - cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (thresh, binarized_im) = cv2.threshold(im_bw, 40.0, 255.0, 0)
    contours, hierarchy = cv2.findContours(binarized_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_list = [[[pixel[0][0], pixel[0][1]] for pixel in contour] for contour in contours]
    return contour_list
