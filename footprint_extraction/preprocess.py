import sys
from os import listdir, mkdir
from os.path import isfile, join, exists
import numpy as np
import argparse
import laspy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="LAS", help="Path to the LAS point cloud file.")
    parser.add_argument('--data_folder', type=str, default="Data", help="Directory for output.")
    parser.add_argument('--out_points', type=int, default=5000000, help="Number of output points.")
    args = parser.parse_args()
    path = args.path
    out_dir = args.data_folder

    if out_dir[-1] != "/":
        out_dir = out_dir + "/"

    num_out_points = args.out_points

    if not exists(out_dir):
        mkdir(out_dir)
    
    # Gets all .LAS files in the directory
    point_cloud_files = [f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[1] == "las"]
    if len(point_cloud_files) == 0:
        print("Could not find .las files at the location: " + path)
        
    # Loads the .LAS point clouds and merges them together
    loaded_points = None
    for f in point_cloud_files:
        points = load_las_points(path + "/" + f)
        if loaded_points is None:
            loaded_points = points
        else:
            loaded_points = np.concatenate((loaded_points, points), axis=0)

    # Subsamples the point cloud randomly, so that the size is at most num_out_points
    if loaded_points.shape[0] > num_out_points:
        np.random.shuffle(loaded_points)
        loaded_points = loaded_points[:num_out_points]
    
    np.savetxt(out_dir + 'points.txt', loaded_points, delimiter=',')

#Loads points from a .las file
def load_las_points(filename):
    f = laspy.file.File(filename, mode = "r")
    points = np.vstack((f.x, f.y, f.z)).transpose()
    return points

if __name__ == "__main__":
    main()
