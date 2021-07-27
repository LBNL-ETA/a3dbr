# Aerial 3D Building Reconstruction (A3DBR)

***

Table of Contents:
----
+ [**Introduction**](#introduction)
+ [**Installation**](#installation)
+ [**Preprocessing**](#Preprocessing)
+ [**Line Detection**](#Line-Detection)
+ [**Polygonization**](#Polygonization)
+ [**Polygon Merging**](#Polygon-Merging)
+ [**Window-to-Wall Ratio**](#Window-Wall-Ratio)


## Introduction

Aerial 3D Building Reconstruction (A3DBR) is a building footprint extraction pipeline, that uses aerial drone images to construct a building footprint with height information, which can be used to create a 3D model of the building. This is the repository for the paper of the same name, which can be found at:
```bash
https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11398/1139803/Aerial-3D-building-reconstruction-from-RGB-drone-imagery/10.1117/12.2558399.short?tab=ArticleLink&SSO=1
```
Pix4D is first used to create a point cloud from a set of drone images. Thereafter, the pipeline consists of 3 major steps:
- Line Extraction:
    - Points are projected onto a 2D grid. Points in each grid cell are counted. 
    - The hough line transform is used to extract lines.
    - Line fit is improved with RANSAC.
    - Line segments are extracted by finding sections of support in the point cloud.

- Polygonization:
    - Line segments are post-processed, and are joined together to form 'edges'.
    - 'Partial' polygons are constructed from the edges.
    - Partial polygons are converted into full, 'completed' polygons by adding additional edges.
    - Completed polygons are filtered out based on the height profile of their internal points.
    - The additional edges lead to new partial polygons.
    - The process is repeated until no new completed polygons are formed.

- Polygon Merging:
    - Neighboring polygons with a similar height profile are merged together.
    - Height is determined using the points internal to the merged polygon.

Note that A3DBR is currently only designed for polygonal buildings with flat rooftops.

We provide example notebooks for all methods in this section as part of the "Example_Notebooks/" directory.

This project was developed primariy using the output from Pix4D. At the time of writing, the relevant files are located in the following folders:
Pix4D project folder -> 1_initial -> params
Pix4D project folder -> 2_densification -> point_cloud

## Installation

From sources:
```bash
git clone https://github.com/avideh/a3dbr.git
cd a3dbr
python setup.py install
```
## Preprocessing
The (possibly several) point cloud(s) extracted from Pix4D should be placed in the 'LAS' folder, but can be specified using the 'path' argument. To merge the separate .las files that Pix4D outputs, run from the top-level folder:

```bash
python footprint_extraction/preprocess.py --path [PATH_TO_LAS_FILE]
```

Note that the point cloud produced by Pix4D generally results in a point cloud that is more dense than necessary for our pipeline, which slows down the pipeline. The above script will subsample the point cloud so that it has no more than 5 million points. You can optionally specify the number of output points using:

```bash
python footprint_extraction/preprocess.py --path [PATH_TO_LAS_FILE] --out_points [NUM_POINTS]
```

## Line Detection
To extract the footprint, we first leverage the commercial software Pix4D to construct a 3D point cloud from RGB drone imagery.  For each point cloud passed as input to the Line Detection module, a set of line segments that correspond to building facades, represented as 2D UTM coordinates, will be constructed. For this stage, two parameters are provided as input, the minimum wall height and minimum wall length that the user wishes to detect, which in our experiments was set to 2m each.

To detect line segments, we project the point cloud onto a regular grid, detect preliminary lines using the Hough transform, refine them via RANSAC, and convert them into line segments by checking the density of the points surrounding the line. 

From the top level directory, run:

```bash
python footprint_extraction/extract_lines.py --data_folder [DATA_FOLDER]
```

## Polygonization
In this stage, we construct polygons from the line segments extracted in the Line Detection stage. As some walls may have been missed by the Hough transform or may be absent from the point cloud itself, this step involves repeatedly completing polygons using geometric rules. The geometric rules depend on the number of corners of the polygon, as well as the number of endpoints that are only connected by a single edge.

To run the polygonization process, run:

```bash
python footprint_extraction/generate_polygons.py --data_folder [DATA_FOLDER]
```

## Polygon Merging
Finally, the polygons are merged to create the building footprint. 

To compute the average height of the merged polygons, we compute the average height of the points in each cell within the polygon, and then average the heights of the cells.

Once this pipeline has completed, the user will have a `Data` folder structured as follows:

```
Data
└───merged_polygons.txt
└───line_segments.txt
└───density.txt
└───processed.las
```

## Window-To-Wall Ratio
With the merged polygons, as well as the window pixel predictions we can then compute the window-to-wall ratio. This proceeds as follows. For each camera image, we load the image file and the corresponding window predictions. Then, from the predictions file, we extract the contours. Next, we project all the merged polygons onto the camera image, one facade at a time. These facades are then triangulated. If a window prediction contour lies within a facade, we determine barycentric coefficients for each of the contour points, using the triangulated facade points. Note that a window may lie within many facades in 2D, but in that case one facade will be occluding the others, and we can determine the closest such facade by choosing the one with the minimum projective distance.

Once we have found the barycentric coordinates for the window prediction contours, we use the corresponding barycentric coefficients for the facade points in 3D to interpolate a corresponding 3D point for each of the 2D countour points.

In order to run this section of the pipeline, a data folder structured as follows is required:

```
Data
└───merged_polygons.txt
└───Drone_Flight/
└───└───predictions/
└───└───RGB/
└───└───params/

```
The params directory must include an 'offset.txt' file (Pix4D provides a .xyz file, but the extension can simply be changed to .txt), and a 'calibrated_camera_parameters.txt' file. In Pix4D, these can be found in:
Pix4D project folder -> 1_initial -> params -> offset.xyz, (project_name)_calibrated_camera_parameters.txt

To print the window-to-wall ratio for a group of images, simply run:
```bash
python image_processing/extract_window_wall_ratio.py --dir [DATA_FOLDER] --merged_polygon_file [MERGED_POLYGONS_FILE] --skip [STEP_SIZE_BETWEEN_IMAGES] --window_predict_dir [WINDOW_PREDICTIONS_DATA_FOLDER] 
```
Note that this expects the window prediction filenames to be structured as:
```
[IMAGE_FILENAME]_pred.png
```

## Copyright

Aerial 3D Building Reconstruction from Drone Imagery (A3DBR) 
Copyright (c) 2021, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy) and Signetron.  
All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.

## License

Aerial 3D Building Reconstruction from Drone Imagery (A3DBR) is available under the following [license](https://github.com/LBNL-ETA/a3dbr/blob/master/LICENSE.txt).
