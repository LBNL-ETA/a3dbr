import numpy as np
import scipy.linalg as lin
import sys
from shapely.geometry import Polygon
import line_extraction_primitives.extraction_helper_methods as hm

"""Ray class definition. Can be used to test intersections with a facade."""
class Ray:
	def __init__(self, start, direction):
		self.direction = direction
		self.start = start

	#Facade defined as an edge with a specified height.
	def get_intersection(self, edge, height):
		intersection_point = edge.get_euclidean_intersection_with_ray(self.start, self.direction)
		t = (intersection_point[0] - self.start[0]) / self.direction[0]
		z = self.start[2] + t * self.direction[2]
		if intersection_point is not None and edge.on_line(intersection_point) and z < height:
			return intersection_point, t
		return None, None

"""Definition for the camera classes. Stores the cameras relevant intrinsic/extrinsic parameters."""
class Camera:
	def __init__(self, file_name, image_w, image_h, k_matrix, rad_distortion, tan_distortion, camera_pos, camera_rot):
		self.file_name = file_name
		self.image_w = image_w
		self.image_h = image_h
		self.K = k_matrix
		self.rad_distor = rad_distortion
		self.tan_distor = tan_distortion
		self.camera_pos = camera_pos
		self.camera_rot = camera_rot
		self.window_polygons = []
		self.windows = []

	def __init__(self):
		self.window_polygons = []
		self.windows = []

	def set_offset(self, offset):
		self.offset = offset

	def set_world_rotation(self, omega, phi, kappa):
		self.R_omega = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(omega), -np.sin(omega)], [0.0, np.sin(omega), np.cos(omega)]])
		self.R_phi = np.array([[np.cos(phi), 0.0, np.sin(phi)], [0.0, 1.0, 0.0], [-np.sin(phi), 0.0, np.cos(phi)]])
		self.R_kappa = np.array([[np.cos(kappa), -np.sin(kappa), 0.0], [np.sin(kappa), np.cos(kappa), 0.0], [0.0, 0.0, 1.0]])


	def set_file_name(self, filename):
		self.file_name = filename

	def set_im_size(self, image_w, image_h):
		self.image_w = image_w
		self.image_h = image_h

	def set_k_matrix(self, k_matrix):
		self.K = k_matrix

	"""Translates a point from world space to the coordinate from of the camera image."""
	def world_to_image_coord(self, point):
		P = self.calc_pmatrix()
		P = np.vstack((P,[0,0,0,1]))

		point = point - self.offset
		point = np.append(point, 1.0)
		point = point.reshape((4,1))

		pixel_loc = np.matmul(P, point)
		pixel_loc = pixel_loc / pixel_loc[2]
		b = pixel_loc.reshape((4)).astype(int)
		return b[:2]


	def set_rad_distortion(self, rad_distor):
		self.rad_distor = rad_distor

	def set_tan_distor(self, tan_distor):
		self.tan_distor = tan_distor

	def set_camera_pos(self, camera_pos):
		self.camera_pos = camera_pos

	def set_camera_rot(self, camera_rot):
		self.camera_rot = camera_rot

	def add_window_polygon(self, polygon):
		self.window_polygons.append(polygon)

	def calc_pmatrix(self):
		R = self.camera_rot
		t = self.camera_pos.reshape((3, 1))
		pose_mat = np.hstack((R,-np.matmul(R, t)))
		P = np.matmul(self.K, pose_mat)
		return P

	"""Computes a ray passing through a pixel location in the image."""
	def calc_ray(self, im_point, offset):

		im_point = np.array(im_point)
		im_point = np.append(im_point, [1.0, 0.0])
		im_point = im_point.reshape((4,1))

		P = self.calc_pmatrix()
		P = np.vstack((P,[0,0,0,1]))

		ray_dir = np.dot(lin.pinv(P), im_point)
		ray_dir = ray_dir.reshape((4))[:3]

		world_pos = self.offset + self.camera_pos
		ray = Ray(world_pos, ray_dir)
		return ray

	def add_window(self, corners):
		self.windows.append(corners)

"""Helper method to load the cameras from a file, with a specified offset. See load_cameras_from_file."""
def initialize_cameras(file_name, offset):
	cameras = load_cameras_from_file(file_name)
	for camera in cameras:
		camera.set_offset(offset)
	return cameras

"""Helper method for loading camera extrincs from a file"""
def load_cameras_extrinsics_from_file(file_name):
	with open(file_name, "r") as f:
		lines = f.readlines()[1:]
		cameras = []
		for line in lines:
			elements = line.split()
			if len(elements) >= 7:
				cam = Camera()
				im_file = elements[0]
				x, y, z = elements[1:4]
				omega, phi, kappa = [float(elem) for elem in elements[4:8]]
				
				cam.set_file_name(im_file)
				pos = np.array([x,y,z]).astype(float)
				cam.set_world_position(pos)
				cam.set_world_rotation(omega, phi, kappa)
				cameras.append(cam)
	return cameras

"""Loads the camera parameters from a file, and creates a list. Camera parameter file can be found in the Pix4D folder -> 1_initial -> params -> calibrated_camera_paramters.txt"""
def load_cameras_from_file(file_name):
	cameras = []
	with open(file_name, "r") as f:
		lines = f.readlines()[8:]

		#Each camera takes up 10 lines in the parameter file
		num_cameras = len(lines) // 10
		for cam_index in range(num_cameras):
			i = cam_index * 10
			cam = Camera()

			for j in range(10):
				#Image filename, width and height
				if j == 0:
					line = lines[i + j]
					txt = line.split()
					im_file = txt[0]
					im_w = float(txt[1])
					im_h = float(txt[2])

					cam.set_file_name(im_file)
					cam.set_im_size(int(im_w), int(im_h))
					cameras.append(cam)

				#K matrix
				elif j == 1:
					k_mat_lines = lines[i + 1: i + 4]
					k_mat = []
					for l in k_mat_lines:
						k_mat.append([float(num) for num in l.split()])
					k_mat = np.array(k_mat)
					cam.set_k_matrix(k_mat)

				#Radial distortion matrix
				elif j == 4:
					line = lines[i + j]
					rad_distor_mat = [float(num) for num in line.split()]
					rad_distor_mat = np.array(rad_distor_mat)
					cam.set_rad_distortion(rad_distor_mat)

				#Tangential distortion matrix
				elif j == 5:
					line = lines[i + j]
					tan_distor_mat = [float(num) for num in line.split()]
					tan_distor_mat = np.array(tan_distor_mat)
					cam.set_tan_distor(tan_distor_mat)

				#Position matrix
				elif j == 6:
					line = lines[i + j]
					pos_mat = [float(num) for num in line.split()]
					pos_mat = np.array(pos_mat)
					cam.set_camera_pos(pos_mat)

				#Rotation Matrix
				elif j == 7:
					rot_mat_lines = lines[i + 7: i + 10]
					rot_mat = []
					for l in rot_mat_lines:
						rot_mat.append([float(num) for num in l.split()])
					rot_mat = np.array(rot_mat)
					cam.set_camera_rot(rot_mat)
	return cameras