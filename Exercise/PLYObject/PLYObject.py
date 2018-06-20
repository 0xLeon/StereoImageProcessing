#!/usr/bin/env python3

import itertools

import plyfile
import numpy as np
import scipy.optimize

class PLYObject:
	def __init__(self, name=None):
		self.path = name
		self.plydata = plyfile.PlyData()

		if not name is None:
			self.read(name)

	def read(self, name):
		"""
		Liest die PLY-Datei name ein.
		"""
		self.path = name
		self.plydata = plyfile.PlyData.read(name)

	def write(self, name):
		"""
		Speichert das Objekt in der PLY-Datei name ab.
		"""
		self.path = name
		self.plydata.write(name)

	def getVertices(self):
		"""
		Liefert die n Knoten des Objekts als 3xn-ndarray.
		"""
		return np.asarray([self.plydata['vertex'][dim] for dim in ('x', 'y', 'z')])

	def setVertices(self, vertices):
		"""
		Nimmt Koordinaten von n Knoten als 3xn-ndarray entgegen und ueberschreibt damit die im Objekt gespeicherten Knoten.
		"""
		for i in range(3):
			self.plydata['vertex'][('x', 'y', 'z')[i]] = vertices[i]

	def flipFaces(self):
		"""
		Vertauscht die Ausrichtung der Flaechen.
		"""
		for i in range(self.plydata['face']['vertex_indices'].shape[0]):
			self.plydata['face']['vertex_indices'][i] = self.plydata['face']['vertex_indices'][i][::-1]

	def apply(self, mat):
		"""
		Wendet eine Transformation (als 3x3/3x4/4x3/4x4-ndarray) auf die Knoten des aktuell geladenen Objektes an.
		"""
		ver = self.getVertices()

		if mat.shape[1] == 4:
			ver = np.concatenate((ver, np.ones((1, ver.shape[1]))))

		ver = mat.dot(ver)

		if ver.shape[0] == 4:
			ver = ver[:3]

		self.setVertices(ver)

	def translate(self, v):
		"""
		Fuehrt eine Verschiebung um den Vektor v (ndarray) durch.
		"""
		if len(v) < 3:
			v = v + [0] * (3 - len(v))

		mat = np.matrix([[1, 0, 0, v[0]],
						 [0, 1, 0, v[1]],
						 [0, 0, 1, v[2]],
						 [0, 0, 0, 1]])

		self.apply(mat)

	def scale(self, f):
		"""
		Skaliert das Objekt um den Faktor f ausgehend vom Koordinatenursprung.
		"""
		mat = np.matrix([[f, 0, 0, 0],
						 [0, f, 0, 0],
						 [0, 0, f, 0],
						 [0, 0, 0, 1]])

		self.apply(mat)

	def rotateX(self, alpha):
		"""
		Fuehrt eine Drehung um die x-Achse um den Winkel alpha (BogenmaB) durch. (Rechte-Hand-Regel im Rechtssystem)
		"""
		ca = np.cos(alpha)
		sa = np.sin(alpha)

		mat = np.matrix([[1, 0, 0, 0],
						 [0, ca, -sa, 0],
						 [0, sa, ca, 0],
						 [0, 0, 0, 1]])

		self.apply(mat)

	def flipX(self):
		"""
		Fuehrt eine Spiegelung an der Ebene durch, welche im Ursprung senkrecht zur x-Achse steht.
		"""
		mat = np.matrix([[-1, 0, 0, 0],
						 [0, 1, 0, 0],
						 [0, 0, 1, 0],
						 [0, 0, 0, 1]])

		self.apply(mat)

	@classmethod
	def from_vertices(cls, vertices, dtype=None):
		if dtype is None:
			dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

		if len(vertices) > 0 and not isinstance(vertices[0], tuple):
			vertices = [tuple(v) for v in vertices]

		plyVertices = np.array(vertices, dtype=dtype)
		plyVertices = plyfile.PlyElement.describe(plyVertices, 'vertex')

		plyObject = cls()
		plyObject.plydata = plyfile.PlyData([plyVertices])

		return plyObject

	def fitSphere(self, vertices=None):
		vertices = self.getVertices() if vertices is None else vertices # type: np.ndarray
		centerEstimate = vertices.mean(axis=1)
		radiusEstimate = np.linalg.norm(vertices.T - centerEstimate, axis=1).mean()
		sphereParams = np.concatenate((centerEstimate, [radiusEstimate]))

		def errorFuncSphere(sphereParams, verticesT):
			return np.linalg.norm(verticesT - sphereParams[:3], axis=1) - sphereParams[3]

		result, status = scipy.optimize.leastsq(errorFuncSphere, sphereParams, args=(vertices.T,))

		if status not in [1, 2, 3, 4]:
			raise RuntimeError('Can\'t fit sphere to given data')

		delta = errorFuncSphere(result, vertices.T)

		return tuple(itertools.chain(result, (np.abs(delta).mean(),)))

	def fitPlane(self, vertices=None):
		vertices = self.getVertices() if vertices is None else vertices # type: np.ndarray
		samplePoints = vertices[:, np.random.choice(vertices.shape[1], 3, replace=False)].T
		normal = np.cross(samplePoints[2] - samplePoints[0], samplePoints[1] - samplePoints[0])
		d = (normal * samplePoints[0]).sum()
		planeParams = np.concatenate((normal, [d]))

		def errorFuncPlane(planeParams, verticesT):
			return (verticesT.dot(planeParams[:3]) + planeParams[3]) / np.linalg.norm(planeParams[:3])

		result, status = scipy.optimize.leastsq(errorFuncPlane, planeParams, args=(vertices.T,))

		if status not in [1, 2, 3, 4]:
			raise RuntimeError('Can\'t fit plane to given data')

		result = result / -result[3]
		delta = errorFuncPlane(result, vertices.T)

		return tuple(itertools.chain(result[:3], (np.abs(delta).mean(),)))

	@classmethod
	def generate_geosphere(cls, sphereParams, frequency=5):
		sin_phi = 2.0 / np.sqrt(5.0)
		cos_phi = 1.0 / np.sqrt(5.0)

		zero_four_pi = 0.4 * np.pi
		zero_eight_pi = 0.8 * np.pi

		sin_phi_sin_zero_four_pi = sin_phi * np.sin(zero_four_pi)
		sin_phi_sin_zero_eight_pi = sin_phi * np.sin(zero_eight_pi)
		sin_phi_cos_zero_four_pi = sin_phi * np.cos(zero_four_pi)
		sin_phi_cos_zero_eight_pi = sin_phi * np.cos(zero_eight_pi)

		polyhedronVertices = np.array([
			[0., 0., 1., 1.],
			[sin_phi, 0., cos_phi, 1.],
			[sin_phi_cos_zero_four_pi, sin_phi_sin_zero_four_pi, cos_phi, 1.],
			[sin_phi_cos_zero_eight_pi, sin_phi_sin_zero_eight_pi, cos_phi, 1.],
			[sin_phi_cos_zero_eight_pi, -sin_phi_sin_zero_eight_pi, cos_phi, 1.],
			[sin_phi_cos_zero_four_pi, -sin_phi_sin_zero_four_pi, cos_phi, 1.],
			[-sin_phi_cos_zero_eight_pi, -sin_phi_sin_zero_eight_pi, -cos_phi, 1.],
			[-sin_phi_cos_zero_eight_pi, sin_phi_sin_zero_eight_pi, -cos_phi, 1.],
			[-sin_phi_cos_zero_four_pi, sin_phi_sin_zero_four_pi, -cos_phi, 1.],
			[-sin_phi, 0., -cos_phi, 1.],
			[-sin_phi_cos_zero_four_pi, -sin_phi_sin_zero_four_pi, -cos_phi, 1.],
			[0., 0., -1., 1.],
		])
		polyhedronFaces = [
			[1, 2, 0],
			[2, 3, 0],
			[3, 4, 0],
			[4, 5, 0],
			[5, 1, 0],
			[5, 6, 1],
			[1, 7, 2],
			[2, 8, 3],
			[3, 9, 4],
			[4, 10, 5],
			[1, 6, 7],
			[2, 7, 8],
			[3, 8, 9],
			[4, 9, 10],
			[5, 10, 6],
			[6, 11, 7],
			[7, 11, 8],
			[8, 11, 9],
			[9, 11, 10],
			[10, 11, 6],
		]

		vertices = []

		masterTri = cls._classOneSample(frequency)
		masterTriCorners = cls._getTriCorners(masterTri)

		for face in polyhedronFaces:
			target = np.array([polyhedronVertices[i] for i in face])
			transform, _, _, _ = np.linalg.lstsq(masterTriCorners, target, -1.0)

			tri = masterTri.dot(transform)[:, :3]
			tri = tri / np.linalg.norm(tri, axis=1)[:, np.newaxis]
			tri = sphereParams[3] * tri + sphereParams[:3]

			for vert in tri:
				vertices.append(tuple(vert))

		return cls.from_vertices(vertices)

	@classmethod
	def generate_sphere(cls, sphereParams, h=30, v=72):
		vertices = []

		pi_h = np.pi / h
		two_pi_v = 2 * np.pi / v

		for m in range(0, h):
			pi_h_m = pi_h * m
			sin_pi_h_m = np.sin(pi_h_m)
			cos_pi_h_m = np.cos(pi_h_m)

			for n in range(0, v):
				two_pi_v_n = two_pi_v * n

				x = sin_pi_h_m * np.cos(two_pi_v_n)
				y = sin_pi_h_m * np.sin(two_pi_v_n)
				z = cos_pi_h_m

				vertices.append(tuple((np.array([x, y, z]) * sphereParams[3]) + sphereParams[:3]))

		return cls.from_vertices(vertices)

	@classmethod
	def generate_plane(cls, planeParams, xrange=50, zrange=50):
		if isinstance(xrange, int):
			xrange = range(-xrange, xrange + 1)

		if isinstance(zrange, int):
			zrange = range(-zrange, zrange + 1)

		vertices = []

		for x in xrange:
			for z in zrange:
				y = (-planeParams[0] * x - planeParams[2] * z - planeParams[3]) / planeParams[1]

				vertices.append((float(x), y, float(z)))

		return cls.from_vertices(vertices)

	@staticmethod
	def _classOneSample(f):
		freq = float(f)

		sin_pi_third = np.sin(np.pi / 3.0)
		cos_pi_third = np.cos(np.pi / 3.0)
		sin_pi_third_third = sin_pi_third / 3.0

		vertices = []

		for r in range(f + 1):
			r_cos_pi_third = r * cos_pi_third

			y = (r * sin_pi_third / freq) - sin_pi_third_third

			for c in range(f - r + 1):
				x = ((r_cos_pi_third + c) / freq) - 0.5

				vertices.append([x, y, 0.0, 1.0])

		return np.array(vertices)

	@staticmethod
	def _getTriCorners(tri):
		return np.array([
			[0.0, tri.T[1].max(), 0.0, 1.0],
			[tri.T[0].max(), tri.T[1].min(), 0.0, 1.0],
			[tri.T[0].min(), tri.T[1].min(), 0.0, 1.0],
		])
