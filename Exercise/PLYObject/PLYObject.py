#!/usr/bin/env python3

import plyfile
import numpy
import scipy.optimize
import itertools


class PLYObject:

	def __init__(self, name=None):
		if not name is None:
			self.read(name)

	def read(self, name):
		"""
		Liest die PLY-Datei name ein.
		"""
		self.plydata = plyfile.PlyData.read(name)

	def write(self, name):
		"""
		Speichert das Objekt in der PLY-Datei name ab.
		"""
		self.plydata.write(name)

	def getVertices(self):
		"""
		Liefert die n Knoten des Objekts als 3xn-ndarray.
		"""
		return numpy.asarray([self.plydata['vertex'][dim] for dim in ('x', 'y', 'z')])

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
			ver = numpy.concatenate((ver, numpy.ones((1, ver.shape[1]))))

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

		mat = numpy.matrix([[1, 0, 0, v[0]],
							[0, 1, 0, v[1]],
							[0, 0, 1, v[2]],
							[0, 0, 0, 1]])

		self.apply(mat)

	def scale(self, f):
		"""
		Skaliert das Objekt um den Faktor f ausgehend vom Koordinatenursprung.
		"""
		mat = numpy.matrix([[f, 0, 0, 0],
							[0, f, 0, 0],
							[0, 0, f, 0],
							[0, 0, 0, 1]])

		self.apply(mat)

	def rotateX(self, alpha):
		"""
		Fuehrt eine Drehung um die x-Achse um den Winkel alpha (BogenmaB) durch. (Rechte-Hand-Regel im Rechtssystem)
		"""
		ca = numpy.cos(alpha)
		sa = numpy.sin(alpha)

		mat = numpy.matrix([[1, 0, 0, 0],
							[0, ca, -sa, 0],
							[0, sa, ca, 0],
							[0, 0, 0, 1]])

		self.apply(mat)

	def flipX(self):
		"""
		Fuehrt eine Spiegelung an der Ebene durch, welche im Ursprung senkrecht zur x-Achse steht.
		"""
		mat = numpy.matrix([[-1, 0, 0, 0],
							[0, 1, 0, 0],
							[0, 0, 1, 0],
							[0, 0, 0, 1]])

		self.apply(mat)

	@staticmethod
	def from_vertices(vertices, dtype=None):
		if dtype is None:
			dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

		plyVertices = numpy.array(vertices, dtype=dtype)
		plyVertices = plyfile.PlyElement.describe(plyVertices, 'vertex')

		plyObject = PLYObject()
		plyObject.plydata = plyfile.PlyData([plyVertices])

		return plyObject

	def fitSphere(self, absoluteDelta=False):
		vertices = self.getVertices() # type: numpy.ndarray
		centerEstimate = vertices.T.mean(axis=0)
		radiusEstimate = numpy.linalg.norm(centerEstimate - vertices.T[0])
		sphereParams = numpy.concatenate((centerEstimate, [radiusEstimate]))

		def errorFuncSphere(sphereParams, verticesT):
			return numpy.linalg.norm(verticesT - sphereParams[:3], axis=1) - sphereParams[3]

		result, status = scipy.optimize.leastsq(errorFuncSphere, sphereParams, args=(vertices.T,))

		if status not in [1, 2, 3, 4]:
			raise RuntimeError('Can\'t fit sphere to given data')

		delta = errorFuncSphere(result, vertices.T)

		if absoluteDelta:
			delta = numpy.abs(delta)

		return tuple(itertools.chain(result, (delta.mean(),)))

	@staticmethod
	def from_sphere(sphereParams, h=30, v=72):
		vertices = []

		for m in range(0, h):
			for n in range(0, v):
				x = numpy.sin(numpy.pi * m / h) * numpy.cos(2 * numpy.pi * n / v)
				y = numpy.sin(numpy.pi * m / h) * numpy.sin(2 * numpy.pi * n / v)
				z = numpy.cos(numpy.pi * m / h)

				vertices.append(tuple((numpy.array([x, y, z]) * sphereParams[3]) + sphereParams[:3]))

		return PLYObject.from_vertices(vertices)

	def fitPlane(self, absoluteDelta=False):
		vertices = self.getVertices() # type: numpy.ndarray
		samplePoints = vertices.T[numpy.random.choice(vertices.T.shape[0], 3, replace=False), :]
		normal = numpy.cross(samplePoints[2] - samplePoints[0], samplePoints[1] - samplePoints[0])
		d = (normal * samplePoints[0]).sum()
		unitNormal = normal / numpy.linalg.norm(normal)
		planeParams = numpy.concatenate((normal, [d]))

		def errorFuncPlane(planeParams, verticesT):
			return (verticesT.dot(planeParams[:3]) + planeParams[3]) / numpy.linalg.norm(planeParams[:3])

		result, status = scipy.optimize.leastsq(errorFuncPlane, planeParams, args=(vertices.T,))

		if status not in [1, 2, 3, 4]:
			raise RuntimeError('Can\'t fit sphere to given data')

		result = result / -result[3]
		delta = errorFuncPlane(result, vertices.T)

		if absoluteDelta:
			delta = numpy.abs(delta)

		return tuple(itertools.chain(result[:3], (delta.mean(),)))

	@staticmethod
	def from_plane(planeParams, xrange=100, zrange=100):
		if isinstance(xrange, int):
			xrange = range(xrange)

		if isinstance(zrange, int):
			zrange = range(zrange)

		vertices = []

		for x in xrange:
			for z in zrange:
				y = (-planeParams[0] * x - planeParams[2] * z - planeParams[3]) / planeParams[1]

				vertices.append((float(x), float(y), float(z)))

		return PLYObject.from_vertices(vertices)
