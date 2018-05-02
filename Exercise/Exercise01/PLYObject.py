import plyfile
import numpy


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

		if mat.shape[1] == 4:
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


# Leitet eine Testumgebung ein. Der folgende Code wird bei der Korrektur nicht ausgefuehrt(!) und kann daher beliebig abgeaendert werden.
if __name__ == "__main__":
	import copy

	print('Loading lucy.ply')

	o1 = PLYObject('lucy.ply')
	o2 = copy.deepcopy(o1)
	o3 = copy.deepcopy(o1)
	o4 = copy.deepcopy(o1)

	print('Loaded lucy.ply')

	# test translate
	print('Translating')
	o1.translate([100, 200, 300])
	print('Translated')
	o1.write('lucy_translate.ply')
	print('Wrote lucy_translate.ply')

	# test scale
	print('Scaling')
	o2.scale(0.5)
	print('Scaled')
	o2.write('lucy_scale.ply')
	print('Wrote lucy_scale.ply')

	# test rotateX
	print('Rotating')
	o3.rotateX(0.5 * numpy.pi)
	print('Rotated')
	o3.write('lucy_rotate.ply')
	print('Wrote lucy_rotate.ply')

	# test flipX
	print('Flipping')
	o4.flipX()
	o4.flipFaces()
	print('Flipped')
	o4.write('lucy_flip.ply')
	print('Wrote lucy_flip.ply')
