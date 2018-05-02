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
	o = PLYObject("lucy.ply")
	o.flipX()
	o.flipFaces()  # sinnvoll in Kombination mit Spiegelungen, damit danach nicht die Rueckseiten von Flaechen zu sehen sind
	# alternative Tests:
	# o.translate(numpy.array([2000,500,-1000]))
	# o.scale(0.4)
	# o.rotateX(1.1)
	# ...
	o.write("lucy_modified.ply")
