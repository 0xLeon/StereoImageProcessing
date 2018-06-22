import numpy as np
import scipy.spatial

import PLYObject

def getRealDensityFromObject(ply):
	# type: (PLYObject.PLYObject) -> float

	vertices = ply.getVertices().T
	hull = scipy.spatial.ConvexHull(vertices)

	return vertices.shape[0] / hull.volume

def getRealDensityFromPlane(ply, planeParams=None, returnAlignedPly=False):
	# type: (PLYObject.PLYObject, tuple) -> float

	if planeParams is None:
		planeParams = ply.fitPlane()
	elif len(planeParams) != 4:
		raise ValueError('Invalid number of values for plane parameters')

	vertices = ply.getVertices().T

	zAxis = np.array([0.0, 0.0, 1.0])
	planeNormal = np.array(planeParams[:3])
	planeNormalLength = np.linalg.norm(planeNormal)
	unitPlaneNormal = planeNormal / planeNormalLength

	vec = planeNormal / planeNormalLength**2

	# A = unitPlaneNormal
	# B = zAxis

	# Shift plane to origin
	vertices += vec

	# Find roation aligning the plane normal vector to Z axis
	vecDot = unitPlaneNormal.dot(zAxis)
	vecCross = np.cross(zAxis, unitPlaneNormal)
	vecCrossNorm = np.linalg.norm(vecCross)

	G = np.array([[vecDot, -1 * vecCrossNorm, 0], [vecCrossNorm, vecDot, 0], [0, 0, 1]])

	u = unitPlaneNormal
	v = (zAxis - vecDot * unitPlaneNormal) / np.linalg.norm(zAxis - vecDot * unitPlaneNormal)
	w = vecCross
	Finv = np.column_stack([u, v, w])

	U = Finv.dot(G.dot(np.linalg.inv(Finv)))

	# Apply rotation to vertices to align them with axis
	vertices = U.dot(vertices.T).T

	# Use naiv method to get density
	hull = scipy.spatial.ConvexHull(vertices[:, :2])
	pDensity = vertices.shape[0] / hull.volume

	if returnAlignedPly:
		return (pDensity, PLYObject.PLYObject.from_vertices(vertices))

	return pDensity

def getNaivDensityFromPlane(ply, alignedAxis=2):
	# type: (PLYObject.PLYObject, int) -> float

	idx = np.array([0, 1, 2])
	idx = idx[idx != alignedAxis]
	v = ply.getVertices().T
	v = v[:, idx]

	hull = scipy.spatial.ConvexHull(v)

	# Note: As the hull was calculated in 2D space,
	# the hull 'volume' is actually the surface area.
	return v.shape[0] / hull.volume

if __name__ == '__main__':
	pass
