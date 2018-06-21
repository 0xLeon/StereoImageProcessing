import argparse
import enum
import glob
import os
import re

import numpy as np

import PLYObject

class Axis(enum.Enum):
	X = 0
	Y = 1
	Z = 2

class Operator(enum.Enum):
	GT = '>'
	LT = '<'
	GE = '>='
	LE = '<='
	EQ = '='
	NE = '!='

class PointCloudFilter(object):
	tokenRegex = re.compile(r'(?P<axis>[x-z])(?P<operator>[^\d\-]+)(?P<limit>-?\d+)', re.I)
	operatorToMethod = {
		Operator.GT: np.ndarray.__gt__,
		Operator.LT: np.ndarray.__lt__,
		Operator.GE: np.ndarray.__ge__,
		Operator.LE: np.ndarray.__le__,
		Operator.EQ: np.ndarray.__eq__, # TODO: replace with robust float equality
		Operator.NE: np.ndarray.__ne__,
	}

	def __init__(self, axis, operator, limit):
		self.axis = axis
		self.operator = operator
		self.limit = limit

	def accept(self, points):
		return self.__class__.operatorToMethod[self.operator](points[:, self.axis.value], self.limit)

	@classmethod
	def fromFilterString(cls, filterStr):
		filterParts = cls.tokenRegex.match(filterStr)

		if not filterParts:
			raise ValueError()

		try:
			axis = Axis[filterParts.group('axis').upper()]
		except KeyError:
			raise ValueError()

		operator = Operator(filterParts.group('operator').strip())
		limit = float(filterParts.group('limit'))

		return cls(axis, operator, limit)

def filterGarbageOld1(vertices, minJump=5, garbageAxis=Axis.Z):
	# type: (np.ndarray) -> np.ndarray
	uv2 = np.unique(np.sort(vertices[:, garbageAxis.value]))
	duv2 = np.abs(np.diff(uv2))
	i = np.argmax(duv2)
	delta = uv2[i + 1] - uv2[i]

	if delta < (duv2[:(i+1)].mean() * minJump):
		return vertices

	limit = uv2[i] + delta / 2

	if (limit - uv2[i]) > 0:
		return vertices[vertices[:, garbageAxis.value] < limit]

	return vertices[vertices[:, garbageAxis.value] > limit]

def filterGarbage(vertices, minJump=5, rounds=8, garbageAxis=Axis.Z):
	# type: (np.ndarray) -> np.ndarray
	uv2 = np.unique(np.sort(vertices[:, garbageAxis.value]))
	duv2 = np.abs(np.diff(uv2))

	i = len(duv2)

	for _ in range(rounds):
		indexes = (duv2 > duv2[:i].mean() * minJump).nonzero()[0]

		if indexes.size == 0 or i == indexes[0]:
			break

		i = indexes[0]

	if i == len(duv2):
		return vertices

	delta = uv2[i + 1] - uv2[i]
	limit = uv2[i] + delta / 2

	if (limit - uv2[i]) > 0:
		return vertices[vertices[:, garbageAxis.value] < limit]

	return vertices[vertices[:, garbageAxis.value] > limit]

def parseFilters(filters):
	pFilters = []

	for pFilter in filters:
		pFilter = pFilter if isinstance(pFilter, PointCloudFilter) else PointCloudFilter.fromFilterString(pFilter)
		pFilters.append(pFilter)

	# TODO: reduce filters to minimal necessary filter set

	return pFilters

def filterPLYObject(ply, filters):
	# type: (PLYObject.PLYObject, List) -> PLYObject.PLYObject
	v = ply.getVertices().T
	v = filterGarbage(v)
	selector = np.array([True] * v.shape[0])

	for pFilter in filters:
		selector = selector & pFilter.accept(v)

	v = v[selector]

	return PLYObject.PLYObject.from_vertices(v)

def main(plyfiles, filters):
	pFilters = parseFilters(filters)

	for plyPath in plyfiles:
		ply = PLYObject.PLYObject(plyPath)
		newPly = filterPLYObject(ply, pFilters)

		plyPathPart = os.path.splitext(os.path.abspath(ply))[0]
		plyPathNew = '{:s}.filtered.ply'.format(plyPathPart)
		newPly.write(plyPathNew)

def main_cli(args=None):
	parser = argparse.ArgumentParser()
	args = parser.parse_args(args)

	main([], [])

if __name__ == '__main__':
	main_cli()
