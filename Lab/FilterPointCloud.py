import argparse
import enum
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
		self.operator = self.__class__.operatorToMethod[operator]
		self.limit = limit

	def accept(self, points):
		return self.operator(points[:, self.axis.value], self.limit)

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

def main(filters):
	pFilters = []

	for pFilter in filters:
		pFilter = PointCloudFilter.fromFilterString(pFilter)
		pFilters.append(pFilter)

	# TODO: reduce filters to minimal necessary filter set

def main_cli(args=None):
	parser = argparse.ArgumentParser()
	args = parser.parse_args(args)

	main([])

if __name__ == '__main__':
	main_cli()
