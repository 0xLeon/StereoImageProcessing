import argparse
import glob
import itertools
import os
import pathlib
import time

import cv2
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.preprocessing

class TimeMeasurement(object):
	def __init__(self, operation, printStd=True):
		self._t0 = 0
		self._t1 = 0

		self._operation = operation
		self._printStd = printStd

	@property
	def delta(self):
		return self._t1 - self._t0

	def __enter__(self):
		self._t1 = 0
		self._t0 = time.perf_counter()

		return self

	def __exit__(self, *args):
		self._t1 = time.perf_counter()

		if self._printStd:
			print('Operation \'{!s}\' took {:.4f} s'.format(self._operation, self.delta))

def findImages(directory='.'):
	rawLabels = list(filter(os.path.isdir, os.listdir(directory)))
	labelEncoder = sklearn.preprocessing.LabelEncoder().fit(rawLabels)

	rawImageSet = [glob.glob(os.path.join(folder, 'image*')) for folder in rawLabels]

	rawImageSetLabels = (itertools.repeat(rawLabel, len(imageSet)) for imageSet, rawLabel in zip(rawImageSet, rawLabels))
	rawImageSetLabels = list(itertools.chain.from_iterable(rawImageSetLabels))

	imageSet = np.array(list(itertools.chain.from_iterable(rawImageSet)))
	imageSetLabels = labelEncoder.transform(rawImageSetLabels)

	return imageSet, imageSetLabels, labelEncoder

def main():
	with TimeMeasurement('Find Images'):
		images, labels, encoder = findImages()

	return

def main_cli():
	main()

if __name__ == '__main__':
	main_cli()
