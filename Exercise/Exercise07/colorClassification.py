import argparse
import glob
import itertools
import os
import pathlib

import cv2
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.preprocessing

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
	images, labels, encoder = findImages()

	return

def main_cli():
	main()

if __name__ == '__main__':
	main_cli()
