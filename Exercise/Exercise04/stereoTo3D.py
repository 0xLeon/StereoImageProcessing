#!/usr/bin/env python3

import argparse
import itertools

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import plyfile

def readCameraData(cameraName):
	return {
		'projection': np.genfromtxt('{:s}_K.txt'.format(cameraName)),
		'rotation': np.genfromtxt('{:s}_R.txt'.format(cameraName)),
		'translation': np.genfromtxt('{:s}_T.txt'.format(cameraName)),
		'distortion': np.genfromtxt('{:s}_D.txt'.format(cameraName)),
	}

def generateDisparityMap():
	pass

def generatePointcloud():
	pass

def drawMatchedImages(imgA, kpA, imgB, kpB, matches, matchesMask):
	drawParams = dict(
		matchColor=(0, 255, 0),
		singlePointColor=(255, 0, 0),
		matchesMask=matchesMask,
		flags=0,
	)

	img = cv.drawMatchesKnn(imgA, kpA, imgB, kpB, matches, None, **drawParams)

	plt.imshow(img)
	plt.show()

def main():
	pass

if __name__ == '__main__':
	main()
