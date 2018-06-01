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
	camA = readCameraData('camA')
	camB = readCameraData('camB')

	imgA = cv.imread('camA_image.jpg')
	imgB = cv.imread('camb_image.jpg')

	orb = cv.ORB_create(50000)
	flann = cv.FlannBasedMatcher(
		dict(algorithm=6, table_number=12, key_size=20, multi_proble_level=2),
		dict(checks=50),
	)

	kpA, desA = orb.detectAndCompute(imgA, None)
	kpB, desB = orb.detectAndCompute(imgB, None)

	matches = flann.knnMatch(desA, desB, k=2)
	matchesMask = [[0, 0] for i in range(len(matches))]

	c = 0

	for i, (m, n) in enumerate(matches):
		if m.distance < 0.8 * n.distance:
			matchesMask[i] = [1, 0]

			c += 1

	print(c)

	drawMatchedImages(imgA, kpA, imgB, kpB, matches, matchesMask)

	return

if __name__ == '__main__':
	main()
