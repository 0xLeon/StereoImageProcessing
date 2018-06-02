#!/usr/bin/env python3

import argparse
import itertools
import pickle

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

def keypointToDict(keypoint):
	return {
		'angle': keypoint.angle,
		'class_id': keypoint.class_id,
		'octave': keypoint.octave,
		'pt': keypoint.pt,
		'response': keypoint.response,
		'size': keypoint.size,
	}

def dictToKeypoint(keypoint):
	return cv.KeyPoint(
		keypoint['pt'][0],
		keypoint['pt'][1],
		keypoint['size'],
		keypoint['angle'],
		keypoint['response'],
		keypoint['octave'],
		keypoint['class_id'],
	)

def matchToDict(match):
	return {
		'queryIdx': match.queryIdx,
		'trainIdx': match.trainIdx,
		'imgIdx': match.imgIdx,
		'distance': match.distance,
	}

def dictToMatch(match):
	return cv.DMatch(
		match['queryIdx'],
		match['trainIdx'],
		match['imgIdx'],
		match['distance'],
	)

def saveFeatures(keypoints, descriptors, file):
	keypoints = [*map(keypointToDict, keypoints)]

	if isinstance(file, str):
		with open(file, 'wb') as f:
			pickle.dump(dict(keypoints=keypoints, descriptors=descriptors), f)
	else:
		pickle.dump(dict(keypoints=keypoints, descriptors=descriptors), file)

def loadFeatures(file):
	data = pickle.load(file)

	keypoints = [*map(dictToKeypoint, data['keypoints'])]

	return keypoints, data['descriptors']

def saveMatches(matches, matchesMask, file):
	matches = [(matchToDict(match[0]), matchToDict(match[1])) for match in matches]

	if isinstance(file, str):
		with open(file, 'wb') as f:
			pickle.dump(dict(matches=matches, matchesMask=matchesMask), f)
	else:
		pickle.dump(dict(matches=matches, matchesMask=matchesMask), file)

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
