#!/usr/bin/env python3

import argparse
import itertools
import os
import pickle
import sys

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
	if isinstance(file, str):
		with open(file, 'rb') as f:
			data = pickle.load(f)
	else:
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

def loadMatches(file):
	if isinstance(file, str):
		with open(file, 'rb') as f:
			data = pickle.load(f)
	else:
		data = pickle.load(file)

	matches = [(dictToMatch(match[0]), dictToMatch(match[1])) for match in data['matches']]

	return matches, data['matchesMask']

def saveStereoMatchingResult(kpA, desA, kpB, desB, matches, matchesMask, file='stereoMatch.pkl'):
	kpA = [*map(keypointToDict, kpA)]
	kpB = [*map(keypointToDict, kpB)]
	matches = [(matchToDict(match[0]), matchToDict(match[1])) for match in matches]

	data = dict(
		kpA=kpA,
		desA=desA,
		kpB=kpB,
		desB=desB,
		matches=matches,
		matchesMask=matchesMask,
	)

	if isinstance(file, str):
		with open(file, 'wb') as f:
			pickle.dump(data, f)
	else:
		pickle.dump(data, file)

def loadStereoMatchingResult(file='stereoMatch.pkl'):
	if isinstance(file, str):
		with open(file, 'rb') as f:
			data = pickle.load(f)
	else:
		data = pickle.load(file)

	kpA = [*map(dictToKeypoint, data['kpA'])]
	kpB = [*map(dictToKeypoint, data['kpB'])]
	matches = [(dictToMatch(match[0]), dictToMatch(match[1])) for match in data['matches']]

	return kpA, data['desA'], kpB, data['desB'], matches, data['matchesMask']

def matchImages(imgA, imgB, nFeatures=50000, qualityThreshold=0.8):
	orb = cv.ORB_create(nFeatures)
	flann = cv.FlannBasedMatcher(
		dict(algorithm=6, table_number=12, key_size=20, multi_proble_level=2),
		dict(checks=50),
	)

	kpA, desA = orb.detectAndCompute(imgA, None)
	kpB, desB = orb.detectAndCompute(imgB, None)

	matches = flann.knnMatch(desA, desB, k=2)
	matchesMask = [[0, 0] for i in range(len(matches))]

	for i, (m, n) in enumerate(matches):
		if m.distance < qualityThreshold * n.distance:
			matchesMask[i] = [1, 0]

	return kpA, desA, kpB, desB, matches, matchesMask

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

def main(camA='camA', camB='camB', readMatch='', output='./output/'):
	if output and not os.path.isdir(output):
		os.makedirs(output)

	camA = readCameraData(camA)
	camB = readCameraData(camB)

	imgA = cv.imread('camA_image.jpg')
	imgB = cv.imread('camb_image.jpg')

	if readMatch:
		kpA, desA, kpB, desB, matches, matchesMask = loadStereoMatchingResult(readMatch)
	else:
		kpA, desA, kpB, desB, matches, matchesMask = matchImages(imgA, imgB)

	saveStereoMatchingResult(kpA, desA, kpB, desB, matches, matchesMask, os.path.join(output, 'stereoMatch.pkl'))
	drawMatchedImages(imgA, kpA, imgB, kpB, matches, matchesMask)

	return

def main_cli(args=None):
	if args is None:
		args = sys.argv[1:]

	parser = argparse.ArgumentParser()
	parser.add_argument('--camA', default='camA')
	parser.add_argument('--camB', default='camB')
	parser.add_argument('--readmatch', default='')
	parser.add_argument('--output', default='./output/')
	args = parser.parse_args(args)

	main(args.camA, args.camB, args.readmatch, args.output)

if __name__ == '__main__':
	main_cli()
