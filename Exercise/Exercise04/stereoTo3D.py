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

class CameraData(object):
	def __init__(self, projection, rotation, translation, distortion):
		self.projection = projection # type: np.ndarray
		self.rotation = rotation # type: np.ndarray
		self.translation = translation # type: np.ndarray
		self.distortion = distortion # type: np.ndarray

	@classmethod
	def fromFile(cls, cameraName):
		return cls(
			np.genfromtxt('{:s}_K.txt'.format(cameraName)),
			np.genfromtxt('{:s}_R.txt'.format(cameraName)),
			np.genfromtxt('{:s}_T.txt'.format(cameraName)),
			np.genfromtxt('{:s}_D.txt'.format(cameraName)),
		)

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
	keypoints = list(map(keypointToDict, keypoints))

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

	keypoints = list(map(dictToKeypoint, data['keypoints']))

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
	kpA = list(map(keypointToDict, kpA))
	kpB = list(map(keypointToDict, kpB))
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

	kpA = list(map(dictToKeypoint, data['kpA']))
	kpB = list(map(dictToKeypoint, data['kpB']))
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
	matchesMask = [[0, 0]] * len(matches)

	for i, (m, n) in enumerate(matches):
		if m.distance < qualityThreshold * n.distance:
			matchesMask[i] = [1, 0]

	return kpA, desA, kpB, desB, matches, matchesMask

def undistortKeypoints(camera, keypoints):
	# type: (CameraData, List[cv.KeyPoint]) -> (np.ndarray, np.ndarray)

	points = np.array([[list(keypoint.pt) + [1] for keypoint in keypoints]]) # type: np.ndarray
	undistortedPoints = cv.undistortPoints(points[:, :, :2], camera.projection, camera.distortion)[0]
	undistortedPoints = np.array([[x[0], x[1], 1] for x in undistortedPoints])

	return undistortedPoints, points[0]

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

def main(camNameA='camA', camNameB='camB', readMatch='', output='./'):
	if output and not os.path.isdir(output):
		os.makedirs(output)

	camA = CameraData.fromFile(camNameA) # type: CameraData
	camB = CameraData.fromFile(camNameB) # type: CameraData

	imgA = cv.imread('{:s}_image.jpg'.format(camNameA))
	imgB = cv.imread('{:s}_image.jpg'.format(camNameB))

	if readMatch:
		kpA, desA, kpB, desB, matches, matchesMask = loadStereoMatchingResult(readMatch)
	else:
		kpA, desA, kpB, desB, matches, matchesMask = matchImages(imgA, imgB)
		saveStereoMatchingResult(kpA, desA, kpB, desB, matches, matchesMask, os.path.join(output, 'stereoMatch.pkl'))

	cA = -np.linalg.inv(camA.rotation).dot(camA.translation)
	cB = -np.linalg.inv(camB.rotation).dot(camB.translation)

	ukpA, kpA_raw = undistortKeypoints(camA, kpA)
	ukpB, kpB_raw = undistortKeypoints(camB, kpB)

	directionsA = np.linalg.inv(camA.rotation).dot(ukpA.T).T
	directionsB = np.linalg.inv(camB.rotation).dot(ukpB.T).T

	directionCACB = cB - cA

	distances = [0.0] * len(matches)
	vertices = []

	for keypointA, dA, keypointB, dB, mask, i in zip(kpA, directionsA, kpB, directionsB, matchesMask, range(len(matches))):
		if mask[0] != 1:
			continue

		D = cA + dA * ((-(dA.dot(dB) * dB.dot(directionCACB)) + dA.dot(directionCACB) * dB.dot(dB)) / (dA.dot(dA) * dB.dot(dB) - dA.dot(dB) * dA.dot(dB)))
		E = cB + dB * ((dA.dot(dB) * dA.dot(directionCACB) - dB.dot(directionCACB) * dA.dot(dA)) / (dA.dot(dA) * dB.dot(dB) - dA.dot(dB) * dA.dot(dB)))

		distances[i] = np.linalg.norm(E - D)

		if distances[i] <= 1:
			point = D + 0.5 * (E - D)
			vertices.append(tuple(point))

	plyVertices = np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
	plyVertices = plyfile.PlyElement.describe(plyVertices, 'vertex')
	plyfile.PlyData([plyVertices]).write(os.path.join(output, 'output.ply'))

	return

def main_cli(args=None):
	if args is None:
		args = sys.argv[1:]

	parser = argparse.ArgumentParser()
	parser.add_argument('--camA', default='camA')
	parser.add_argument('--camB', default='camB')
	parser.add_argument('--readmatch', default='')
	parser.add_argument('--output', default='./')
	args = parser.parse_args(args)

	main(args.camA, args.camB, args.readmatch, args.output)

if __name__ == '__main__':
	main_cli()
