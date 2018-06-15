import argparse
import os
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

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

#region Disk Saving

#region Data Conversion
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
	return cv2.KeyPoint(
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
	return cv2.DMatch(
		match['queryIdx'],
		match['trainIdx'],
		match['imgIdx'],
		match['distance'],
	)

#endregion

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

#endregion

def matchImages(imgA, imgB, nFeatures=50000, qualityThreshold=0.8):
	orb = cv2.ORB_create(nFeatures)
	flann = cv2.FlannBasedMatcher(
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

def drawMatchedImages(imgA, kpA, imgB, kpB, matches, matchesMask=None):
	while not isinstance(matches[0], cv2.DMatch):
		matches = [match[0] for match in matches]

		if matchesMask is not None:
			matchesMask = [mask[0] for mask in matchesMask]

	drawParams = dict(
		matchColor=(0, 255, 0),
		singlePointColor=(255, 0, 0),
		matchesMask=matchesMask,
		flags=0,
	)

	return cv2.drawMatches(imgA, kpA, imgB, kpB, matches, None, **drawParams)

def calculateFundamentalMatrix(npkpA, npkpB):
	A = np.array([
		[b[0] * a[0], b[0] * a[1], b[0], b[1] * a[0], b[1] * a[1], b[1], a[0], a[1], 1]
		for a, b in zip(npkpA, npkpB)
	])
	A2 = A.T.dot(A)

	eigVal, eigVec = np.linalg.eig(A2)

	F = eigVec[:, np.argmin(eigVal)].reshape(3, 3)
	U, S, VT = np.linalg.svd(F)
	S[-1] = 0
	F = (U * S).dot(VT)

	return (F / F[-1, -1])

def calculateFundamentalMatrixRansac(kpA, kpB, matches, matchesMask=None, iterations=1000, epsilon=0.01):
	while not isinstance(matches[0], cv2.DMatch):
		matches = [match[0] for match in matches]

		if matchesMask is not None:
			matchesMask = [mask[0] for mask in matchesMask]

	originalMatchesLen = len(matches)

	if matchesMask is not None:
		matches = np.array([match for match, mask in zip(matches, matchesMask) if mask == 1])

	iterations = int(iterations)
	epsilon = float(epsilon)

	candidateSets = []

	npkpA = np.zeros((originalMatchesLen, 3))
	npkpB = np.zeros((originalMatchesLen, 3))

	npkpA[:, 2] = 1.0
	npkpB[:, 2] = 1.0

	for i, (pA, pB) in enumerate(zip(kpA, kpB)):
		npkpA[i, :2] = pA.pt
		npkpB[i, :2] = pB.pt

	for _ in range(iterations):
		matchesChoice = np.random.choice(matches, 8, False)
		kpAChoice = [npkpA[match.queryIdx] for match in matchesChoice]
		kpBChoice = [npkpB[match.trainIdx] for match in matchesChoice]

		F = calculateFundamentalMatrix(kpAChoice, kpBChoice)

		candidateSet = []

		for match in matches:
			a = npkpA[match.queryIdx]
			b = npkpB[match.trainIdx]

			d = np.abs(b.dot(F.dot(a)))

			if d < epsilon:
				candidateSet.append(match)

		if len(candidateSet) >= 8:
			candidateSets.append(candidateSet)

	if not candidateSets:
		raise ValueError('Unable to calculate at least one consensus set')

	bestConsensusSetIdx = np.argmax([len(consensusSet) for consensusSet in candidateSets])

	consensusKpA = [npkpA[match.queryIdx] for match in candidateSets[bestConsensusSetIdx]]
	consensusKpB = [npkpB[match.trainIdx] for match in candidateSets[bestConsensusSetIdx]]

	F = calculateFundamentalMatrix(consensusKpA, consensusKpB)

	return F, candidateSets[bestConsensusSetIdx]

def main(images=None, readMatch='', output='./'):
	if images is None:
		images = ['a.jpg', 'b.jpg']

	imgA = cv2.imread(images[0])
	imgB = cv2.imread(images[1])

	if readMatch:
		with TimeMeasurement('Read Matching Result'):
			kpA, desA, kpB, desB, matches, matchesMask = loadStereoMatchingResult(readMatch)
	else:
		with TimeMeasurement('Feature Matching'):
			kpA, desA, kpB, desB, matches, matchesMask = matchImages(imgA, imgB, qualityThreshold=0.65)

		with TimeMeasurement('Save Matching Result'):
			saveStereoMatchingResult(kpA, desA, kpB, desB, matches, matchesMask, os.path.join(output, 'stereoMatch.pkl'))

	with TimeMeasurement('Calculate Fundamental Matrix with RANSAC'):
		F, cMatches = calculateFundamentalMatrixRansac(kpA, kpB, matches, matchesMask)

	imgMatch = drawMatchedImages(imgA, kpA, imgB, kpB, cMatches)

	np.savetxt(os.path.join(output, 'fmat.txt'), F)
	cv2.imwrite(os.path.join(output, 'matches.png'), imgMatch)

def main_cli(args=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--readmatch', default='')
	parser.add_argument('--output', default='./')
	parser.add_argument('images', default=['a.jpg', 'b.jpg'], nargs='*')
	args = parser.parse_args(args)

	if len(args.images) != 2:
		parser.error('Specifiy exactly two images!')

	main(args.images, args.readmatch, args.output)

if __name__ == '__main__':
	main_cli()
