import argparse
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

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

def drawMatchedImages(imgA, kpA, imgB, kpB, matches, matchesMask):
	while not isinstance(matches[0], cv2.DMatch):
		matches = [match[0] for match in matches]
		matchesMask = [mask[0] for mask in matchesMask]

	drawParams = dict(
		matchColor=(0, 255, 0),
		singlePointColor=(255, 0, 0),
		matchesMask=matchesMask,
		flags=0,
	)

	img = cv2.drawMatches(imgA, kpA, imgB, kpB, matches, None, **drawParams)

	plt.imshow(img)
	plt.show()

def calculateFundamentalMatrix(kpA, kpB):
	A = np.array([
		[b.pt[0] * a.pt[0], b.pt[0] * a.pt[1], b.pt[0], b.pt[1] * a.pt[0], b.pt[1] * a.pt[1], b.pt[1], a.pt[0], a.pt[1], 1]
		for a, b in zip(kpA, kpB)
	])
	A2 = A.T.dot(A)

	eigVal, eigVec = np.linalg.eig(A2)

	F = eigVec[:, np.argmin(eigVal)].reshape(3, 3)
	U, S, VT = np.linalg.svd(F)
	S[-1] = 0
	F = (U * S).dot(VT)

	return (F / F[-1, -1])

def main(images=None, readMatch='', output='./'):
	if images is None:
		images = ['a.jpg', 'b.jpg']

	imgA = cv2.imread(images[0])
	imgB = cv2.imread(images[1])

	if readMatch:
		kpA, desA, kpB, desB, matches, matchesMask = loadStereoMatchingResult(readMatch)
	else:
		kpA, desA, kpB, desB, matches, matchesMask = matchImages(imgA, imgB, qualityThreshold=0.65)
		saveStereoMatchingResult(kpA, desA, kpB, desB, matches, matchesMask, os.path.join(output, 'stereoMatch.pkl'))

	drawMatchedImages(imgA, kpA, imgB, kpB, matches, matchesMask)

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
