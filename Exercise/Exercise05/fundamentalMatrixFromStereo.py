import argparse

import cv2
import numpy as np

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

def main(images=None):
	if images is None:
		images = ['a.jpg', 'b.jpg']

	imgA = cv2.imread(images[0])
	imgB = cv2.imread(images[1])

	kpA, desA, kpB, desB, matches, matchesMask = matchImages(imgA, imgB, qualityThreshold=0.65)

def main_cli(args=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('images', default=['a.jpg', 'b.jpg'], nargs='*')
	args = parser.parse_args(args)

	if len(args.images) != 2:
		parser.error('Specifiy exactly two images!')

	main(args.images)

if __name__ == '__main__':
	main_cli()
