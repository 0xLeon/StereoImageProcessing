import argparse

import cv2
import numpy as np

def pixelCost(deltaImg, v, u, d):
	if u < 0:
		return 0
	elif (u - d) < 0:
		return 0

	# TODO: calculate actual cost
	return None

def preCalculateCosts(imgL, imgR, numDisp):
	C = np.zeros((imgL.shape[0], imgL.shape[1], numDisp))

def main(imgL, imgR, disparityRange=(0, 20), directions=8):
	imgL = cv2.imread(imgL)
	imgR = cv2.imread(imgR)

	imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
	imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

	imgL = imgL.astype(float) / np.iinfo(imgL.dtype).max
	imgR = imgR.astype(float) / np.iinfo(imgR.dtype).max

	C = preCalculateCosts(imgL, imgR, len(range(disparityRange[0], disparityRange[1])))

def main_cli():
	main('tsukuba_l.png', 'tsukuba_r.png')

if __name__ == '__main__':
	main_cli()
