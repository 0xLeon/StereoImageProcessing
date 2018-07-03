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
	deltaImg = np.abs(imgL - imgR)

	for v in range(imgL.shape[0]):
		for u in range(imgL.shape[1]):
			for d in range(numDisp):
				C[v, u, d] = pixelCost(deltaImg, v, u, d)

	return C

def main(imgL, imgR, disparityRange=(0, 20), directions=8):
	imgL = cv2.imread(imgL)
	imgR = cv2.imread(imgR)

	imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
	imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

	dispRange = range(disparityRange[0], disparityRange[1])
	numDisp = len(dispRange)

	C = preCalculateCosts(imgL, imgR, numDisp)

	# TODO: maybe add support for 16 directions
	directionsMapping = {
		8: [
			(np.array([0, 1]), np.array(list(zip(range(imgL.shape[1]), [0] * imgL.shape[1])))),
			(np.array([-1, 1]), np.vstack((np.array(list(zip(range(imgL.shape[1]), [0] * imgL.shape[1]))), np.array(list(zip([imgL.shape[1] - 1] * (imgL.shape[0] - 1), range(1, imgL.shape[0] + 1))))))),
			(np.array([-1, 0]), np.array(list(zip([imgL.shape[1] - 1] * imgL.shape[0], range(imgL.shape[0]))))),
			(np.array([-1, -1]), np.vstack((np.array(list(zip(range(imgL.shape[1]), [imgL.shape[0] - 1] * imgL.shape[1]))), np.array(list(zip([imgL.shape[1] - 1] * (imgL.shape[0] - 1), range(imgL.shape[0] - 1))))))),
			(np.array([0, -1]), np.array(list(zip(range(imgL.shape[1]), [imgL.shape[0] - 1] * imgL.shape[1])))),
			(np.array([1, -1]), np.vstack((np.array(list(zip(range(imgL.shape[1]), [imgL.shape[0] - 1] * imgL.shape[1]))), np.array(list(zip([0] * (imgL.shape[0] - 1), range(imgL.shape[0] - 1))))))),
			(np.array([1, 0]), np.array(list(zip([0] * imgL.shape[0], range(imgL.shape[0]))))),
			(np.array([1, 1]), np.vstack((np.array(list(zip(range(imgL.shape[1]), [0] * imgL.shape[1]))), np.array(list(zip([0] * (imgL.shape[0] - 1), range(imgL.shape[0] - 1))))))),
		]
	}

	Lr = np.zeros((directions, imgL.shape[0], imgL.shape[1], numDisp))

	# TODO: make configurable
	P1 = 8
	P2 = 32

	for i, direction in enumerate(directionsMapping[directions]):
		p = direction[1].copy()

		Lr[i, p[:, 1], p[:, 0], :] += C[p[:, 1], p[:, 0], :].T

		p += direction[0]
		p = p[(p[:, 0] > -1) & (p[:, 0] < imgL.shape[1]) & (p[:, 1] > -2) & (p[:, 1] < imgL.shape[0])]

		while p.size > 0:
			prev = p - direction[0]

			for d in range(numDisp):
				# TODO: min() won't work, replace with actual working function call
				# TODO: np.argmin() probably won't work, check
				currLr = C[p[:, 1], p[:, 0], d] + min(
					Lr[i, prev[:, 1], prev[:, 0], d],
					Lr[i, prev[:, 1], prev[:, 0], d - 1] + P1,
					Lr[i, prev[:, 1], prev[:, 0], d + 1] + P1,
					np.argmin(Lr[i, prev[:, 1], prev[:, 0], :]) + P2
				) - np.argmin(Lr[i, prev[:, 1], prev[:, 0], :])

				Lr[i, p[:, 1], p[:, 0], d] += currLr

			p += direction[0]
			p = p[(p[:, 0] > -1) & (p[:, 0] < imgL.shape[1]) & (p[:, 1] > -2) & (p[:, 1] < imgL.shape[0])]

def main_cli():
	main('tsukuba_l.png', 'tsukuba_r.png')

if __name__ == '__main__':
	main_cli()