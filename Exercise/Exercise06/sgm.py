import argparse
import decimal

import cv2
import matplotlib.pyplot as plt
import numpy as np

def pixelCostSimple(v, u, d, imgL, imgR):
	if u < 0:
		return 0
	elif (u - d) < 0:
		return 0

	return np.abs(imgL[v, u] - imgR[v, u - d])

def pixelCostGithub(row, leftCol, rightCol, imgL, imgR):
	leftValue = 0
	rightValue = 0
	beforeRightValue = 0
	afterRightValue = 0
	rightValueMinus = 0
	rightValuePlus = 0
	rightValueMin = 0
	rightValueMax = 0

	if leftCol >= 0:
		leftValue = imgL[row, leftCol]

	if rightCol >= 0:
		rightValue = imgR[row, rightCol]


	if rightCol > 0:
		beforeRightValue = imgR[row, rightCol - 1]
	else:
		beforeRightValue = rightValue

	if rightCol > 0 and (rightCol + 1) < imgR.shape[1]:
		afterRightValue = imgR[row, rightCol + 1]
	else:
		afterRightValue = rightValue

	rightValueMinus = int(decimal.Decimal((rightValue + beforeRightValue) / 2.0).quantize(0, decimal.ROUND_HALF_UP))
	rightValuePlus = int(decimal.Decimal((rightValue + afterRightValue) / 2.0).quantize(0, decimal.ROUND_HALF_UP))

	rightValueMin = np.min([rightValue, rightValueMinus, rightValuePlus])
	rightValueMax = np.max([rightValue, rightValueMinus, rightValuePlus])

	return np.max([0, leftValue - rightValueMax, rightValueMin - leftValue])

def preCalculateCostsSimple(imgL, imgR, numDisp):
	C = np.zeros((imgL.shape[0], imgL.shape[1], numDisp))

	for v in range(imgL.shape[0]):
		for u in range(imgL.shape[1]):
			for d in range(numDisp):
				C[v, u, d] = pixelCostSimple(v, u, d, imgL, imgR)

	return C

def preCalculateCostsGithub(imgL, imgR, numDisp):
	C = np.zeros((imgL.shape[0], imgL.shape[1], numDisp))

	for v in range(imgL.shape[0]):
		for u in range(imgL.shape[1]):
			for d in range(numDisp):
				C[v, u, d] = np.min([
					pixelCostGithub(v, u, u - d, imgL, imgR),
					pixelCostGithub(v, u - d, u, imgR, imgL),
				])

	return C

def preCalculateCosts(imgL, imgR, numDisp):
	imgLShifted = imgL.copy()

	C = np.zeros((imgL.shape[0], imgL.shape[1], numDisp))
	C[:, :, 0] = np.abs(imgR - imgLShifted)

	for d in range(1, numDisp):
		imgLShifted[:, :-1] = imgLShifted[:, 1:]
		imgLShifted[:, -1:] = 0

		C[:, :, d] = np.abs(imgR - imgLShifted)

	return C

def sgm(imgL, imgR, p1, p2, disparityRange, directions=8):
	imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY).astype(np.float)
	imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY).astype(np.float)

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

	for i, direction in enumerate(directionsMapping[directions]):
		p = direction[1].copy()

		Lr[i, p[:, 1], p[:, 0], :] += C[p[:, 1], p[:, 0], :]

		p += direction[0]
		p = p[(p[:, 0] > -1) & (p[:, 0] < imgL.shape[1]) & (p[:, 1] > -2) & (p[:, 1] < imgL.shape[0])]

		while p.size > 0:
			prev = p - direction[0]
			minPrevD = Lr[i, prev[:, 1], prev[:, 0], :].min(axis=1)

			for d in range(numDisp):
				# TODO: check if wrap-around if out-of-range disparities is correct
				currLr = C[p[:, 1], p[:, 0], d] + np.amin([
					Lr[i, prev[:, 1], prev[:, 0], d],
					Lr[i, prev[:, 1], prev[:, 0], d - 1] + p1,
					Lr[i, prev[:, 1], prev[:, 0], (d + 1) % numDisp] + p1,
					minPrevD + p2,
				], axis=0) - minPrevD

				Lr[i, p[:, 1], p[:, 0], d] += currLr

			p += direction[0]
			p = p[(p[:, 0] > -1) & (p[:, 0] < imgL.shape[1]) & (p[:, 1] > -2) & (p[:, 1] < imgL.shape[0])]

	S = Lr.sum(axis=0)
	dispImage = S.min(axis=2)

	return dispImage

def main(imgLPath, imgRPath, p1, p2, disparityRange, directions=8):
	imgL = cv2.imread(imgLPath)
	imgR = cv2.imread(imgRPath)

	dispImage = sgm(imgL, imgR, p1, p2, disparityRange, directions)

	plt.imshow(dispImage)
	plt.savefig('depth.png')

def main_cli():
	# TODO: add CLI arguments
	main('tsukuba_l.png', 'tsukuba_r.png', 8, 32, (0, 20), 8)

if __name__ == '__main__':
	main_cli()
