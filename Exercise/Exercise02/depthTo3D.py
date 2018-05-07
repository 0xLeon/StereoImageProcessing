#!/usr/bin/env python3

import argparse
import sys
import time

import cv2 as cv
import numpy as np
import plyfile as ply

def depthMapMatrix(fx, fy, cx, cy, imgDepth, imgColor, skipInvalid):
	intrinsicCamMatrix = np.array([[fx, 0, cx],
								   [0, fy, cy],
								   [0, 0, 1]], dtype=np.float)

	# Matrix could also be directly written as:
	# [[1/fx	0		-cx/fx]
	#  [0		1/fy	-cy/fy]
	#  [0		0		1	  ]]
	invIntrinsicCamMatrix = np.linalg.inv(intrinsicCamMatrix)

	u = np.arange(imgDepth.shape[1])
	v = np.arange(imgDepth.shape[0])
	uu, vv = np.meshgrid(v, u)

	vertices = np.array([vv, uu, imgDepth.T]).T
	verticesShape = vertices.shape
	vertices = vertices.reshape((verticesShape[0] * verticesShape[1], verticesShape[2]))

	imgColor = imgColor.view()
	colorShape = imgColor.shape
	imgColor.shape = (colorShape[0] * colorShape[1], verticesShape[2])

	verticesZ = np.repeat(vertices[:, 2], 2).reshape((-1, 2))
	verticesZ[verticesZ == 0.0] = 1.0
	vertices[:, :2] *= verticesZ

	keepInvalid = not skipInvalid
	vertices = vertices.dot(invIntrinsicCamMatrix.T)
	vertices = [(*v, *c[::-1]) for v, c in zip(vertices, imgColor) if keepInvalid or v[2] != 0.0]

	return vertices

def depthMapLoop(fx, fy, cx, cy, imgDepth, imgColor, skipInvalid):
	vertices = []

	for v in range(imgDepth.shape[0]):
		for u in range(imgDepth.shape[1]):
			z = float(imgDepth[v, u])

			if skipInvalid and z == 0.0:
				continue

			x = ((u - cx) * z) / fx
			y = ((v - cy) * z) / fy

			vertices.append((x, y, z, *imgColor[v, u][::-1]))

	return vertices

def main(args=None):
	if args is None:
		args = sys.argv[1:]

	_mode_to_func = {
		'matrix': depthMapMatrix,
		'loop': depthMapLoop,
	}

	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='matrix', choices=[*_mode_to_func.keys()])
	parser.add_argument('--fx', type=float, default=5.8818670481438744e+02)
	parser.add_argument('--fy', type=float, default=5.8724220649505514e+02)
	parser.add_argument('--cx', type=float, default=3.1076280589210484e+02)
	parser.add_argument('--cy', type=float, default=2.2887144980135292e+02)
	parser.add_argument('--skipinvalid', default=False, action='store_true')
	parser.add_argument('DEPTHFILE', nargs='?', default='depth.tif')
	parser.add_argument('COLORFILE', nargs='?', default='color.tif')
	parser.add_argument('PLYFILE', nargs='?', default='output.ply')
	args = parser.parse_args(args)

	print('Reading depth channel file {:s}'.format(args.DEPTHFILE))
	imgDepth = cv.imread(args.DEPTHFILE, cv.IMREAD_UNCHANGED)

	if imgDepth is None:
		raise IOError('Unable to read file {:s}'.format(args.DEPTHFILE))

	print('Reading color channel file {:s}'.format(args.COLORFILE))
	imgColor = cv.imread(args.COLORFILE, cv.IMREAD_UNCHANGED)

	if imgColor is None:
		raise IOError('Unable to read file {:s}'.format(args.COLORFILE))

	if imgDepth.shape[:2] != imgColor.shape[:2]:
		raise ValueError('Dimensions of depth channel and color channel mismatch: {} != {}'.format(imgDepth.shape[:2], imgColor.shape[:2]))

	print('Operation mode {:s}'.format(args.mode))
	print('{:s} invalid depth values'.format('Skipping' if args.skipinvalid else 'Including'))

	t0 = time.perf_counter()
	vertices = _mode_to_func[args.mode](
		args.fx,
		args.fy,
		args.cx,
		args.cy,
		imgDepth,
		imgColor,
		args.skipinvalid,
	)
	t1 = time.perf_counter()
	print('Took {:.4f} s'.format(t1 - t0))
	print('{:d} vertices'.format(len(vertices)))

	plyVertices = np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
	plyVertices = ply.PlyElement.describe(plyVertices, 'vertex')

	print('Writing PLY data to {:s}'.format(args.PLYFILE))
	ply.PlyData([plyVertices]).write(args.PLYFILE)

	return 0


if __name__ == '__main__':
	main()
