#!/usr/bin/env python3

import argparse
import glob
import multiprocessing.dummy
import os
import sys

import cv2 as cv
import numpy as np

def splitfn(fn):
	path, fn = os.path.split(fn)
	name, ext = os.path.splitext(fn)

	return path, name, ext

def find_chessboard(imageFilename, patternsize, pattern, initDimensions, output=''):
	print('Processing {:s}...'.format(imageFilename))

	img = cv.imread(imageFilename, 0)

	if img is None:
		print('Failed to load {:s}'.format(imageFilename))
		return None

	assert initDimensions == img.shape, 'Size {:d} x {:d} ...'.format(img.shape[1], img.shape[0])

	found, corners = cv.findChessboardCorners(img, patternsize)

	if found:
		term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
		cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
	else:
		print('Chessboard not found')
		return None

	if output:
		vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
		cv.drawChessboardCorners(vis, patternsize, corners, found)
		name = splitfn(imageFilename)[1]
		outfile = os.path.join(output, '{:s}_chess.png'.format(name))
		cv.imwrite(outfile, vis)

	print('           {:s}... OK'.format(imageFilename))
	return (corners.reshape(-1, 2), pattern)

def find_chessboards(images, patternsize, squaresize, threads, output=''):
	pattern = np.zeros((np.prod(patternsize), 3), np.float32)
	pattern[:, :2] = np.indices(patternsize).T.reshape(-1, 2)
	pattern *= squaresize

	objPoints = []
	imgPoints = []
	chessboards = []
	dimensions = cv.imread(images[0], 0).shape[:2]

	if threads <= 1:
		chessboards = [find_chessboard(imageFilename, patternsize, pattern, dimensions, output) for imageFilename in images]
	else:
		print('Run with {:d} threads'.format(threads))

		def _find_chessboard(_imageFilename):
			return find_chessboard(_imageFilename, patternsize, pattern, dimensions, output)

		chessboards = multiprocessing.dummy.Pool(threads).map(_find_chessboard, images)

	chessboards = [board for board in chessboards if board is not None]
	for (corners, pattern) in chessboards:
		imgPoints.append(corners)
		objPoints.append(pattern)

	print('Processing done')
	return (imgPoints, objPoints, dimensions)

def calibrate(imgPoints, objPoints, dimensions):
	rms, camMatrix, distCoefs, _, _ = cv.calibrateCamera(objPoints, imgPoints, dimensions[::-1], None, None)

	print('')
	print('RMS:\t\t\t{:f}'.format(rms))
	print('Distortion Coefficients: {!s}'.format(distCoefs.ravel()))
	print('Camera Matrix:\n{!s}'.format(camMatrix))

	return (rms, camMatrix, distCoefs)

def undistort_image(imageFilename, camMatrix, distCoefs, output):
	name = splitfn(imageFilename)[1]
	imgOut = os.path.join(output, '{:s}_undistorted.png'.format(name))

	img = cv.imread(imageFilename)

	if img is None:
		return None

	dimensions = img.shape[:2]
	newCamMatrix, roi = cv.getOptimalNewCameraMatrix(camMatrix, distCoefs, dimensions[::-1], 1, dimensions[::-1])
	x, y, width, height = roi

	imgUndist = cv.undistort(img, camMatrix, distCoefs, None, newCamMatrix)[y:y+height, x:x+width]

	print('Writing undistorted image to {:s}'.format(imgOut))
	cv.imwrite(imgOut, imgUndist)

def undistort_images(images, camMatrix, distCoefs, threads, output):
	print('')
	print('Undistorting images')

	if threads <= 1:
		for imageFilename in images:
			undistort_image(imageFilename, camMatrix, distCoefs, output)
	else:
		print('Run with {:d} threads'.format(threads))

		def _undistort_image(_imageFilename):
			return undistort_image(_imageFilename, camMatrix, distCoefs, output)

		multiprocessing.dummy.Pool(threads).map(_undistort_image, images)

def main(images, patternsize=(9,6), squaresize=1.0, savecalib=False, readcalib='', debugoutput=False, threads=4, output='./output/'):
	threads = threads if threads > 0 else 1
	squaresize = squaresize if squaresize > 0 else 1.0
	output = os.path.abspath(output)

	if output and not os.path.isdir(output):
		os.makedirs(output)

	if readcalib:
		calibData = np.load(readcalib)
		camMatrix = calibData['camMatrix']
		distCoefs = calibData['distCoefs']
	else:
		imgPoints, objPoints, dimensions = find_chessboards(images, patternsize, squaresize, threads, output if debugoutput else '')
		_, camMatrix, distCoefs = calibrate(imgPoints, objPoints, dimensions)

	if savecalib:
		np.savez(
			os.path.join(output, 'calibData.npz'),
			camMatrix=camMatrix,
			distCoefs=distCoefs,
		)

	undistort_images(images, camMatrix, distCoefs, threads, output)
	cv.destroyAllWindows()

def main_cli(args=None):
	if args is None:
		args = sys.argv[1:]

	parser = argparse.ArgumentParser()
	parser.add_argument('--threads', default=4, type=int, help='Number of threads used for image processing, defaults to 4')
	parser.add_argument('--patternsize', default='9,6', help='Number of grid points in both dimensions, comma separated, defaults to 9,6')
	parser.add_argument('--squaresize', default=1.0, type=float, help='Size of one square in the chessboard [mm]')
	parser.add_argument('--savecalib', default=False, action='store_true', help='Save camera matrix and distortion coefficients to file sytem')
	parser.add_argument('--readcalib', default='', help='Instead of generating calibration data from images, read this file saved before via --savecalib')
	parser.add_argument('--debugoutput', default=False, action='store_true', help='Save additional data to file system along with undistorted images')
	parser.add_argument('--output', default='./output/', help='Output target folder, will be created if not existing; defaults to ./output/')
	parser.add_argument('images', nargs='+', help='Image files, can be a single glob pattern or list of files; all images need to be the same size and orientation')
	args = parser.parse_args(args)

	images = args.images if len(args.images) > 1 else glob.glob(args.images[0])
	patternsize = tuple(map(int, args.patternsize.split(',')))

	main(images, patternsize, args.squaresize, args.savecalib, args.readcalib, args.debugoutput, args.threads, args.output)

if __name__ == '__main__':
	main_cli()
