#!/usr/bin/env python3

import argparse
import itertools

import cv2 as cv
import numpy as np
import plyfile

def readCameraData(cameraName):
	return {
		'projection': np.genfromtxt('{:s}_K.txt'.format(cameraName)),
		'rotation': np.genfromtxt('{:s}_R.txt'.format(cameraName)),
		'translation': np.genfromtxt('{:s}_T.txt'.format(cameraName)),
		'distortion': np.genfromtxt('{:s}_D.txt'.format(cameraName)),
	}

def generateDisparityMap():
	pass

def generatePointcloud():
	pass

def main():
	pass

if __name__ == '__main__':
	main()
