import argparse
import glob
import os
import pickle
import re
import time

import numpy as np

import FilterPointCloud
import PointCloudDensity
import PLYObject

def main(searchFolder, filters=None, distanceReg=r'(\d+(?:\.\d+))m', resolutionReg=r'(\d+x\d+)'):
	parsedFilters = {}
	distanceReg = re.compile(distanceReg, re.I)
	resolutionReg = re.compile(resolutionReg, re.I)

	if filters is not None:
		for distance in filters:
			parsedFilters[distance] = FilterPointCloud.parseFilters(filters[distance])

	pcProcessingData = {}

	for folder in glob.glob(os.path.join(searchFolder, '*')):
		plyFiles = glob.glob(os.path.join(folder, '*.ply'))

		if not plyFiles:
			continue

		folderName = os.path.basename(os.path.dirname(plyFiles[0]))
		distance = float(distanceReg.search(folderName).group(1))
		resolution = resolutionReg.search(folder).group(1).lower()
		pFilters = parsedFilters[distance] if distance in parsedFilters else []

		print('Processing folder {:s}'.format(folderName))
		print('Filters: {!s}'.format(pFilters))
		print('n = {:d}'.format(len(plyFiles)))

		fPlyFiles = FilterPointCloud.main(plyFiles, pFilters)

		try:
			planeFits = [ply.fitPlane() for ply in fPlyFiles]
			planeFitErrors = [plane[3] for plane in planeFits]
			planeFitErrorStds = [((ply.getVertices().T.dot(planeParams[:3]) + planeParams[3]) / np.linalg.norm(planeParams[:3])).std() for ply, planeParams in zip(fPlyFiles, planeFits)]
			pointDensities = [PointCloudDensity.getRealDensityFromPlane(ply, plane) for ply, plane in zip(fPlyFiles, planeFits)]

			print('e = {!s}'.format(planeFitErrors))
			print('d = {!s}'.format(pointDensities))

			if resolution not in pcProcessingData:
				pcProcessingData[resolution] = {
					'planeFitError': {},
					'planeFitErrorStd': {},
					'pointDensity': {},
				}

			if distance not in pcProcessingData[resolution]['planeFitError']:
				pcProcessingData[resolution]['planeFitError'][distance] = planeFitErrors
			else:
				pcProcessingData[resolution]['planeFitError'][distance].extend(planeFitErrors)

			if distance not in pcProcessingData[resolution]['planeFitErrorStd']:
				pcProcessingData[resolution]['planeFitErrorStd'][distance] = planeFitErrorStds
			else:
				pcProcessingData[resolution]['planeFitErrorStd'][distance].extend(planeFitErrorStds)

			if distance not in pcProcessingData[resolution]['pointDensity']:
				pcProcessingData[resolution]['pointDensity'][distance] = pointDensities
			else:
				pcProcessingData[resolution]['pointDensity'][distance].extend(pointDensities)
		except RuntimeError:
			print('Unable to fit plane for every PLY file')

		print('')

	with open(os.path.join(searchFolder, 'PointCloudProcessingData.pkl'), 'wb') as f:
		pickle.dump(pcProcessingData, f)

	return pcProcessingData

def main_cli(args=None):
	parser = argparse.ArgumentParser()
	args = parser.parse_args(args)

	t0 = time.perf_counter()
	main('RealSense-D415-Data-01-Near/Laser/', {
		2.5: [
			'x<1.1447',
			'y<0.58',
		]
	})
	main('RealSense-D415-Data-02-Near/Laser', {
		2.0: [
			'x>-1.03',
			'x<1.26',
		],
		2.5: [
			'x>-0.742',
			'x<1.26',
			'y>-0.88',
		],
	})
	t1 = time.perf_counter()
	print('Operation took {:.4f} s'.format(t1 - t0))

if __name__ == '__main__':
	main_cli()
