import argparse
import glob
import os
import pickle
import re
import time

import FilterPointCloud
import PLYObject

def main(searchFolder, filters=None, distanceReg=r'(\d+(?:\.\d+))m', resolutionReg=r'(\d+x\d+)'):
	parsedFilters = {}
	distanceReg = re.compile(distanceReg, re.I)
	resolutionReg = re.compile(resolutionReg, re.I)

	if filters is not None:
		for distance in filters:
			parsedFilters[distance] = FilterPointCloud.parseFilters(filters[distance])

	dataFitErrors = {}

	for folder in glob.glob(os.path.join(searchFolder, '*')):
		plyFiles = glob.glob(os.path.join(folder, '*.ply'))

		if not plyFiles:
			continue

		folderName = os.path.basename(os.path.dirname(plyFiles[0]))
		distance = float(distanceReg.search(folderName).group(1))
		resolution = resolutionReg.search(folder).group(1).lower()
		pFilters = parsedFilters[distance] if distance in parsedFilters else []

		print('Processing folder {:s}'.format(folderName))
		print('n = {:d}'.format(len(plyFiles)))

		fPlyFiles = [FilterPointCloud.filterPLYObject(PLYObject.PLYObject(plyFile), pFilters) for plyFile in plyFiles]

		try:
			planeFitErrors = [ply.fitPlane()[3] for ply in fPlyFiles]

			print('e = {!s}'.format(planeFitErrors))

			if resolution not in dataFitErrors:
				dataFitErrors[resolution] = {}

			if distance not in dataFitErrors[resolution]:
				dataFitErrors[resolution][distance] = planeFitErrors
			else:
				dataFitErrors[resolution][distance].extend(planeFitErrors)
		except RuntimeError:
			print('Unable to fit plane for every PLY file')

		print('')

	with open(os.path.join(searchFolder, 'PlaneFitErrorData.pkl'), 'wb') as f:
		pickle.dump(dataFitErrors, f)

	return dataFitErrors

def main_cli(args=None):
	parser = argparse.ArgumentParser()
	args = parser.parse_args(args)

	t0 = time.perf_counter()
	main('RealSense-D415-Data-01/Laser/', {
		2.5: [
			'x<1.1447',
			'y<0.58',
		]
	})
	t1 = time.perf_counter()
	print('Operation took {:.4f} s'.format(t1 - t0))

if __name__ == '__main__':
	main_cli()
