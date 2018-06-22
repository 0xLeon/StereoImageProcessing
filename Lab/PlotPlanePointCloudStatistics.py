import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

def plotToSubplot(axis, data, curveSetKeys, curveSetKey):
	cdata = dict(zip(curveSetKeys, ['b', 'g', 'r']))

	intervals = np.abs(np.array([st.t.interval(0.95, len(data[i])-1, loc=0, scale=st.sem(data[i])) for i in data])).T
	means = [np.mean(data[i]) for i in data]

	axis.errorbar(data.keys(), means, intervals, fmt='{:s}o-'.format(cdata[curveSetKey]), ms=5, elinewidth=0.8, barsabove=True, capsize=4, alpha=0.7, label=curveSetKey)
	axis.grid(True)
	axis.legend()

def main(dataFilePath):
	with open(dataFilePath, 'rb') as f:
		data = pickle.load(f)

	fig, ax = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(15, 7.5))
	resolutions = data.keys()

	for res in data:
		plotToSubplot(ax[0], data[res]['planeFitError'], resolutions, res)
		plotToSubplot(ax[1], data[res]['pointDensity'], resolutions, res)

	# plt.show()
	graphFilePath = os.path.join(os.path.dirname(dataFilePath), 'PointCloudProcessingData.png')
	fig.savefig(graphFilePath, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
	if len(sys.argv) < 2:
		sys.argv.append('./RealSense-D415-Data-01-Near/Laser/PointCloudProcessingData.pkl')

	main(sys.argv[1])
