import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

def plotToSubplot(axis, data, curveSetKeys, curveSetKey):
	cdata = dict(zip(curveSetKeys, ['b', 'g', 'r']))

	intervals = np.abs(np.array([st.t.interval(0.95, len(data[i])-1, loc=0, scale=st.sem(data[i])) for i in data])).T
	means = [np.mean(data[i]) for i in data]

	axis.errorbar(data.keys(), means, intervals, fmt='{:s}o-'.format(cdata[curveSetKey]), ms=5, ecolor='r', elinewidth=0.8, barsabove=True, capsize=3, alpha=0.4, label=curveSetKey)
	axis.grid(True)
	axis.legend()

if __name__ == '__main__':
	with open('./RealSense-D415-Data-01-Near/Laser/PointCloudProcessingData.pkl', 'rb') as f:
		data = pickle.load(f)

	fig, ax = plt.subplots(1, 2, sharex=True, sharey=False)
	resolutions = data.keys()

	for res in data:
		plotToSubplot(ax[0], data[res]['planeFitError'], resolutions, res)
		plotToSubplot(ax[1], data[res]['pointDensity'], resolutions, res)

	plt.show()
