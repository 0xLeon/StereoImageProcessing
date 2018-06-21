import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

if __name__ == '__main__':
	with open('./RealSense-D415-Data-01/Laser/PlaneFitErrorData.pkl', 'rb') as f:
		data = pickle.load(f)

	cdata = dict(zip(data.keys(), ['b', 'g', 'r']))

	for res in data:
		intervals = np.abs(np.array([st.t.interval(0.95, len(data[res][i])-1, loc=0, scale=st.sem(data[res][i])) for i in data[res]])).T
		means = [np.mean(data[res][i]) for i in data[res]]

		plt.errorbar(data[res].keys(), means, intervals, fmt='{:s}o-'.format(cdata[res]), ms=5, ecolor='r', elinewidth=0.8, barsabove=True, capsize=3, alpha=0.4, label=res)

	plt.grid(True)
	plt.legend()
	plt.show()
