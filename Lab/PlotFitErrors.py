import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

if __name__ == '__main__':
	with open('DataFitErrors.pkl', 'rb') as f:
		data = pickle.load(f)

	data = data['1280x720']
	intervals = np.abs(np.array([st.t.interval(0.95, len(data[i])-1, loc=0, scale=st.sem(data[i])) for i in data])).T
	means = [np.mean(data[i]) for i in data]

	plt.errorbar(data.keys(), means, intervals, fmt='o-', ecolor='k', elinewidth=0.7, barsabove=True, capsize=2.5)
	plt.grid(True)
	plt.show()
