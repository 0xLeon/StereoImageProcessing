import sys

import matplotlib.pyplot as plt
import numpy as np

import FilterPointCloud
import PLYObject

if __name__ == '__main__':
	if len(sys.argv) != 2:
		raise ValueError('Invalid parameters')

	vertices = PLYObject.PLYObject(sys.argv[1]).getVertices().T
	data0 = np.unique(np.sort(vertices[:, 2]))
	data1 = np.diff(data0)

	fVertices = FilterPointCloud.filterGarbage(vertices)
	data2 = np.unique(np.sort(fVertices[:, 2]))
	data3 = np.diff(data2)

	fig, ax = plt.subplots(1, 2, sharex=True, sharey=False)
	fig.suptitle('Garbage Point Filtering', fontsize=16)
	ax[0].set_title('Without Garbage Filtering')
	ax[0].set_ylabel('Z values')
	ax[0].grid(True)
	ax[0].plot(data0)
	ax[0].plot(data1, '--')
	ax[1].set_title('With Garbage Filtering')
	ax[1].set_ylabel('Z values')
	ax[1].grid(True)
	ax[1].plot(data2)
	ax[1].plot(data3, '--')
	plt.get_current_fig_manager().window.state('zoomed')
	plt.show()
