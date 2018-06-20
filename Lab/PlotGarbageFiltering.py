import sys

import matplotlib.pyplot as plt
import numpy as np

import FilterPointCloud
import PLYObject

if __name__ == '__main__':
	if len(sys.argv) != 2:
		raise ValueError('Invalid parameters')

	vertices = PLYObject.PLYObject(sys.argv[1]).getVertices().T

	fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
	fig.suptitle('Garbage Point Filtering', fontsize=16)

	for axis in ax:
		axis.grid(True)
		axis.set_ylabel('Z values')

	data0 = np.unique(np.sort(vertices[:, 2]))
	data1 = np.diff(data0)

	ax[0].set_title('Without Garbage Filtering')
	ax[0].plot(data0)
	ax[0].plot(data1, '--')

	fVertices = FilterPointCloud.filterGarbageOld1(vertices)
	data0 = np.unique(np.sort(fVertices[:, 2]))
	data1 = np.diff(data0)

	ax[1].set_title('With Garbage Filtering v1')
	ax[1].plot(data0)
	ax[1].plot(data1, '--')

	fVertices = FilterPointCloud.filterGarbage(vertices)
	data0 = np.unique(np.sort(fVertices[:, 2]))
	data1 = np.diff(data0)

	ax[2].set_title('With Garbage Filtering v2')
	ax[2].plot(data0)
	ax[2].plot(data1, '--')

	plt.get_current_fig_manager().window.state('zoomed')
	plt.show()
