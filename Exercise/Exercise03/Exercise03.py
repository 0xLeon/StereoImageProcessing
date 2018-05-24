from PLYObject import PLYObject

from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
	p = PLYObject('sphere.ply')
	print(p.getVertices().shape)

	sphereInfo = p.fitSphere()
	print(sphereInfo)

	sphere = PLYObject.from_sphere(sphereInfo[:3], sphereInfo[3])
	print(sphere.getVertices().shape)
	sphere.write('sphere_generated.ply')
