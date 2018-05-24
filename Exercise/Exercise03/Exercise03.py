import time

from PLYObject import PLYObject

print('Loading sphere')
p = PLYObject('sphere.ply')
print('Number of vertices: {:d}'.format(p.getVertices().shape[1]))

print('Fitting sphere')
t0 = time.perf_counter()
sphereInfo = p.fitSphere()
t1 = time.perf_counter()
print('Took {:.4f} s'.format(t1 - t0))
print('Fitted sphere: {!s}'.format(sphereInfo))

print('Generating sphere vertices')
t0 = time.perf_counter()
sphere = PLYObject.from_sphere(sphereInfo)
t1 = time.perf_counter()
print('Took {:.4f} s'.format(t1 - t0))
print('Number of vertices: {:d}'.format(sphere.getVertices().shape[1]))
print('Writing generated sphere vertices')
sphere.write('sphere_generated.ply')

print('')

print('Loading sphere2')
p = PLYObject('sphere2.ply')
print('Number of vertices: {:d}'.format(p.getVertices().shape[1]))

print('Fitting sphere')
t0 = time.perf_counter()
sphereInfo = p.fitSphere()
t1 = time.perf_counter()
print('Took {:.4f} s'.format(t1 - t0))
print('Fitted sphere2: {!s}'.format(sphereInfo))

print('Generating sphere2 vertices')
t0 = time.perf_counter()
sphere = PLYObject.from_sphere(sphereInfo)
t1 = time.perf_counter()
print('Took {:.4f} s'.format(t1 - t0))
print('Number of vertices: {:d}'.format(sphere.getVertices().shape[1]))
print('Writing generated sphere2 vertices')
sphere.write('sphere2_generated.ply')

print('')

print('Loading plane')
p = PLYObject('plane.ply')
print('Number of vertices: {:d}'.format(p.getVertices().shape[1]))

print('Fitting plane')
t0 = time.perf_counter()
planeInfo = p.fitPlane()
t1 = time.perf_counter()
print('Took {:.4f} s'.format(t1 - t0))
print('Fitted plane: {!s}'.format(planeInfo))

print('Generating plane vertices')
t0 = time.perf_counter()
plane = PLYObject.from_plane(planeInfo)
t1 = time.perf_counter()
print('Took {:.4f} s'.format(t1 - t0))
print('Number of vertices: {:d}'.format(plane.getVertices().shape[1]))
print('Writing generated plane vertices')
plane.write('plane_generated.ply')
