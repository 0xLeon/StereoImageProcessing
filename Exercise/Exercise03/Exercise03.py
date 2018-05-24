from PLYObject import PLYObject

p = PLYObject('sphere.ply')
print(p.getVertices().shape)

sphereInfo = p.fitSphere()
print(sphereInfo)

sphere = PLYObject.from_sphere(sphereInfo)
print(sphere.getVertices().shape)
sphere.write('sphere_generated.ply')

print('')

p = PLYObject('plane.ply')
print(p.getVertices().shape)

planeInfo = p.fitPlane()
print(planeInfo)

plane = PLYObject.from_plane(planeInfo)
print(plane.getVertices().shape)
plane.write('plane_generated.ply')
