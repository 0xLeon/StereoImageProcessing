import copy

import numpy

from PLYObject import PLYObject

print('Loading lucy.ply')

o1 = PLYObject('lucy.ply')
o2 = copy.deepcopy(o1)
o3 = copy.deepcopy(o1)
o4 = copy.deepcopy(o1)

print('Loaded lucy.ply')

# test translate
print('Translating')
o1.translate([100, 200, 300])
print('Translated')
o1.write('lucy_translate.ply')
print('Wrote lucy_translate.ply')

# test scale
print('Scaling')
o2.scale(0.5)
print('Scaled')
o2.write('lucy_scale.ply')
print('Wrote lucy_scale.ply')

# test rotateX
print('Rotating')
o3.rotateX(0.5 * numpy.pi)
print('Rotated')
o3.write('lucy_rotate.ply')
print('Wrote lucy_rotate.ply')

# test flipX
print('Flipping')
o4.flipX()
o4.flipFaces()
print('Flipped')
o4.write('lucy_flip.ply')
print('Wrote lucy_flip.ply')
