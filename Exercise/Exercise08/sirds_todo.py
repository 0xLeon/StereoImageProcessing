import numpy
import cv2
import scipy.misc

def loadPattern(filename):
	img=cv2.imread(filename)
	return img,img.shape[1]

def loadNormalizedDepth(filename,width,whiteIsNear=True):
	img=cv2.imread(filename,cv2.IMREAD_GRAYSCALE).astype(float)
	if whiteIsNear: img*=-1
	#an vorgegebene Breite anpassen:
	height=int(img.shape[0]/img.shape[1]*width)
	img_scaled=scipy.misc.imresize(img,(height,width))
	#auf [0,1] normalisieren:
	img_scaled-=img_scaled.min()
	if not img_scaled.max()==0:
		img_scaled=img_scaled/img_scaled.max()
	#von [0,1] auf [0,2/3] transformieren, damit man nicht versehentlich in herkömmliche Betrachtungsweise (Disparität 0) verfällt:
	return img_scaled*2/3

def fill(w,h,pattern=None):
	if pattern is None:
		return numpy.random.randint(0,2,(h,w),dtype="uint8")*255
	else:
		overhang=numpy.tile(pattern,(h//pattern.shape[0]+1,w//pattern.shape[1]+1,1))
		return overhang[:h,:w]

def export(filename,img):
	cv2.imwrite(filename,img)

if __name__=="__main__":

	usePattern=True
	textureFile="moon_texture.png"
	depthFile="dragon.png"
	outFile="stereogram.jpg"

	if usePattern:
		pattern,shift=loadPattern(textureFile)
		width=shift*10
	else:
		shift=100
		pattern=None

	depthmap=loadNormalizedDepth(depthFile,width)
	height,_=depthmap.shape
	stereogram=fill(width+shift,height,pattern)
	for x in range(width):
		stereogram[:,y+shift]=? #TODO
	export(outFile,stereogram)
