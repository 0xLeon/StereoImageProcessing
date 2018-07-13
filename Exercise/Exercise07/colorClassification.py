import argparse
import glob
import itertools
import os
import pathlib
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.svm

class TimeMeasurement(object):
	def __init__(self, operation, printStd=True):
		self._t0 = 0
		self._t1 = 0

		self._operation = operation
		self._printStd = printStd

	@property
	def delta(self):
		return self._t1 - self._t0

	def __enter__(self):
		self._t1 = 0
		self._t0 = time.perf_counter()

		return self

	def __exit__(self, *args):
		self._t1 = time.perf_counter()

		if self._printStd:
			print('Operation \'{!s}\' took {:.4f} s'.format(self._operation, self.delta))

def findImages(directory='.'):
	rawLabels = list(filter(os.path.isdir, os.listdir(directory)))
	labelEncoder = sklearn.preprocessing.LabelEncoder().fit(rawLabels)

	rawImageSet = [glob.glob(os.path.join(folder, 'image*')) for folder in rawLabels]

	rawImageSetLabels = (itertools.repeat(rawLabel, len(imageSet)) for imageSet, rawLabel in zip(rawImageSet, rawLabels))
	rawImageSetLabels = list(itertools.chain.from_iterable(rawImageSetLabels))

	imageSet = np.array(list(itertools.chain.from_iterable(rawImageSet)))
	imageSetLabels = labelEncoder.transform(rawImageSetLabels)

	return imageSet, imageSetLabels, labelEncoder

def extractFeature(image):
	img = cv2.imread(image, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

	if img is None:
		return None

	try:
		img = img.astype(np.float) / np.iinfo(img.dtype).max
	except ValueError:
		pass

	imgLab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2Lab)

	meanBgr = img.mean(axis=(0, 1))[np.newaxis, np.newaxis, :].astype(np.float32)
	meanLab = cv2.cvtColor(meanBgr, cv2.COLOR_BGR2Lab)
	meanHsv = cv2.cvtColor(meanBgr, cv2.COLOR_BGR2HSV)

	# Try to eliminate white background
	# This feature is actually unused due to poor performance on the dataset
	# Problem is probably lots of other stuff in the background
	meanBgrNoBackground = img[img.mean(axis=2) < 0.9, :].mean(axis=0)[np.newaxis, np.newaxis, :].astype(np.float32)
	meanLabNoBackground = cv2.cvtColor(meanBgrNoBackground, cv2.COLOR_BGR2Lab)
	meanHsvNoBackground = cv2.cvtColor(meanBgrNoBackground, cv2.COLOR_BGR2HSV)

	# Get the a*b* values for the center pixel
	# This feature is actually unused due to poor performance on the dataset
	# Problem is pobably lots of other stuff in the background and fruits not being centered
	centerPoint = (np.array(img.shape[:2]) / 2).astype(int)
	centerLab = imgLab[centerPoint[0], centerPoint[1]]

	return np.fromiter(itertools.chain(
		meanLab[0, 0, 1:],
		meanHsv[0, 0, :1],
		# meanLabNoBackground[0, 0, 1:],
		# meanHsvNoBackground[0, 0, :1],
		# centerLab[1:]
	), np.float)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def main():
	with TimeMeasurement('Find Images'):
		images, labels, encoder = findImages()

	with TimeMeasurement('Extract Features'):
		images = [extractFeature(image) for image in images]

	images_train, images_test, labels_train, labels_test = sklearn.model_selection.train_test_split(images, labels, test_size=0.33, random_state=42)

	classifier = sklearn.svm.SVC()

	with TimeMeasurement('Train'):
		classifier.fit(images_train, labels_train)

	with TimeMeasurement('Predict'):
		labels_predicted = classifier.predict(images_test)

	print('')

	print('Train Score: {:.2f}'.format(classifier.score(images_train, labels_train) * 100))
	print('Test Score: {:.2f}'.format(classifier.score(images_test, labels_test) * 100))

	cnf = sklearn.metrics.confusion_matrix(labels_test, labels_predicted)

	plt.figure()
	plot_confusion_matrix(cnf, encoder.classes_, False)
	plt.show()

	return

def main_cli():
	main()

if __name__ == '__main__':
	main_cli()
