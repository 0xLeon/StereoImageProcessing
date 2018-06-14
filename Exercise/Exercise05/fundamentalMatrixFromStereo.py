import argparse

import cv2
import numpy as np

def main(images):
	pass

def main_cli(args=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('images', default=['a.jpg', 'b.jpg'], nargs='*')
	args = parser.parse_args(args)

	if len(args.images) != 2:
		parser.error('Specifiy exactly two images!')

	main(args.images)

if __name__ == '__main__':
	main_cli()
