# Copyright (C) 2018 Stefan Hahn <stefan.hahn@hu-berlin.de>, Andre Niendorf <niendora@hu-berlin.de>

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pyrealsense2 as rs


def intrinsics_to_dict(intrinsics: rs.intrinsics):
	return {
		'ppx': intrinsics.ppx,
		'ppy': intrinsics.ppy,
		'fx': intrinsics.fx,
		'fy': intrinsics.fy,
		'width': intrinsics.width,
		'height': intrinsics.height,
		'model': int(intrinsics.model),
		'coeffs': intrinsics.coeffs,
	}

def dict_to_intrinsics(d: dict):
	intrinsics = rs.intrinsics()
	intrinsics.ppx = d['ppx']
	intrinsics.ppy = d['ppy']
	intrinsics.fx = d['fx']
	intrinsics.fy = d['fy']
	intrinsics.width = d['width']
	intrinsics.height = d['height']
	intrinsics.model = rs.distortion(d['model'])
	intrinsics.coeffs = d['coeffs']

	return intrinsics

def getDepthFrames(nImages=10, resolution=(1280,720), fps=30, sleepTime=5, laser=False, outputFolder='./'):
	if nImages < 1:
		raise ValueError()

	if sleepTime < 0:
		raise ValueError()

	if not os.path.isdir(outputFolder):
		os.makedirs(outputFolder)

	pointcloud = rs.pointcloud() # type: rs.pointcloud
	points = rs.points() # type: rs.points

	config = rs.config() # type: rs.config
	config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, fps)
	config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, fps)

	align = rs.align(rs.stream.depth) # type: rs.align

	pipeline = rs.pipeline() # type: rs.pipeline
	profile = pipeline.start(config) # type: rs.pipeline_profile
	device = profile.get_device() # type: rs.device
	depthSensor = device.first_depth_sensor() # type: rs.depth_sensor

	if depthSensor.supports(rs.option.emitter_enabled):
		depthSensor.set_option(rs.option.emitter_enabled, 1.0 if laser else 0.0)
	elif laser:
		raise EnvironmentError('Device does not support laser')

	np.savez(
		os.path.join(outputFolder, 'depth.scale.npz'),
		depthscale=depthSensor.get_depth_scale(),
	)

	counter = 0

	print('Writing depth intrinsics')
	print('')
	while True:
		frames = pipeline.wait_for_frames() # type: rs.frameset
		alignedFrames = align.process(frames) # type: rs.frameset
		depth = alignedFrames.get_depth_frame() # type: rs.depth_frame

		if depth:
			depthIntrinsics = intrinsics_to_dict(depth.get_profile().as_video_stream_profile().get_intrinsics())

			with open(os.path.join(outputFolder, 'depth.intrinsics.pkl'), 'wb') as f:
				pickle.dump(depthIntrinsics, f)

			break

	while True:
		print('Waiting for frame {:d}'.format(counter))
		frames = pipeline.wait_for_frames() # type: rs.frameset
		alignedFrames = align.process(frames) # type: rs.frameset
		depth = alignedFrames.get_depth_frame() # type: rs.depth_frame
		color = alignedFrames.get_color_frame() # type: rs.video_frame

		if not depth:
			print('No depth data, retry...')
			continue

		if not color:
			print('No color data, retry...')
			continue

		print('Depth data available')
		depthImage = np.asanyarray(depth.get_data())

		print('Writing raw numpy file')
		np.savez(os.path.join(outputFolder, '{:d}.depth.image.npz'.format(counter)), depthImage=depthImage)

		print('Generating point cloud')
		pointcloud.map_to(color)
		points = pointcloud.calculate(depth) # type: rs.points

		print('Writing point cloud to PLY file')
		points.export_to_ply(os.path.join(outputFolder, '{:d}.pointcloud.ply'.format(counter)), color)

		counter = counter + 1

		if counter >= nImages:
			break

		print('Sleeping before next frame')
		print('')
		time.sleep(sleepTime)

	pipeline.stop()

def main(args=None):
	if args is None:
		args = sys.argv[1:]

	parser = argparse.ArgumentParser()
	parser.add_argument('--nimages', type=int, default=10, help='Number of depth frames to capture')
	parser.add_argument('--resolution', default='1280,720', help='Sensor resolution, defaults to to 1280,720')
	parser.add_argument('--fps', type=int, default=30, help='Sensor framerate, defaults to to 30')
	parser.add_argument('--sleeptime', type=int, default=5, help='Time to sleep between capturing depth frames [s]')
	parser.add_argument('--laser', default=False, action='store_true', help='Activate laser, if supported')
	parser.add_argument('--output', default='./', help='Output path, defaults to current working dir')
	args = parser.parse_args(args)

	resolution = (*map(int, args.resolution.split(',')),)
	getDepthFrames(args.nimages, resolution, args.fps, args.sleeptime, args.laser, args.output)

if __name__ == '__main__':
	main()
