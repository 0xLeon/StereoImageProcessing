# Copyright (C) 2018 Stefan Hahn <stefan.hahn@hu-berlin.de>, Andre Niendorf <niendora@hu-berlin.de>

import pyrealsense2 as rs

def main():
	for device in rs.context().query_devices():
		name = device.get_info(rs.camera_info.name).replace(' ', '_')

		for sensor in device.query_sensors(): # type: rs.sensor
			filename = '{:s}__{:s}.txt'

			if sensor.is_depth_sensor():
				filename = filename.format(name, 'Depth')
				sensor = sensor.as_depth_sensor()
			# elif sensor.is_roi_sensor():
			# 	filename = filename.format(name, 'ROI')
			# 	sensor.as_roi_sensor()
			else:
				filename = filename.format(name, 'Color')

			with open(filename, 'w') as f:
				for profile in sensor.get_stream_profiles(): # type: rs.stream_profile
					if profile.is_video_stream_profile():
						profile = profile.as_video_stream_profile() # type: rs.video_stream_profile

						print('{:s}({:d}) {:d}x{:d} @ {:d}fps {:s}'.format(
							str(profile.stream_type()).split('.')[-1].title(),
							profile.stream_index(),
							profile.width(),
							profile.height(),
							profile.fps(),
							str(profile.format()).split('.')[-1].upper(),
						), file=f)
					else:
						print('{:s}({:d}) @ {:d}fps {:s}'.format(
							str(profile.stream_type()).split('.')[-1].title(),
							profile.stream_index(),
							profile.fps(),
							str(profile.format()).split('.')[-1].upper(),
						), file=f)

if __name__ == '__main__':
	main()
