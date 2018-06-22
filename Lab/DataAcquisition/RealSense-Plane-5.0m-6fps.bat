@echo off
py GetDepthFrames.py --nimages=10 --resolution=424,240 --fps=6 --laser --output=data/Laser/RealSense-D415-Plane-5.0m-424x240-6fps-Laser/
py GetDepthFrames.py --nimages=10 --resolution=640,480 --fps=6 --laser --output=data/Laser/RealSense-D415-Plane-5.0m-640x480-6fps-Laser/
py GetDepthFrames.py --nimages=10 --resolution=1280,720 --fps=6 --laser --output=data/Laser/RealSense-D415-Plane-5.0m-1280x720-6fps-Laser/

py GetDepthFrames.py --nimages=10 --resolution=424,240 --fps=6 --output=data/NoLaser/RealSense-D415-Plane-5.0m-424x240-6fps-NoLaser/
py GetDepthFrames.py --nimages=10 --resolution=640,480 --fps=6 --output=data/NoLaser/RealSense-D415-Plane-5.0m-640x480-6fps-NoLaser/
py GetDepthFrames.py --nimages=10 --resolution=1280,720 --fps=6 --output=data/NoLaser/RealSense-D415-Plane-5.0m-1280x720-6fps-NoLaser/
