This is a readme for the following supporting code files for this project-

-extractFrames.py
-image_resizer.m
-rename_files.py

The first file extractFrames.py is used for converting the video data into framed
image data by placing this file in the directory containing the video files and 
executing it will generate jpeg images of frames for all video files in the directory.
This file is executed in python using the OpenCV library which is required to run this code.

The second file image_resizer.m which is used for scaling the image(Frames) to a smaller
size of 64*48provided the image is bigger than those dimensions and this is carried out
to equalize the input provided to the model to a standard size. The requirement to run this
file is the path of the directory where the frames are stored is to be given to the code where
indicated. The output images are exported to the same location where the original images are placed. This file is executed in Matlab and that is a requirement to run the code. 

The third file is rename_files.py takes the output images from file two and based on its label
it renames the image as '_v' and '_nv' which represents violence and non-violence images respectively
and stores them in the same directory, as a precaution the file should have a backup copy as when it is
executed for a directory the code deletes all previous files in it including the code itself.
This file is developed in python and that is a requirement to run the code.