
"""
This code extracts frames from all video files in a folder provided this code is
executed in the same directory.
The frames are directly extracted to the folder.

Author- Varun Mantri (vm9324@g.rit.edu)
      - Sanket Sheth (sas6792@g.rit.edu)
"""



import os 



def frames():
    '''
    This function extracts frames from all video file in the current directory where this code is placed
    '''
    path=os.getcwd() #Get path for current directory
    arr=[path + "\\" +files for files in next(os.walk(path))[2]] #Modify path name
    
    lst=[files for files in arr]    #For all files in directory 
    
    import cv2
    
    for p in lst: #For all images in the directory
        vidcap = cv2.VideoCapture(p) #Video file object for each video file in the directory
        success,image = vidcap.read() #Read video data
        count = 0
        success = True
        while success:  #For each frame
          success,image = vidcap.read() #Read frame
          if(success):
              print('Read a new frame: ', success)
              cv2.imwrite(p+"frame%d.jpg" % count, image)     # Save frame as JPEG file
              count += 1
        print(count) #Total number of frames per video

frames()