# Pytorch_VideoFolder

## About
A Dataset using opencv that can sample the videos and handle videos with different size on-the-fly.

## Requirement
Currently only run and test on torch.\_\_version\_\_ == '1.4.0' and cv2.\_\_version\_\_ == '4.2.0'. 
Note that for different version of cv2 parts of the code could be different. An example for 
older version could be find at https://stackoverflow.com/questions/33650974/opencv-python-read-specific-frame-using-videocapture.

## To do
The video loading could be accelarated by using another thread. This is of great importance for application using
large scale dataset like kinetics. There is a detailed tutorial explaing why and how 
https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
