# Pytorch_VideoFolder

## About
A Dataset using opencv that can sample the videos and handle videos with different size on-the-fly. This Dataset can be used like ImageFolder where names of each file determines the class for the videos underneath. For the fasterVideoLoader, please note that my
code is based on the tutorial and code provided by https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/. Due to the Queue structure in Python 3 is different from that in Python 2, the actual performance can be worse. The 
filevideostream.py is based on https://github.com/jrosebr1/imutils but changes are also made as for this VideoFolder Dataset we only
want to sample few frames thus the uage of Queue is different.

## Requirement
Currently only run and test on torch.\_\_version\_\_ == '1.4.0' and cv2.\_\_version\_\_ == '4.2.0'. 
Note that for different version of cv2 parts of the code could be different. An example for 
older version could be find at https://stackoverflow.com/questions/33650974/opencv-python-read-specific-frame-using-videocapture.

## To do
The video loading could be accelarated by using another thread. But a lot of work need to be done for Queue structure in Python3.


## Update
As currently I'm solving a video classfication task where the data I got for each category is extremely inbalanced,
I modified this videofolder to give it ability of handling unbalanced cases. Generally, the idea is get the size of each category
and find the minimum size. Then a threshold can be set as p*min_size where p is the portion e.g. 1.2. If a dataset contains more files
than this threshold, only a certain number of files will be randomly (without replacement) loaded to the dataset. 

## Update 2
For action recognition problem, there are two cases from the perspective of dataset that is important for the classification result. The first one is when the video is two short, there can be no enough frame to sample. Thus I made the
new_video_loader to sample frames looply if there such case happens. This made the smapled frames a perodic motion. The second case is that the videos can also be really long, making uniform sample from the beginning to end 
meaningfulless as the action is not consistent any more. So I changed the sample strategy for this cases: if the video is longer than 4s, sample uniformly at the middle 4s.

## Update 3
I also added a VisualFolder,py file for visualize the statistical information of the dataset and the classifcation accuracy by class.