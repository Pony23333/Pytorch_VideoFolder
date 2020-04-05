"""
This file is from https://github.com/jrosebr1/imutils
It is really impressive and a detailed tutorial can be found in
https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
"""

# import the necessary packages
from threading import Thread
import sys
import cv2
import time
import numpy as np

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue

# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue


class FileVideoStream:
    def __init__(self, path, size, length=10, queue_size=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.h = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.w = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.num_frames = self.stream.get(cv2.CAP_PROP_FRAME_COUNT)
        self.length = length
        self.size = size
        self.path = path
        # print('the video path is{}'.format(path))

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely
        rate = int(self.num_frames / self.length)
        count = 0
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                self.stream.set(cv2.CAP_PROP_POS_FRAMES, count * rate)
                ret, frame = self.stream.read()
                if ret:
                    # center crop the frame
                    i = int(round((self.h - self.size[0]) / 2.))
                    j = int(round((self.w - self.size[1]) / 2.))
                    frame = frame[i:-i, j:-j, :]
                    count += 1

                # if the `ret` boolean is `False`, then we have
                # reached the end of the video file
                if not ret:
                    print("Video {} is Skipped!".format(self.path))
                    self.stopped = True

                if count >= self.length:
                    self.stopped = True
                    print('loaded video {}'.format(self.path))

                # add the frame to the queue
                self.Q.put(frame)
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()