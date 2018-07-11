############################################################
# June 11th, 2018                                          # 
# RIPS USING AI                                            #
# Turning a Youtube Video to a set of images               #
############################################################
#install cv2
#https://www.codingforentrepreneurs.com/blog/install-opencv-3-for-python-on-mac
import cv2
import pafy
import numpy as np
import os
import PIL as pil
from PIL import Image

def vid_to_image(url, dst): 
    """
    Given a link to a youtube video as a string and a location 
    to save as a string, this function fills the location with images.
    Also resizes to 150 by 150 
    """
    height = 150
    width = 150
    vPafy = pafy.new(url)
    play = vPafy.getbest(preftype="webm")
    #start the video                                                               
    cap = cv2.VideoCapture(play.url)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( length )
    frame = 0
    while True:
        check, img = cap.read()
        if check:
            #resized = cv2.resize(img, height, width)
            img = cv2.resize(img, (height, width))
            cv2.imwrite(os.path.join(dst,"%d.jpg") %frame, img)
            frame += 1
        else: 
            break 
    cap.release()

#change URL here
url_example = 'https://www.youtube.com/watch?v=Qa_ZSRj0WM0'

#change destination here

dst_example = 'Desktop/RIPS/video_test/'

#run function 

vid_to_image(url_example, dst_example)
