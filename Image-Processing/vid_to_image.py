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
    """
    if not os.path.exists(dst):
        os.makedirs(dst)

    vPafy = pafy.new(url)
    play = vPafy.getbest(preftype="webm")
    #start the video                                                               
    cap = cv2.VideoCapture(play.url)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( "Number of picures captured: " + str(length) )
    frame = 0
    while True:
        check, img = cap.read()
        if check:
            cv2.imwrite(os.path.join(dst,"%d.jpg") %frame, img)
            frame += 1
        else: 
            break 
    cap.release()


