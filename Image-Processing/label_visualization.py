"""
Use this file to visualize the labels to the corresponding images
"""

import cv2
import xml.dom.minidom
import os

def label_image(name, img_path, label_path, vis_path):
    """
    visualize the image with specific name (label "name.jpg" from "name.xml"
    """
    # get the image
    image_name = img_path + name + '.jpg'  
    img = cv2.imread(image_name)

    #open xml
    dom = xml.dom.minidom.parse(label_path + name+'.xml')

    #Get element of the root node 
    root = dom.documentElement

    # Create lists for wheels
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []

    # Get the all the tags for x,y
    xmins_tag = root.getElementsByTagName('xmin')
    ymins_tag = root.getElementsByTagName('ymin')
    xmaxs_tag = root.getElementsByTagName('xmax')
    ymaxs_tag = root.getElementsByTagName('ymax')

    
    # Fill the lists for wheels
    for i in xmins_tag:
        xmins.append(int(i.firstChild.data))
    for i in ymins_tag:
        ymins.append(int(i.firstChild.data))
    for i in xmaxs_tag:
        xmaxs.append(int(i.firstChild.data))
    for i in ymaxs_tag:
        ymaxs.append(int(i.firstChild.data))


    # Draw the bounding box
    for i in range(len(xmins)):
        cv2.rectangle(img, (xmins[i],ymins[i]), (xmaxs[i],ymaxs[i]), (0,255,0), 4)


    cv2.imwrite(vis_path + name + '.jpg', img)



def get_files_name(filepath):
    """
    Get a list of all filenames in the file path
    """
    pathDir =  os.listdir(filepath)
    name = []
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        name.append(child.split('/')[-1].split('.')[0])  #only keep the name between / and .
    return name
        


def data_vis(img_path, label_path, vis_path):
    """
    Use this function to generate visualized pictures.
    Path for input image, label, and output need to be set
    Input:
        img_path = '../data/wheel/img/'
        label_path = '../data/wheel/label/'
        vis_path = '../data/wheel/img_vis/'
    Output: 
        visualized pictures in vis_path
    """
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    file_name_list = get_files_name(label_path)
    
    for file_name in file_name_list:
        label_image(file_name, img_path, label_path, vis_path)

