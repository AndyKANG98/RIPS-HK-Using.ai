# import Image library and numpy
import PIL as pil
import numpy as np
import os
from PIL import Image
import xml.dom.minidom
from xml.etree.ElementTree import ElementTree,Element


def label_resize(height, width, path1, new_path1):
    """
    Resize the label xml file according to the height and width
    Input：
        height: height of xml you want to resize to
        width: width of xml you want to resize to
        path2: the input path for xml
        new_path2: the output path for resized xml
    """

    if not os.path.exists(new_path1):
        os.makedirs(new_path1)
    files= os.listdir(path1)
    for file in files:
        if not (os.path.isdir(file)):
            dom = xml.dom.minidom.parse(path1 + file) # read single xml file
            root = dom.documentElement
        im_height = root.getElementsByTagName('height')
        im_width = root.getElementsByTagName('width')
        # get resize ratio

        height_ratio = float(im_height[0].firstChild.data) / height
        width_ratio = float(im_width[0].firstChild.data) / width

        # get root
        tree = ElementTree()
        tree.parse(path1 + file)
        Root = tree.getroot()
        element1 = Element('height ratio')
        element1.text = str(height_ratio)
        Root.append(element1)

        element2 = Element('width ratio')
        element2.text = str(width_ratio)
        Root.append(element2)
        tree.write(new_path1 + file ,encoding='utf-8',xml_declaration=True)


        new_height = height
        new_width = width

        # modify the info between labels
        im_height[0].firstChild.data = height
        im_width[0].firstChild.data = width

        # resize the bounding boxes
        xmax = root.getElementsByTagName('xmax')
        xmin = root.getElementsByTagName('xmin')
        ymax = root.getElementsByTagName('ymax')
        ymin = root.getElementsByTagName('ymin')
        for i in xmax:
            i.firstChild.data = int(float(i.firstChild.data) / width_ratio)
        for i in xmin:
            i.firstChild.data = int(float(i.firstChild.data) / width_ratio)
        for i in ymax:
            i.firstChild.data = int(float(i.firstChild.data) / height_ratio)
        for i in ymin:
            i.firstChild.data = int(float(i.firstChild.data) / height_ratio)

        with open(os.path.join(new_path1,file),'w') as fh:
            dom.writexml(fh)

def image_resize(height, width, path2, new_path2):
    """
    Resize the label xml file according to the height and width
    Input：
        height: height of images you want to resize to
        width: width of images you want to resize to
        path2: the input path for images
        new_path2: the output path for resized images
    """
    files= os.listdir(path2)
    if not os.path.exists(new_path2):
        os.makedirs(new_path2)
    for file in files:
        if not os.path.isdir(file):
            im = Image.open(path2 + file)
            im = im.resize((height,width),Image.ANTIALIAS)
            im.save(new_path2 + file,"JPEG")

