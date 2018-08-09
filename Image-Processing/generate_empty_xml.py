################### Generate Empty XML#########################
# Use this file to generate empty labeled xml files
# Maybe useful if you want to add more empty(negative) example to the training set
# However this may not be useful to improve the object detection model
"""
<annotation>
    <folder>img</folder>
    <filename>download_1_31_1.jpg</filename>
    <path>/data/dataset/wheel/img/download_1_31_1.jpg</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>179</width>
        <height>112</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
</annotation>
"""
#################################################################

import xml.etree.cElementTree as ET
import os
import numpy as np
import cv2


# Function to write empty label
def write_empty_xml(name, input_path, width, height, depth, output_path):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = input_path.split('/')[-2]
    ET.SubElement(root, "filename").text = name+'.jpg'
    ET.SubElement(root, "path").text = input_path

    source = ET.SubElement(root, "source")

    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(root, "segmented").text = '0'

    tree = ET.ElementTree(root)
    tree.write(output_path + '.xml')


# Function to get the name of files
def get_files_name(filepath):
    pathDir =  os.listdir(filepath)
    name = []
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        name.append(child.split('/')[-1].split('.')[0])  #only keep the name between / and .
    return name

# Get the size of the picture
def get_img_size(img_path, file_name):
    img = cv2.imread(img_path + file_name + '.jpg')
    return img.shape


def generate_empty_xml(img_path, label_path):
    """
    Input:
        img_path: folder path store all the images
        label_path: label path you want to write to
    """
    # Get the list of filenames
    file_name_list = get_files_name(img_path)
    
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    # write all the empty xml for every images
    for file_name in file_name_list:
        input_path = img_path + file_name
        output_path = label_path + file_name
        height, width, depth = get_img_size(img_path, file_name)
        write_empty_xml(file_name, input_path, width, height, depth, output_path)

