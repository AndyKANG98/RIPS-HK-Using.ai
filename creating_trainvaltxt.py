''' July 5, 2018
BP: Tool to list of names of all images with annotations in a .txt file. This is needed to separate
data into training and test sets.

Example usage:
--------------
input_path=/path/to/images
output_path=/path/to/output

python creating_trainvaltxt
    --input_dir=$input_path
    --output_dir=$output_path
'''

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="Path with all the images")
parser.add_argument("--output_dir", help="Path to put the trainval.txt file")

args = parser.parse_args()

input_path = args.input_dir
output_path = args.output_dir

# list all name of the images
img_names = os.listdir(input_path)

# remove the extensions out of the images
strip_img_names = sorted(list(map(lambda x: os.path.splitext(x)[0], img_names)))

# remove the .DS_Store file because it's an extra file
try:
    strip_img_names.remove('.DS_Store')
except:
    pass

# create a tranval.txt
f = open(output_path + '/trainval.txt', 'w')

# write in the file
with f as text_file:
    for i in range(len(strip_img_names)):
        f.write(('%s \n' % strip_img_names[i]))
f.close()








