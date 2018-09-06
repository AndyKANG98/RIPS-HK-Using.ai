''' July 5, 2018
BP: Tool to randomly separate images and their labels labels into training and test set.

NOTES:
* The input directory should have two folders "img" and "label".
* Script automatically makes train and test folders and puts the respective images in them
* If random seed is not given, 22 is passed as random seed.


Example Usage:
--------------
# When all the images in the dataset have labels.
input_path=/path/to/images and label
output_path=/path/to/save/output
test_prcnt=0.15
random_seed=22

python create_test_set.py
    --input_dir=$input_path
    --output_dir=$output_path
    --test_set_percentage=$test_prcnt
    --random_seed=$random_seed

# When not all of the images in the dataset have labels, pass path to a trainval.txt file which
contains name of the images with labels. The script only considers those images in the
trainval.txt file.

input_path=/path/to/images and label
out_path=/path/to/save/output
test_prcnt=0.15
random_seed=22
trainval_path=/path/to/trainval.txt

python create_test_set.py
    --input_dir=$input_path
    --output_dir=$output_path
    --test_set_percentage=$test_prcnt
    --trainval_path=$trainval_path
    --random_seed=$random_seed
'''

import os
import random
import numpy as np
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", help="directory with all the images and labels")
parser.add_argument("--output_dir", help="directory to put the test and train sets")
parser.add_argument("--test_set_percentage", help="percentage of the data towards the test set",
                    type=float)
parser.add_argument("--trainval_dir", help="OPTIONAL. dir with trainval.txt file that containes names of images")
parser.add_argument("--random_seed", help="seed for randomly separating into train and test set",
                    type=int)

args = parser.parse_args()

input_path = args.input_dir
output_path = args.output_dir
test_prcnt = args.test_set_percentage

# if trainval_path is given
if args.trainval_dir:
    trainval_path = args.trainval_dir
    file = open(trainval_path, 'r')
    image_names = [line.rstrip(' \n') for line in file]
else:
    image_names = os.listdir(input_path + '/img')
    image_names = sorted(list(map(lambda x: os.path.splitext(x)[0], image_names)))

# if random seed is given
if args.random_seed:
    random_seed = args.random_seed
else:
    random_seed = 22

# num of total examples
random.seed(random_seed)
random.shuffle(image_names)
num_examples = len(image_names)

# select test_prcnt of text images for test
num_test = int(test_prcnt * num_examples)
test_images = list(np.random.choice(image_names, num_test, replace=False))

# create the dirs if they don't exist
if not os.path.exists(output_path + '/test'):
    os.makedirs(output_path + '/test/img')
    os.makedirs(output_path + '/test/label')

if not os.path.exists(output_path + '/train'):
    os.makedirs(output_path + '/train/img')
    os.makedirs(output_path + '/train/label')

# move train images and test images to different directories
for img in image_names:
    if img in test_images:  # move to test folder
        copyfile(input_path + "/img/%s.jpg" % img, output_path + "/test/img/%s.jpg" % img)
        copyfile(input_path + "/label/%s.xml" % img, output_path + "/test/label/%s.xml" % img)
    else:  # move to train folder
        copyfile(input_path + "/img/%s.jpg" % img, output_path + "/train/img/%s.jpg" % img)
        copyfile(input_path + "/label/%s.xml" % img, output_path + "/train/label/%s.xml" % img)

