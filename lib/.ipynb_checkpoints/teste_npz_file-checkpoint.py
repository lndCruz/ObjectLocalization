'''
Esse arquivo eh para testar um dataset.
Em posse das imagens, vou compacta-las em um arquivo .npz test_input and test_target
esses arquivos ficam dentro da pasta data.

O VOC2012_npz_files, divide os datasets em train and test,
ja esse arquivo eh soh para gerar o test.
'''

import numpy as np
import os
import random
import sys
import psutil
import tensorflow as tf
from PIL import Image

import os
import io
import glob
import xml.etree.ElementTree as ET
import random
import numpy as np
from PIL import Image
import pickle
import gzip

DEFAULT_SEED = 123456

from collections import namedtuple

# Path to the dataset annotation
xml_path = "../../VOC2007/Annotations/*.xml"
# Path to the prepared data
destination = "../../data/"


def class_text_to_int(row_label):
    """
    This function assigns a specific digit to every class of objects.
    Args: 
       row_label: label
    Returns:  
       Digit corresponding to the label
    """

    switcher = {

        "person": 1,
        "bird": 2,
        "cat": 3,
        "cow": 4,
        "dog": 5,
        "horse": 6,
        "sheep": 7,
        "aeroplane": 8,
        "bicycle": 9,
        "boat": 10,
        "bus": 11,
        "car": 12,
        "motorbike": 13,
        "train": 14,
        "bottle": 15,
        "diningtable": 16,
        "pottedplant": 17,
        "sofa": 18,
        "tvmonitor": 19,
        "chair": 20

    }

    if row_label in switcher.keys():
        return switcher.get(row_label)
    else:
        raise ValueError('The class is not defined: {0}'.format(row_label))
        

def load_image(addr):
    """
    Converting an image to string
    Args:
       addr: Address to an image
    Returns:
       Converted image to string
    """

    img = np.array(Image.open(addr))
    return img.tostring()


def create_example(xml_file):
    """
    Creating a dict from datapoints
    Args: 
      xml_file: Path to xml file
    Returns:
      Record corresponding to an image including image and its ground truth
    """

    #Loading xml file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    #Getting image name and its dimentions
    image_name = root.find('filename').text
    file_name = image_name.encode('utf8')
    size=root.find('size')
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    depth = int(size.find("depth").text)

    #Initilizing variables for the loaded image
    #Please note that following variables are vector. Since an image might have more than one object a vector is used.
    #So, forexample if an image includes 5 objects xmin will hold five numberse corresponding to each of the objects.
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    #Reading objects in an image one by one
    for member in root.findall('object'):

        #Adding name of the object
        classes_text.append(member.find("name").text)

        #Adding boundary of the object
        boundBox = member.find("bndbox")
        for elem in boundBox:
            if elem.tag == "xmax":
                xmax.append(float(elem.text))
            elif elem.tag == "xmin":
                xmin.append(float(elem.text))
            elif elem.tag == "ymin":
                ymin.append(float(elem.text))
            elif elem.tag == "ymax":
                ymax.append(float(elem.text))

        #Adding difficulty attributes. It means how far it is difficult to recognise the object
        difficult_obj.append(int(member.find("difficult").text))

        classes.append(class_text_to_int(member.find("name").text))

        # truncated feature is ommited becuase it nolonger availible in VOC2012 dataset.
        #To keep the dataset consistent, it is removed.
        #However, you can add it again just by uncomenting the line below and its corresponding part in building and restoring tfrecord.
        #truncated.append(int(member.find("truncated").text))

        #Adding position of the object that the image is taken.
        poses.append(member.find("pose").text)

    #Finding the corresponding image and turnining it to a string.
    full_path = os.path.join('../../VOC2007/JPEGImages', '{}'.format(image_name))
    img = load_image(full_path)

    #Creating a dictionary from the image deatures
    example =             {
                'image_height': height,
                'image_width': width,
                'image_depth': depth,
                'image_filename': file_name,
                'image': img,
                'xmin': xmin,
                'xmax': xmax,
                'ymin': ymin,
                'ymax': ymax,
                'classes': classes_text
                # The following features are unnecessay for this task but they can be uncommented for other purposes.
                #'label': classes,
                #'difficult': difficult_obj,
                #'truncated': int64_list_feature(truncated),
                #'view': poses,
            }


    return example


def writting_files(xml_dir, dest_dir):
    """
    Creating .npz files
    Args:
      xml_dir: Path to xml file
      dest_dir: Destination that .npz files are wrriten
    """

    i=1
    tst=0   #to count number of images for evaluation

    files_counter = 1
    test_counter = 0

    test_input = []
    test_target = []

    print("Reading dataset is started. Please wait it might take several minutes to create .npz files ...")
    for xml_file in  glob.glob(xml_dir):
        # Create a tfrecord
        example = create_example(xml_file)

        # Every 5th file (xml and image) is writen for test set
        if (i%5)==0:

            temp = {'image_height':example['image_height'], 'image_width':example['image_width'], 'image_depth':example['image_depth'], 'image':example['image'], 'image_filename':example['image_filename']}
            test_input.append(temp)

            temp = {'xmin':example['xmin'], 'xmax':example['xmax'], 'ymin':example['ymin'], 'ymax':example['ymax'], 'objName':example['classes']}
            test_target.append(temp)

            tst=tst+1

        i=i+1

    print('test_target.npz and test_input.npz are being written ...'.format(files_counter))
    np.savez_compressed(dest_dir + 'test_target.npz', test_target)
    np.savez_compressed(dest_dir + 'test_input.npz', test_input)
    print("Files are written. It's done.")


    print('test dataset: # ')
    print(tst)


def main():

    writting_files(xml_path, destination)
    print("Files are ready!!!")
    
    
if __name__ == "__main__":
    main()