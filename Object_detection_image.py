######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph_dice'
file_type = '.jpg'
file_name = '20181102_155854'
file_dir = 'images_dice/eval/'
save_dir = 'images_dice/detected/'
#IMAGE_LOC = 'images_dice/train/'
IMAGE_NAME = file_dir+file_name+file_type

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training_dice','labelmap.pbtxt')

# Path to image
#PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 12

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

file_name_list = ['20181103_082204', '20181103_082228', '20181103_082240', '20181103_082255', 
'20181103_082305', '20181103_082315', '20181103_082328', '20181103_082335', '20181103_082343',
 '20181103_082354', 
'20181103_082359', '20181103_082406', '20181103_082417', '20181103_082429', '20181103_082436', 
'20181103_082452', '20181103_082457', '20181103_082504', '20181103_082509', '20181103_082516', 
'20181103_082535', '20181103_082545', '20181103_082557', '20181103_082621', '20181103_082629', 
'20181103_082636']


for i in file_name_list:

    IMAGE_NAME = file_dir+i+file_type
    PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    image = cv2.resize(image,(1920,1080))

    image_expanded = np.expand_dims(image, axis=0)


    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    #print(boxes,scores,classes,num)
    # Draw the results of the detection (aka 'visulaize the results')
    print(np.squeeze(classes).astype(np.int32))
    image, class_list = vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.80)


    print(class_list)
    class_sum = 0
    for class_val in class_list:
        i_hold = int(class_val[0].split(":")[0])
        class_sum += i_hold
    #print('list_sum: ',class_sum)

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    CornerOfText = (10,50)
    fontScale              = 2
    fontColor              = (0,255,0)
    lineType               = 2

    cv2.putText(image,'Dice value: '+str(class_sum), 
    CornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)

    # All the results have been drawn on image. Now display the image.
    #cv2.imshow(i, image)

    # Press any key to close the image
    #cv2.waitKey(0)
    



    cv2.imwrite(save_dir+i+'_eval_detected.jpg',image)

    # Clean up
    cv2.destroyAllWindows()
