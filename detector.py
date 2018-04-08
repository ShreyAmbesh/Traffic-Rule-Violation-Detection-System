import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils

from collections import defaultdict
from io import StringIO
from PIL import Image

if tf.__version__ < '1.4.0':
	raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
# This is needed to display the images.


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


from utils import label_map_util

from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'training'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('training', 'License-Plate-Detection.pbtxt')

NUM_CLASSES = 14


'''tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
	file_name = os.path.basename(file.name)
	if 'frozen_inference_graph.pb' in file_name:
		tar_file.extract(file, os.getcwd())'''

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')


category_index={1: {'id': 1, 'name': u'person'},
                2: {'id': 2, 'name': u'bicycle'},
                3: {'id': 3, 'name': u'car'},
                4: {'id': 4, 'name': u'motorcycle'},
                5: {'id': 5, 'name': u'airplane'},
                6: {'id': 6, 'name': u'bus'},
                7: {'id': 7, 'name': u'train'},
                8: {'id': 8, 'name': u'truck'},
                9: {'id': 9, 'name': u'boat'},
                10: {'id': 10, 'name': u'traffic light'},
                11: {'id': 11, 'name': u'fire hydrant'},
                13: {'id': 13, 'name': u'stop sign'},
                14: {'id': 14, 'name': u'parking meter'}} 

import cv2
cap=cv2.VideoCapture('video.mp4') # 0 stands for very first webcam attach
filename="testoutput.avi"
codec=cv2.VideoWriter_fourcc('m','p','4','v')#fourcc stands for four character code
framerate=30
resolution=(640,480)
    
VideoFileOutput=cv2.VideoWriter(filename,codec,framerate, resolution)
vs = WebcamVideoStream(src='test.mp4').start()

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
    
		ret=True
		start=time.time()
		c=0



		
		'''fps = FPS().start()

		# loop over some frames...this time using the threaded stream
		while True:
			# grab the frame from the threaded video stream and resize it
			# to have a maximum width of 400 pixels
			frame = vs.read()
			frame = imutils.resize(frame, width=400)

			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# update the FPS counter
			fps.update()

		# stop the timer and display FPS information
		fps.stop()
		'''

		
		while (ret):
        
			r,image_np=cap.read()
			#image_np = imutils.resize(image_np, width=400) 
			c=c+1
			# Definite input and output Tensors for detection_graph
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			       
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)
			# Actual detection.
			(boxes, scores, classes, num) = sess.run(
				[detection_boxes, detection_scores, detection_classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})
			# Visualization of the results of a detection.
			vis_util.visualize_boxes_and_labels_on_image_array(
				image_np,
				np.squeeze(boxes),
				np.squeeze(classes).astype(np.int32),
				np.squeeze(scores),
				category_index,
				use_normalized_coordinates=True,
				line_thickness=8)
  
			#VideoFileOutput.write(image_np)
			cv2.imshow('live_detection',image_np)
			if cv2.waitKey(25) & 0xFF==ord('q'):
				elapsed=time.time()-start
				print('Run Time = ',elapsed)
				print('fps = ',c/elapsed)
				break
				cv2.destroyAllWindows()
				cap.release()
