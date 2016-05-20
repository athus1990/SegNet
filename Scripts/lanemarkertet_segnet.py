import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import cv2
import sys
import time


sys.path.append('/usr/local/lib/python2.7/site-packages')
# Make sure that caffe is on the python path:
caffe_root = '/home/Athma/Downloads/SegNet/caffe-segnet/'
sys.path.insert(0, caffe_root + 'python')
import caffe

# Import arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, required=True)
# parser.add_argument('--weights', type=str, required=True)
# parser.add_argument('--colours', type=str, required=True)
# parser.add_argument('--input', type=str, required=False,default="/home/Athma/Downloads/SegNet/vid7.MP4")
# parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',default=0, type=int)
# args = parser.parse_args()

model='/home/Athma/Downloads/SegNet/Models/segnet_lanemarker.prototxt'
weight='/home/Athma/Downloads/SegNet/Models/Training/SegnetLM__iter_30000.caffemodel'
#weight='/home/Athma/Downloads/SegNet/Models/Inference/LM/test_weights.caffemodel'
colours='Scripts/camvid12.png'


video = '/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151207_120040.MP4'  # InnerCity_Traffic
video='/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151207_115227.MP4'#InnerCity_Traffic
video='/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151207_120230.MP4'
#video='/home/Athma/Downloads/20160129_093059.MP4'#snow
#video='/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151209_113928.MP4'

input=video

net = caffe.Net(model,
				weight,
				caffe.TEST)

caffe.set_mode_gpu()
caffe.set_device(0)


input_shape = net.blobs['data'].data.shape
output_shape = net.blobs['argmax'].data.shape

label_colours = cv2.imread(colours).astype(np.uint8)

cv2.namedWindow("Input")
cv2.namedWindow("SegNet")

cap = cv2.VideoCapture(input) # Change this to your webcam ID, or file name for your video file


if cap.isOpened(): # try to get the first frame
	rval, frame = cap.read()
else:
	rval = False
i=1
while rval:
	start = time.time()
	rval, frame = cap.read()
	end = time.time()
	print '%30s' % 'Grabbed camera frame in ', str((end - start)*1000), 'ms'


	start = time.time()
	frame=frame[0:-300,:,:]
	frame = cv2.resize(frame, (input_shape[3],input_shape[2]))


	input_image = frame.transpose((2,0,1))
	input_image = input_image[(2,1,0),:,:]
	input_image = np.asarray([input_image])
	end = time.time()
	print '%30s' % 'Resized image in ', str((end - start)*1000), 'ms'

	start = time.time()
	out = net.forward_all(data=input_image)
	end = time.time()
	print '%30s' % 'Executed SegNet in ', str((end - start)*1000), 'ms'

	start = time.time()
	segmentation_ind = np.squeeze(net.blobs['argmax'].data)
	segmentation_ind_3ch = np.resize(segmentation_ind,(3,input_shape[2],input_shape[3]))
	segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
	segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

	cv2.LUT(segmentation_ind_3ch,label_colours,segmentation_rgb)

	#ATHMA
	segmentation_rgb2=segmentation_rgb

	segmentation_rgb = segmentation_rgb.astype(float)/255

	end = time.time()
	print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms\n'

	cv2.imshow("Input", frame)
	cv2.imshow("SegNet", segmentation_rgb)

	#ATHMA
	# cv2.imwrite('/home/Athma/Downloads/SegNet/Output/SEGNET/image'+str(i)+'.jpg',segmentation_rgb2);
	# cv2.imwrite('/home/Athma/Downloads/SegNet/Output/INPUT/image'+str(i)+'.jpg',frame);
	i+=1

	key = cv2.waitKey(1)
	if key == 27: # exit on ESC
		break
cap.release()
cv2.destroyAllWindows()

