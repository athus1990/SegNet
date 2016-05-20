import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import math
import cv2
import sys
import time

sys.path.append('/usr/local/lib/python2.7/site-packages')
# Make sure that caffe is on the python path:
caffe_root = '/home/Athma/Downloads/SegNet/caffe-segnet/'
sys.path.insert(0, caffe_root + 'python')
import caffe


def sliding_window(image, stepSizex, stepSizey, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSizey):
		for x in xrange(0, image.shape[1], stepSizex):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def processroadsidewalksigns(net, input_shape, label_colours):
	segmentation_ind = np.squeeze(net.blobs['argmax'].data)
	segmentation_ind_3ch = np.resize(segmentation_ind, (3, input_shape[2], input_shape[3]))
	segmentation_ind_3ch = segmentation_ind_3ch.transpose(1, 2, 0).astype(np.uint8)
	segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)
	cv2.LUT(segmentation_ind_3ch, label_colours, segmentation_rgb)
	segmentation_rgb = segmentation_rgb.astype(float) / 255.0
	seg_backup = segmentation_ind.copy()
	ROAD_ONLY = np.zeros_like(segmentation_rgb)
	# road
	segmentation_ind[segmentation_ind > 3] = 0
	segmentation_ind[segmentation_ind < 2.5] = 0
	ROAD_ONLY[:, :, 1] = segmentation_ind * 60
	# sidewalk
	segmentation_ind = seg_backup.copy()
	segmentation_ind[segmentation_ind > 4] = 0
	segmentation_ind[segmentation_ind < 3.5] = 0
	ROAD_ONLY[:, :, 2] = segmentation_ind * 60
	# sign
	segmentation_ind = seg_backup.copy()
	segmentation_ind[segmentation_ind > 6] = 0
	segmentation_ind[segmentation_ind < 5.5] = 0
	ROAD_ONLY[:, :, 0] = segmentation_ind * 40

	# remove small signs
	SIGN_ONLY_GRAY1 = ROAD_ONLY[:, :, 0].astype(np.uint8)
	contours1, hierarchy = cv2.findContours(SIGN_ONLY_GRAY1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	pixelpoint1 = []
	[pixelpoint1.append(cnt) for cnt in contours1 if cv2.contourArea(cnt) > 30]
	A3_3ch = np.zeros_like(ROAD_ONLY)
	for cnt in pixelpoint1:
		rect = cv2.minAreaRect(cnt)
		box = cv2.cv.BoxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(A3_3ch,[box],0,(255,255,255),2)
	#cv2.drawContours(A3_3ch, pixelpoint1, -1, (255, 255, 255), thickness=-1)
	ROAD_ONLY[:, :, 0] = cv2.cvtColor(A3_3ch.astype(np.uint8), cv2.COLOR_RGB2GRAY)
	return (ROAD_ONLY, segmentation_rgb)


# ~~~~~~~PARAMS~~~~~~~~~~~~~~~~~~~~~
colours = 'Scripts/camvid12.png'
bluron = 0
doboth = 1
video = '/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151207_120040.MP4'  # InnerCity_Traffic
# video='/home/Athma/Downloads/20160129_093059.MP4'
video='/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151207_115227.MP4'#InnerCity_Traffic
model = '/home/Athma/Downloads/SegNet/Models/segnet_camvid.prototxt'
weights = '/home/Athma/Downloads/SegNet/Models/Training/Segnet4_iter_60000.caffemodel'
weights2 = '/home/Athma/Downloads/SegNet/Models/Training/Segnet5_iter_90000.caffemodel'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
label_colours = cv2.imread(colours).astype(np.uint8)
cap = cv2.VideoCapture(video)

# ~~~~~~~SETUP NET~~~~~~~~~~~~~~~~~~~~~
if doboth == 1:
	net = caffe.Net(model, weights, caffe.TEST)
net2 = caffe.Net(model, weights2, caffe.TEST)
caffe.set_mode_gpu()
if doboth == 1:
	input_shape = net.blobs['data'].data.shape
	output_shape = net.blobs['argmax'].data.shape
else:
	input_shape = net2.blobs['data'].data.shape
	output_shape = net2.blobs['argmax'].data.shape

if cap.isOpened():  # try to get the first frame
	rval, frame = cap.read()
else:
	rval = False
i = 1
while rval:
	subimage=[]
	subimage2=[]
	print 'FRAME = ' + str(i)
	rval, frame = cap.read()
	# print frame.shape
	# print input_shape
	# i=i+1
	frame=frame[360:2*360,480:4*480,:]
	cv2.imshow("Input", frame)
	for (x, y, window) in sliding_window(frame, stepSizex=480, stepSizey=360, windowSize=(480, 360)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != 360 or window.shape[1] != 480:
			continue
		input_image = window.transpose((2, 0, 1))
		input_image = input_image[(2, 1, 0), :, :]
		input_image = np.asarray([input_image])

		if doboth == 1:
			out = net.forward_all(data=input_image)
			(ROAD_ONLY1, segmentation_rgb) = processroadsidewalksigns(net, input_shape, label_colours)
			frame_seg=cv2.addWeighted(window.astype(np.uint8), 1, ROAD_ONLY1.astype(np.uint8), 0.5, 0)
			subimage.append((segmentation_rgb,frame_seg))
		out2 = net2.forward_all(data=input_image)
		(ROAD_ONLY2, segmentation_rgb2) = processroadsidewalksigns(net2, input_shape, label_colours)
		frame_seg2=cv2.addWeighted(window.astype(np.uint8), 1, ROAD_ONLY2.astype(np.uint8), 0.5, 0)
		subimage2.append((segmentation_rgb2,frame_seg2))
	pos=0
	FRAME_SEG=np.zeros_like(frame)
	SEG_RGB=np.zeros_like(frame)
	FRAME_SEG2=np.zeros_like(frame)
	SEG_RGB2=np.zeros_like(frame)

	for (img,img2) in zip(subimage,subimage2):
		loc=np.array(np.unravel_index(pos, (1,4)))
		pos=pos+1
		FRAME_SEG[360*loc[0]:360*(loc[0]+1),480*loc[1]:480*(loc[1]+1),:]=img[1]
		SEG_RGB[360*loc[0]:360*(loc[0]+1),480*loc[1]:480*(loc[1]+1),:]=img[0]
		FRAME_SEG2[360*loc[0]:360*(loc[0]+1),480*loc[1]:480*(loc[1]+1),:]=img2[1]
		SEG_RGB2[360*loc[0]:360*(loc[0]+1),480*loc[1]:480*(loc[1]+1),:]=img2[0]

	cv2.imshow("SegNet", FRAME_SEG)
	#cv2.imshow("FRAME+SegNet", SEG_RGB)
	cv2.imshow("SegNet2", FRAME_SEG2)
	#cv2.imshow("FRAME+SegNet2", SEG_RGB2)
	key = cv2.waitKey(5)
	if key == 27:  # exit on ESC
		break
	i=i+1

