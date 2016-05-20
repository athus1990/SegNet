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


def processsigns(scnt, img):
	returnims = np.zeros_like(img)
	for cnt in scnt:
		rect = cv2.minAreaRect(cnt)
		box = cv2.cv.BoxPoints(rect)
		box = np.int0(box)
		x1 = min(box[:, 0])
		x2 = max(box[:, 0])
		y1 = min(box[:, 1])
		y2 = max(box[:, 1])
		area=((x2-x1)*(y2-y1))
		if area>50:
			cv2.drawContours(returnims, [box], 0, (255, 255, 0), 1)
		#
		# if (x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0):
		# 	sign = img[y1:y2, x1:x2, :]
		# 	A3_3ch = np.zeros_like(sign)
		# 	dst = cv2.fastNlMeansDenoisingColored(sign,None,5,5,7,21)
		# 	dst = cv2.GaussianBlur(dst, (5, 5), 0)
		# 	edges = cv2.Canny(dst, 0, 255)
		# 	contours1, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# 	#contours1 = sorted(contours1, key=cv2.contourArea, reverse=True)[:1]
		# 	for cnt2 in contours1:
		# 		if len(cnt2) > 5:
		# 			ellipse = cv2.fitEllipse(cnt2)
		# 			cv2.ellipse(A3_3ch, ellipse, (255, 255, 0), 2)
		# 	returnims[y1:y2, x1:x2, :] = A3_3ch
	return returnims


def processroadsidewalksigns(net, input_shape, label_colours, showrect=1):
	segmentation_ind = np.squeeze(net.blobs['argmax'].data)
	segmentation_ind_3ch = np.resize(segmentation_ind, (3, input_shape[2], input_shape[3]))
	segmentation_ind_3ch = segmentation_ind_3ch.transpose(1, 2, 0).astype(np.uint8)
	segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)
	cv2.LUT(segmentation_ind_3ch, label_colours, segmentation_rgb)
	segmentation_rgb = segmentation_rgb.astype(float) / 255.0
	seg_backup = segmentation_ind.copy()

	SignRoadSide = np.zeros_like(segmentation_rgb)

	# sign
	segmentation_ind = seg_backup.copy()
	segmentation_ind[segmentation_ind > 6] = 0
	segmentation_ind[segmentation_ind < 5.5] = 0
	SignRoadSide[:, :, 0] = segmentation_ind * 40
	contours1, hierarchy = cv2.findContours(SignRoadSide[:, :, 0].astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	pixelpoint1 = []
	[pixelpoint1.append(cnt) for cnt in contours1 if cv2.contourArea(cnt) > 2]
	pixelpoint1 = sorted(pixelpoint1, key=cv2.contourArea, reverse=True)[:20]
	signcnts = pixelpoint1
	A3_3ch = np.zeros_like(SignRoadSide)
	if showrect == 1:
		for cnt in pixelpoint1:
			rect = cv2.minAreaRect(cnt)
			box = cv2.cv.BoxPoints(rect)
			box = np.int0(box)
			cv2.drawContours(A3_3ch, [box], 0, (255, 255, 255), -1)
	else:
		cv2.drawContours(A3_3ch, pixelpoint1, -1, (255, 255, 255), -1)
	SignRoadSide[:, :, 0] = cv2.cvtColor(A3_3ch.astype(np.uint8), cv2.COLOR_RGB2GRAY)

	# road
	segmentation_ind = seg_backup.copy()
	segmentation_ind[segmentation_ind > 3] = 0
	segmentation_ind[segmentation_ind < 2.5] = 0
	SignRoadSide[:, :, 1] = segmentation_ind * 60
	SignRoadSide[0:200,:,1]=np.zeros((200,480))
	contours2, hierarchy = cv2.findContours(SignRoadSide[:, :, 1].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	pixelpoint2 = []
	[pixelpoint2.append(cnt2) for cnt2 in contours2 if cv2.contourArea(cnt2) > 2]
	pixelpoint2 = sorted(pixelpoint2, key=cv2.contourArea, reverse=True)[:20]
	A3_3ch = np.zeros_like(SignRoadSide)
	if showrect == 1:
		for cnt in pixelpoint1:
			rect = cv2.minAreaRect(cnt)
			box = cv2.cv.BoxPoints(rect)
			box = np.int0(box)
			cv2.drawContours(A3_3ch, [box], 0, (255, 255, 255), -1)
	else:
		cv2.drawContours(A3_3ch, pixelpoint2, -1, (255, 255, 255), -1)
	SignRoadSide[:, :, 1] = cv2.cvtColor(A3_3ch.astype(np.uint8), cv2.COLOR_RGB2GRAY)

	# sidewalk
	segmentation_ind = seg_backup.copy()
	segmentation_ind[segmentation_ind > 4] = 0
	segmentation_ind[segmentation_ind < 3.5] = 0
	SignRoadSide[:, :, 2] = segmentation_ind * 60
	contours1, hierarchy = cv2.findContours(SignRoadSide[:, :, 2].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	pixelpoint1 = []
	[pixelpoint1.append(cnt) for cnt in contours1 if cv2.contourArea(cnt) > 2]
	pixelpoint1 = sorted(pixelpoint1, key=cv2.contourArea, reverse=True)[:]
	A3_3ch = np.zeros_like(SignRoadSide)
	if showrect == 1:
		for cnt in pixelpoint1:
			rect = cv2.minAreaRect(cnt)
			box = cv2.cv.BoxPoints(rect)
			box = np.int0(box)
			cv2.drawContours(A3_3ch, [box], 0, (255, 255, 255), 2)
	else:
		cv2.drawContours(A3_3ch, pixelpoint1, -1, (255, 255, 255), 2)

	SignRoadSide[:, :, 2] = cv2.cvtColor(A3_3ch.astype(np.uint8), cv2.COLOR_RGB2GRAY)

	return SignRoadSide, segmentation_rgb, seg_backup.copy(),signcnts


# ~~~~~~~PARAMS~~~~~~~~~~~~~~~~~~~~~
colours = 'Scripts/camvid12.png'
video = '/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151207_120040.MP4'  # InnerCity_Traffic
video='/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151207_115227.MP4'#InnerCity_Traffic
#video='/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151207_120230.MP4'
#video='/home/Athma/Downloads/20160129_093059.MP4'#snow
#video='/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151209_113928.MP4'

model1 = '/home/Athma/Downloads/SegNet/Models/segnet_camvid.prototxt'
model2 = '/home/Athma/Downloads/SegNet/Models/segnet_camvid.prototxt'
weights1 = '/home/Athma/Downloads/SegNet/Models/Training/Segnet4_iter_60000.caffemodel'
weights2 = '/home/Athma/Downloads/SegNet/Models/Training/Segnet5_iter_90000.caffemodel'

model3='/home/Athma/Downloads/SegNet/Example_Models/segnet_model_driving_webdemo.prototxt'
weights3='/home/Athma/Downloads/SegNet/Example_Models/segnet_weights_driving_webdemo.caffemodel'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~SETUP NET~~~~~~~~~~~~~~~~~~~~~
net1 = caffe.Net(model1, weights1, caffe.TEST)
net2 = caffe.Net(model2, weights2, caffe.TEST)
net3 = caffe.Net(model3, weights3, caffe.TEST)

caffe.set_mode_gpu()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~SETUP INPUT/OTPUT~~~~~~~~~~~~~~~~~~~~~
input_shape = net1.blobs['data'].data.shape
output_shape = net1.blobs['argmax'].data.shape
label_colours = cv2.imread(colours).astype(np.uint8)
cap = cv2.VideoCapture(video)  # Change this to your webcam ID, or file name for your video file
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



fourcc = cv2.cv.CV_FOURCC(*'H264')
vidout1 = cv2.VideoWriter('/home/Athma/Downloads/SegNet/InputOutput/resultvideos/contrastbrightnesspdated/seg4.avi', fourcc, 20.0, (480,360))
vidout2 = cv2.VideoWriter('/home/Athma/Downloads/SegNet/InputOutput/resultvideos/contrastbrightnesspdated/seg5.avi', fourcc, 20.0, (480,360))
vidout3 = cv2.VideoWriter('/home/Athma/Downloads/SegNet/InputOutput/resultvideos/contrastbrightnesspdated/combined.avi', fourcc, 20.0, (480,360))




if cap.isOpened():  # try to get the first frame
	rval, frame = cap.read()
else:
	rval = False
i = 1
while rval:
	print 'FRAME = ' + str(i)
	i = i + 1
	rval, frame = cap.read()
	if rval==True :
		# PROCESS INPUT FOR DETECTION
		frame = frame[80:-280, :, :]
		#CONTRAST AND BRIGHTNESS SWITCH
		alpha = float(0.4)     # Simple contrast control
		beta = float(50)             # Simple brightness control
		mul_img = cv2.multiply(frame,np.array([alpha]))# mul_img = img*alpha
		beta2=beta*np.ones_like(mul_img)
		frame =mul_img+beta2# new_img = img*alpha + beta
		# #CONTRAST AND BRIGHTNESS SWITCH

		frame = cv2.GaussianBlur(frame, (5, 5), 0)
		frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
		input_image = frame.transpose((2, 0, 1))
		input_image = input_image[(2, 1, 0), :, :]
		input_image = np.asarray([input_image])

		# DETECT
		out1 = net1.forward_all(data=input_image)
		out2 = net2.forward_all(data=input_image)
		out3 = net3.forward_all(data=input_image)

		# PROCESS RESULT

		(ROAD_ONLY1, segmentation_rgb1, segind1,signcnts1) = processroadsidewalksigns(net1, input_shape, label_colours, 0)
		SegSigns1=processsigns(signcnts1, frame)
		(ROAD_ONLY2, segmentation_rgb2, segind2,signcnts2) = processroadsidewalksigns(net2, input_shape, label_colours, 0)
		SegSigns2=processsigns(signcnts2, frame)

		(ROAD_ONLY3, segmentation_rgb3, segind3,signcnts3) = processroadsidewalksigns(net3, input_shape, label_colours, 0)


		ROAD_ONLYF = (ROAD_ONLY1 + ROAD_ONLY2) / 2.0
		SEGSIGNF=(SegSigns1+SegSigns1) /2.0

		#sky processing
		segind3[segind3!=0]=1
		ROAD_ONLYF[segind3==0]=0
		ROAD_ONLYF[0:200,:,1:]=np.zeros([200,480,2])

		###VIEW INPUT AND OUTPUT
		#cv2.imshow("Input", frame)
		FRAME_SegNet_combined=cv2.addWeighted(frame.astype(np.uint8), 1, ROAD_ONLYF.astype(np.uint8), 1, 0)
		#cv2.imshow("FRAME+SegNet_combined", FRAME_SegNet_combined)
		FRAME_SegNet_combined_SegSigns=cv2.addWeighted(FRAME_SegNet_combined.astype(np.uint8), 1,SEGSIGNF.astype(np.uint8), 1, 0)
		#cv2.imshow("FRAME+SegNet_combined+SegSigns", FRAME_SegNet_combined_SegSigns)

		vidout1.write(cv2.addWeighted(frame.astype(np.uint8), 1, ROAD_ONLY1.astype(np.uint8), 1, 0))
		vidout2.write(cv2.addWeighted(frame.astype(np.uint8), 1, ROAD_ONLY2.astype(np.uint8), 1, 0))
		vidout3.write(FRAME_SegNet_combined_SegSigns)

		# key = cv2.waitKey(5)
		# if key == 27:  # exit on ESC
		# 	break
# #time.sleep(1)
#
cap.release()
vidout1.release()
vidout2.release()
vidout3.release()
cv2.destroyAllWindows()
