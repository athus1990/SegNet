import numpy as np
import os.path
import scipy
import argparse
import scipy.io as sio
import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import cv2
import sys
import scipy.ndimage as ndi
from scipy.signal import argrelextrema

# Make sure that caffe is on the python path:
caffe_root = '/home/Athma/Downloads/SegNet/caffe-segnet/'  # Change this to the absolute directoy to SegNet Caffe
sys.path.insert(0, caffe_root + 'python')

import caffe






def showonlyroadsidewalksign(segmentation_rgb, segmentation_ind):
	SignRoadSide = np.zeros_like(segmentation_rgb)
	seg_backup = segmentation_ind.copy()

	# sign
	segmentation_ind = seg_backup.copy()
	segmentation_ind[segmentation_ind > 6] = 0
	segmentation_ind[segmentation_ind < 5.5] = 0
	SignRoadSide[:, :, 0] = segmentation_ind * 40
	contours1, hierarchy = cv2.findContours(SignRoadSide[:, :, 0].astype(np.uint8), cv2.RETR_TREE,
											cv2.CHAIN_APPROX_SIMPLE)
	pixelpoint1 = []
	[pixelpoint1.append(cnt) for cnt in contours1 if cv2.contourArea(cnt) > 2]
	pixelpoint1 = sorted(pixelpoint1, key=cv2.contourArea, reverse=True)[:20]
	signcnts = pixelpoint1
	A3_3ch = np.zeros_like(SignRoadSide)
	cv2.drawContours(A3_3ch, pixelpoint1, -1, (255, 255, 255), -1)
	SignRoadSide[:, :, 0] = cv2.cvtColor(A3_3ch.astype(np.uint8), cv2.COLOR_RGB2GRAY)

	# road
	segmentation_ind = seg_backup.copy()
	segmentation_ind[segmentation_ind > 3] = 0
	segmentation_ind[segmentation_ind < 2.5] = 0
	SignRoadSide[:, :, 1] = segmentation_ind * 60
	SignRoadSide[0:200, :, 1] = np.zeros((200, 480))
	contours2, hierarchy = cv2.findContours(SignRoadSide[:, :, 1].astype(np.uint8), cv2.RETR_TREE,
											cv2.CHAIN_APPROX_NONE)
	pixelpoint2 = []
	[pixelpoint2.append(cnt2) for cnt2 in contours2 if cv2.contourArea(cnt2) > 2]
	pixelpoint2 = sorted(pixelpoint2, key=cv2.contourArea, reverse=True)[:20]
	A3_3ch = np.zeros_like(SignRoadSide)
	cv2.drawContours(A3_3ch, pixelpoint2, -1, (255, 255, 255), -1)
	SignRoadSide[:, :, 1] = cv2.cvtColor(A3_3ch.astype(np.uint8), cv2.COLOR_RGB2GRAY)

	# sidewalk
	segmentation_ind = seg_backup.copy()
	segmentation_ind[segmentation_ind > 4] = 0
	segmentation_ind[segmentation_ind < 3.5] = 0
	SignRoadSide[:, :, 2] = segmentation_ind * 60
	contours1, hierarchy = cv2.findContours(SignRoadSide[:, :, 2].astype(np.uint8), cv2.RETR_TREE,
											cv2.CHAIN_APPROX_SIMPLE)
	pixelpoint1 = []
	[pixelpoint1.append(cnt) for cnt in contours1 if cv2.contourArea(cnt) > 2]
	pixelpoint1 = sorted(pixelpoint1, key=cv2.contourArea, reverse=True)[:20]
	A3_3ch = np.zeros_like(SignRoadSide)
	cv2.drawContours(A3_3ch, pixelpoint1, -1, (255, 255, 255), -1)
	SignRoadSide[:, :, 2] = cv2.cvtColor(A3_3ch.astype(np.uint8), cv2.COLOR_RGB2GRAY)

	return SignRoadSide, signcnts


# Import arguments
video = '/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151207_120040.MP4'  # InnerCity_Traffic
video = '/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151207_115227.MP4'  # InnerCity_Traffic

#video = '/home/Athma/Downloads/20160129_093059.MP4'  # snow
#video = '/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151209_113928.MP4'
#video = '/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151207_120230.MP4'
#video='/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151209_113758.MP4'


#video = '/home/Athma/Downloads/Professordata/specialIntersections/Feb-Mar 2016-selected/20160420_200045.MP4'
#video = '/home/Athma/Downloads/Professordata/specialIntersections/Feb-Mar 2016-selected/20160420_200445.MP4'
video = '/home/Athma/Downloads/Professordata/specialIntersections/Feb-Mar 2016-selected/20160420_201117.MP4'
#video='/home/Athma/Downloads/Professordata/specialIntersections/Feb-Mar 2016-selected/20160421_092344.MP4'



model = '/home/Athma/Downloads/SegNet/Example_Models/bayesian_segnet_camvid.prototxt'
weights = '/home/Athma/Downloads/SegNet/Models/Training/bayesian_SEGNET1_iter_60000.caffemodel'
# weights='/home/Athma/Downloads/SegNet/Models/Final_models/B/test_weights.caffemodel'
colours = 'Scripts/camvid12.png'
data = '/home/Athma/Downloads/SegNet/CamVid/train.txt'
caffe.set_mode_gpu()

net = caffe.Net(model, weights, caffe.TEST)
input_shape = net.blobs['data'].data.shape
label_colours = cv2.imread(colours).astype(np.uint8)
i = 1
cap = cv2.VideoCapture(video)  # Change this to your webcam ID, or file name for your video file
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if cap.isOpened():  # try to get the first frame
	rval, frame = cap.read()
else:
	rval = False
i = 1
while rval:
	print 'FRAME = ' + str(i)
	i = i + 1
	rval, frame = cap.read()
	frame = frame[80:-280, :, :]
	frame = cv2.GaussianBlur(frame, (5, 5), 0)
	input_image_raw = frame.copy() / 255.0
	frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
	input_image = caffe.io.resize_image(input_image_raw, (input_shape[2], input_shape[3]))
	input_image = input_image * 255
	input_image = input_image.transpose((2, 0, 1))
	input_image = input_image[(2, 1, 0), :, :]
	input_image = np.asarray([input_image])
	input_image = np.repeat(input_image, input_shape[0], axis=0)
	out = net.forward(data=input_image)
	predicted = net.blobs['prob'].data
	output = np.mean(predicted, axis=0)
	uncertainty = np.var(output, axis=0)
	uncertainty *= 255.0 / uncertainty.max()
	ind = np.argmax(output, axis=0)
	segmentation_ind_3ch = np.resize(ind, (3, input_shape[2], input_shape[3]))
	segmentation_ind_3ch = segmentation_ind_3ch.transpose(1, 2, 0).astype(np.uint8)
	segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)
	cv2.LUT(segmentation_ind_3ch, label_colours, segmentation_rgb)

	# cv2.imshow('seg',segmentation_rgb)
	# cv2.imshow('uncertainity',uncertainty.astype(np.uint8))
	cv2.waitKey(5)
	ROAD_ONLY, signcnts = showonlyroadsidewalksign(segmentation_rgb, ind)
	FRAME_SegNet_combined = cv2.addWeighted(frame.astype(np.uint8), 1, ROAD_ONLY.astype(np.uint8), 1, 0)
	# cv2.imshow("FRAME+SegNet_combined2", FRAME_SegNet_combined)


	thresh = 100
	backup = ROAD_ONLY.copy()
	road = ROAD_ONLY[:, :, 1].copy()
	road[uncertainty <= thresh] = 0
	backup[:, :, 1] = road
	sidewalk = ROAD_ONLY[:, :, 2].copy()
	sidewalk[uncertainty <= thresh] = 0
	backup[:, :, 2] = sidewalk
	sign = ROAD_ONLY[:, :, 0].copy()
	sign[uncertainty <= thresh] = 0
	backup[:, :, 0] = sign
	PROCESSED_FRAME_SegNet_combined = cv2.addWeighted(frame.astype(np.uint8), 1, backup.astype(np.uint8), 1, 0)

	kernel = np.ones((2, 2), np.uint8)
	road = cv2.morphologyEx(backup[:, :, 1], cv2.MORPH_OPEN, kernel)
	road = cv2.dilate(road, kernel, iterations=1)
	res = cv2.bitwise_and(frame, frame, mask=road)

	gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)



	YELLOW = 30
	hue = YELLOW // 2
	lower_range = np.array([hue - 10, 0, 0], dtype=np.uint8)
	upper_range = np.array([hue + 20, 255, 255], dtype=np.uint8)
	hsvimg = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsvimg, lower_range, upper_range)
	gray = cv2.addWeighted(gray, 1, mask, 1, 0)
	ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
	gray = cv2.bitwise_and(gray, gray, mask=thresh)
	cv2.imshow('gray', gray)
	opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
	opening = cv2.fastNlMeansDenoising(opening.astype(np.uint8),h=20,templateWindowSize=7,searchWindowSize=21)
	cv2.imshow('morphology',opening)
	gray=opening.copy()
	x1 = 0
	y1 = 240
	x2 = 480
	y2 = 240
	x3 = 0
	y3 = 360
	x4 = 480
	y4 = 360

	pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
	pts2 = np.float32([[0, 0], [400, 0], [0, 300], [400, 300]])
	M = cv2.getPerspectiveTransform(pts1, pts2)
	perspective = cv2.warpPerspective(gray, M, (400, 300))

	laplacian = cv2.Laplacian(perspective,cv2.CV_64F)
	sobelx = cv2.Sobel(perspective,cv2.CV_64F,1,0,ksize=11)
	sobely = cv2.Sobel(perspective,cv2.CV_64F,0,1,ksize=11)
	sob=cv2.addWeighted(sobelx,1,sobely,1,0)
	sob=sob/np.max(sob)
	sob[sob<0]=0
	sob=sob*255.0
	ret,thresh=cv2.threshold(sob.astype(np.uint8),0,255,cv2.THRESH_BINARY)
	sob=cv2.bitwise_and(sob,sob,mask=thresh)
	kernel = np.ones((3,3),np.uint8)
	sob = cv2.morphologyEx(sob, cv2.MORPH_OPEN, kernel)
	#sob = cv2.fastNlMeansDenoising(sob.astype(np.uint8),h=10,templateWindowSize=7,searchWindowSize=21)
	cv2.imshow('sob', sob)
	pts1 = np.float32([[0, 0], [400, 0], [0, 300], [400, 300]])
	pts2 = np.float32([[0, 240], [480, 240], [0, 360], [480, 360]])
	M = cv2.getPerspectiveTransform(pts1, pts2)
	remapperspective = cv2.warpPerspective(sob, M, (480, 360))
	N=np.repeat(remapperspective[:,:,np.newaxis],3,axis=2)
	sob2=remapperspective.copy()
	minLineLength = 10
	maxLineGap = 20
	lines = cv2.HoughLinesP(sob2.astype(np.uint8),1,np.pi/180,20,minLineLength,maxLineGap)
	if lines!=None:
		for x1,y1,x2,y2 in lines[0]:
			cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
	contours, hierarchy = cv2.findContours(backup[:,:,1].astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# A3_3ch = np.zeros_like(backup)
	cv2.drawContours(frame, contours, -1, (0, 255, 255), thickness=2)
	# sob=cv2.addWeighted(sob.astype(np.uint8), 1, A3_3ch.astype(np.uint8), 1, 0)
	cv2.imshow('lines',sob2)
	cv2.imshow('remap', N)
	cv2.imshow('input2', frame)
	cv2.imshow("PROCESSED_FRAME+SegNet_combined2", PROCESSED_FRAME_SegNet_combined)


# uncomment to save results
# scipy.misc.toimage(segmentation_rgb, cmin=0.0, cmax=255.0).save(IMAGE_FILE+'_segnet_segmentation.png')
# cm = matplotlib.pyplot.get_cmap('bone_r')
# matplotlib.image.imsave(input_image_file+'_segnet_uncertainty.png',average_unc,cmap=cm, vmin=0, vmax=max_average_unc)

# print 'Processed: ', input_image_file

print 'Success!'
