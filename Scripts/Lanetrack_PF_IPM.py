#! /usr/bin/env python
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
import glob
import numpy as np



# Make sure that caffe is on the python path:
caffe_root = '/home/Athma/Downloads/SegNet/caffe-segnet/'  # Change this to the absolute directoy to SegNet Caffe
sys.path.insert(0, caffe_root + 'python')
import caffe

sys.path.insert(0, '/home/Athma/Downloads/SegNet/Scripts/')
from ParticleFilter import ParticleFilter


global oldxlvals
global oldxrvals
oldxlvals=1.0
oldxrvals=1.0

####TRACKING intialisation
cv2.namedWindow('Lane Markers')
intercepts = []
xl_int_pf=ParticleFilter(N=1000,x_range=(0,1500),sensor_err=1,par_std=100)
xl_phs_pf=ParticleFilter(N=1000,x_range=(15,90),sensor_err=0.3,par_std=1)
xr_int_pf=ParticleFilter(N=1000,x_range=(100,1800),sensor_err=1,par_std=100)
xr_phs_pf=ParticleFilter(N=1000,x_range=(15,90),sensor_err=0.3,par_std=1)

#tracking queues
xl_int_q = [0]*15
xl_phs_q = [0]*15
count = 0


def mainpartickletrack(orig_img):
		global oldxrvals
		global oldxlvals
		#orig_img=orig_img[0:-300,:,:]
		# Scale down the image - Just for better display.
		orig_height,orig_width=orig_img.shape[:2]
		# orig_img=cv2.resize(orig_img,(orig_width/2,orig_height/2),interpolation = cv2.INTER_CUBIC)
		# orig_height,orig_width=orig_img.shape[:2]
		# Part of the image to be considered for lane detection
		upper_threshold=0.4
		lower_threshold=0.2
		# Copy the part of original image to temporary image for analysis.
		img=orig_img[int(upper_threshold*orig_height):int((1- lower_threshold)*orig_height),:]
		#img=orig_img.copy()
		# Convert temp image to GRAY scale
		img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
		height,width=img.shape[:2]
		# Image processing to extract better information form images.
		# Adaptive Biateral Filter:
		#img = cv2.adaptiveBilateralFilter(img,ksize=(5,5),sigmaSpace=2)
		# Equalize the histogram to account for better contrast in the images.
		#img = cv2.equalizeHist(img);
		# Apply Canny Edge Detector to detect the edges in the image.
		bin_img = cv2.Canny(img,30,60,apertureSize = 5)
		cv2.imshow('edges',bin_img)
		#Thresholds for lane detection. Emperical values, detected from trial and error.
		xl_low = int(-1*orig_width) # low threshold for left x_intercept
		xl_high = int(0.8*orig_width) # high threshold for left x_intercept
		xr_low = int(0.2*orig_width)  # low threshold for right x_intercept
		xr_high = int(2*orig_width) # high threshold for right x_intercept
		xl_phase_threshold = 15  # Minimum angle for left x_intercept
		xr_phase_threshold = 14  # Minimum angle for right x_intercept
		xl_phase_upper_threshold = 80  # Maximum angle for left x_intercept
		xr_phase_upper_threshold = 80  # Maximum angle for right x_intercept

		# Arrays/Containers for intercept values and phase angles.
		xl_arr = np.zeros(xl_high-xl_low)
		xr_arr = np.zeros(xr_high-xr_low)
		xl_phase_arr = []
		xr_phase_arr = []
		# Intercept Bandwidth: Used to assign weights to neighboring pixels.
		intercept_bandwidth = 6

		# Run Probabilistic Hough Transform to extract line segments from Binary image.
		lines=cv2.HoughLinesP(bin_img,rho=1,theta=np.pi/180,threshold=10,minLineLength=10,maxLineGap=200)

		# Loop for every single line detected by Hough Transform
		# print len(lines[0])
		if(lines!=None):
			for x1,y1,x2,y2 in lines[0]:
				if(x1<x2 and y1>y2):
					norm = cv2.norm(float(x1-x2),float(y1-y2))
					phase = cv2.phase(np.array(x2-x1,dtype=np.float32),np.array(y1-y2,dtype=np.float32),angleInDegrees=True)
					# if(phase<xl_phase_threshold or phase > xl_phase_upper_threshold or x1 > 0.5 * orig_width): #Filter out the noisy lines
					#     continue
					xl = int(x2 - (height+lower_threshold*orig_height-y2)/np.tan(phase*np.pi/180))
					# Show the Hough Lines
					# cv2.line(orig_img,(x1,y1+int(orig_height*upper_threshold)),(x2,y2+int(orig_height*upper_threshold)),(0,0,255),2)

					# If the line segment is a lane, get weights for x-intercepts
					try:
						for i in range(xl - intercept_bandwidth,xl + intercept_bandwidth):
							xl_arr[i-xl_low] += (norm**0.5)*y1*(1 - float(abs(i - xl))/(2*intercept_bandwidth))*(phase**2)
					except IndexError:
						# print "Debug: Left intercept range invalid:", xl
						continue
					xl_phase_arr.append(phase[0][0])

				elif(x1<x2 and y1<y2):
					norm = cv2.norm(float(x1-x2),float(y1-y2))
					phase = cv2.phase(np.array(x2-x1,dtype=np.float32),np.array(y2-y1,dtype=np.float32),angleInDegrees=True)
					# if(phase<xr_phase_threshold or phase > xr_phase_upper_threshold or x2 < 0.5 * orig_width): #Filter out the noisy lines
					#     continue
					xr = int(x1 + (height+lower_threshold*orig_height-y1)/np.tan(phase*np.pi/180))
					# Show the Hough Lines
					# cv2.line(orig_img,(x1,y1+int(orig_height*upper_threshold)),(x2,y2+int(orig_height*upper_threshold)),(0,0,255),2)
					# If the line segment is a lane, get weights for x-intercepts
					try:
						for i in range(xr - intercept_bandwidth,xr + intercept_bandwidth):
							xr_arr[i-xr_low] += (norm**0.5)*y2*(1 - float(abs(i - xr))/(2*intercept_bandwidth))*(phase**2)
					except IndexError:
						# print "Debug: Right intercept range invalid:", xr
						continue
					xr_phase_arr.append(phase[0][0])
				else:
					pass # Invalid line - Filter out orizontal and other noisy lines.

			# Sort the phase array and get the best estimate for phase angle.
			try:
				xl_phase_arr.sort()
				if(len(xl_phase_arr)==0):
					xl_phase=[]
				else:
					xl_phase =  xl_phase_arr[-1] if (xl_phase_arr[-1] < np.mean(xl_phase_arr) + np.std(xl_phase_arr)) else np.mean(xl_phase_arr) + np.std(xl_phase_arr)
			except IndexError:
				# print "Debug: ", fname + " has no left x_intercept information"
				pass
			try:
				xr_phase_arr.sort()
				if(len(xr_phase_arr)==0):
					xr_phase=[]
				else:
					xr_phase =  xr_phase_arr[-1] if (xr_phase_arr[-1] < np.mean(xr_phase_arr) + np.std(xr_phase_arr)) else np.mean(xr_phase_arr) + np.std(xr_phase_arr)
			except IndexError:
				# print "Debug: ", fname + " has no right x_intercept information"
				pass

			# Get the index of x-intercept (700 is for positive numbers for particle filter.)
			pos_int = np.argmax(xl_arr)+xl_low+700
			# Apply Particle Filter.
			xl_int = xl_int_pf.filterdata(data=pos_int)
			if(str(xl_phase)!='[]'):
				xl_phs = xl_phs_pf.filterdata(data=xl_phase)
				oldxlvals=xl_phs.astype(np.int)
			else:
				xl_phs=oldxlvals
				# Draw lines for display
			cv2.line(orig_img,
					(int(xl_int-700), orig_height),
					(int(xl_int-700) + int(orig_height*0.3/np.tan(xl_phs*np.pi/180)),int(0.7*orig_height)),(0,255,255),2)
			# Apply Particle Filter.
			xr_int = xr_int_pf.filterdata(data=np.argmax(xr_arr)+xr_low)
			if(str(xr_phase)!='[]'):
				xr_phs = xr_phs_pf.filterdata(data=xr_phase)
				oldxrvals=xr_phs.astype(np.int)
			else:
				xr_phs=oldxrvals
			# Draw lines for display
			cv2.line(orig_img,
					(int(xr_int), orig_height),
					(int(xr_int) - int(orig_height*0.3/np.tan(xr_phs*np.pi/180)),int(0.7*orig_height)),(0,255,255),2)

			# print "Degbug: %5d\t %5d\t %5d\t %5d %s"%(xl_int-700,np.argmax(xl_arr)+xl_low,xr_int,np.argmax(xr_arr)+xr_low,fname)
			# intercepts.append((os.path.basename(fname), xl_int[0]-700, xr_int[0]))

			# Show image
		cv2.imshow('Lane Markers', orig_img)

def showonlyroads(segmentation_rgb, segmentation_ind):
	road = np.zeros_like(segmentation_rgb)
	segmentation_ind[segmentation_ind > 3] = 0
	segmentation_ind[segmentation_ind < 2.5] = 0
	road[:, :, 1] = segmentation_ind * 60
	road[0:200, :, 1] = np.zeros((200, 480))
	contours2, hierarchy = cv2.findContours(road[:, :, 1].astype(np.uint8), cv2.RETR_TREE,
											cv2.CHAIN_APPROX_NONE)
	pixelpoint2 = []
	[pixelpoint2.append(cnt2) for cnt2 in contours2 if cv2.contourArea(cnt2) > 10]
	pixelpoint2 = sorted(pixelpoint2, key=cv2.contourArea, reverse=True)[:5]
	A3_3ch = np.zeros_like(road)
	cv2.drawContours(A3_3ch, pixelpoint2, -1, (255, 255, 255), -1)
	road[:, :, 1] = cv2.cvtColor(A3_3ch.astype(np.uint8), cv2.COLOR_RGB2GRAY)

	return road

def genIPM(I,p,w):
	Wc = np.zeros([nRows, nCols,3])
	for c in reversed(range(0,3)):
		Id = np.double(I[:,:,c])/255.0
		[idxr,idxc]=np.unravel_index(p-1,(Id.shape[0],Id.shape[1]),order='F')
		Wc[:,:,c]= np.sum(np.multiply(Id[idxr,idxc],w),2)
	Wc = np.minimum(Wc, 1.0000)
	Wc = np.maximum(Wc, 0.0000)
	Wg = np.mean(Wc, 2)
	return [Wc,Wg]



# Import arguments


video = '/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151207_120040.MP4'  # InnerCity_Traffic
video = '/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151207_115227.MP4'  # InnerCity_Traffic
video = '/home/Athma/Downloads/20160129_093059.MP4'  # snow
video = '/home/Athma/Downloads/Professordata/specialIntersections/Feb-Mar 2016-selected/20160420_200445.MP4'
video = '/home/Athma/Downloads/Professordata/specialIntersections/Feb-Mar 2016-selected/20160420_201117.MP4'
video = '/home/Athma/Downloads/Professordata/specialIntersections/Feb-Mar 2016-selected/20160421_091814.MP4' #--working
video = '/home/Athma/Downloads/Professordata/specialIntersections/Feb-Mar 2016-selected/20160421_092344.MP4'
video = '/home/Athma/Downloads/Professordata/specialIntersections/Feb-Mar 2016-selected/20160420_200045.MP4'


fourcc = cv2.cv.CV_FOURCC(*'H264')
#outputvideo1 = cv2.VideoWriter('/home/Athma/Downloads/SegNet/InputOutput/IPMLaneResults/laneresults.avi', fourcc, 20.0, (752*3,660))

model = '/home/Athma/Downloads/SegNet/Example_Models/bayesian_segnet_camvid.prototxt'
weights = '/home/Athma/Downloads/SegNet/Models/Training/bayesian_SEGNET1_iter_60000.caffemodel'
colours = 'Scripts/camvid12.png'
data = '/home/Athma/Downloads/SegNet/CamVid/train.txt'
caffe.set_mode_gpu()
net = caffe.Net(model, weights, caffe.TEST)
input_shape = net.blobs['data'].data.shape
label_colours = cv2.imread(colours).astype(np.uint8)
i = 1
cap = cv2.VideoCapture(video)  # Change this to your webcam ID, or file name for your video file
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#IPM related setup
mat_contents = sio.loadmat('/home/Athma/Downloads/SegNet/Scripts/dataInterMap.mat', struct_as_record=False, squeeze_me=True)['data']
interpMap=mat_contents.interpMap
[nRows, nCols, dummy] = interpMap.pixels.shape

mat_contents2 = sio.loadmat('/home/Athma/Downloads/SegNet/Scripts/calibrationSession.mat', struct_as_record=False, squeeze_me=True)




if cap.isOpened():  # try to get the first frame
	rval, frame = cap.read()
else:
	rval = False
i = 1
while rval:
	print 'FRAME = ' + str(i)
	i = i + 1
	rval, frame = cap.read()
	#chopoff
	if rval:
		[Wc,Wg]=genIPM(frame,interpMap.pixels,interpMap.weights)
		laplacian = cv2.Laplacian(Wg,cv2.CV_64F)
		sobelx = cv2.Sobel(Wg,cv2.CV_64F,1,0,ksize=3)
		sobely = cv2.Sobel(Wg,cv2.CV_64F,0,1,ksize=3)
		sobel=np.sqrt(np.power(sobelx,2)+np.power(sobely,2))
		std=np.sqrt((np.sum(np.sum(np.power(sobel-np.mean(sobel),2))))/(sobel.shape[0]*sobel.shape[1]))
		newsobel=(sobel-np.mean(sobel))/std
		newsobel[newsobel<0.5]=0

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,1))
		sibeledge = cv2.morphologyEx(newsobel, cv2.MORPH_OPEN, kernel)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,2))
		sibeledge = cv2.erode(sibeledge, kernel, iterations=1)


		M = cv2.getRotationMatrix2D((newsobel.shape[1]/2,newsobel.shape[0]/2),90,1)
		dstnewsobel = cv2.warpAffine(newsobel,M,(newsobel.shape[1],newsobel.shape[0]))
		dstWc = cv2.warpAffine(Wc,M,(newsobel.shape[1],newsobel.shape[0]))
		dstsibeledge = cv2.warpAffine(sibeledge,M,(newsobel.shape[1],newsobel.shape[0]))




		cv2.imshow('ipmedges',dstnewsobel)
		cv2.imshow('ipm',dstWc)
		cv2.imshow('processed ipm edges',dstsibeledge)

		#
		frame=frame[500:-270, 700:, :]
		#frame=Wc.copy()
		frame = cv2.GaussianBlur(frame, (3, 3), 0)
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
		#
		# # cv2.imshow('seg',segmentation_rgb)
		# # cv2.imshow('uncertainity',uncertainty.astype(np.uint8))
		#
		#
		cv2.waitKey(5)
		ROAD_ONLY = showonlyroads(segmentation_rgb, ind)
		FRAME_SegNet_combined = cv2.addWeighted(frame.astype(np.uint8), 1, ROAD_ONLY.astype(np.uint8), 1, 0)
		thresh = 100
		backup = ROAD_ONLY.copy()
		road = ROAD_ONLY[:, :, 1].copy()
		road[uncertainty <= thresh] = 0
		backup[:, :, 1] = road
		PROCESSED_FRAME_SegNet_combined = cv2.addWeighted(frame.astype(np.uint8), 1, backup.astype(np.uint8), 1, 0)

		kernel = np.ones((5, 5), np.uint8)
		road = cv2.morphologyEx(backup[:, :, 1], cv2.MORPH_OPEN, kernel)
		road = cv2.dilate(road, kernel, iterations=1)
		res = cv2.bitwise_and(frame, frame, mask=road)
		cv2.imshow('res',res)

		gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)


		lanemap=np.zeros_like(res)
		YELLOW = 30
		hue = YELLOW // 2
		lower_range = np.array([hue - 10, 0, 0], dtype=np.uint8)
		upper_range = np.array([hue + 30, 255, 255], dtype=np.uint8)
		hsvimg = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsvimg, lower_range, upper_range)
		lanemap[:,:,2]=mask
		ret, threshmap = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
		lanemap[:,:,0]=threshmap
		cv2.imshow('lanemarkers',lanemap)
		res2=cv2.resize(lanemap,(1920,1080))


		mainpartickletrack(lanemap)
		#outputvideo1.write(lanemap)
		#
		#
		cv2.imshow('input1', frame)
# outputvideo1.release()

print 'Success!'
