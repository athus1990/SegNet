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

# ~~~~~~~PARAMS~~~~~~~~~~~~~~~~~~~~~
colours = 'Scripts/camvid12.png'
video = '/home/Athma/Downloads/SegNet/InputOutput/vid5.avi' #Suburb
#video ='/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151207_120040.MP4' #InnerCity_Traffic
#video='/home/Athma/Downloads/Professordata/Videos/2015_1002_104637_004.MOV'#Highway
video='/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151207_115227.MP4'#InnerCity_Traffic
#video='/home/Athma/Downloads/Professordata/PEDESWORKING/Videos/20151207_120230.MP4'

model='/home/Athma/Downloads/SegNet/Models/segnet_camvid.prototxt'
weights='/home/Athma/Downloads/SegNet/Models/Training/Segnet3_iter_40000.caffemodel'
weights='/home/Athma/Downloads/SegNet/Models/Training/Segnet4_iter_40000.caffemodel'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~SETUP NET~~~~~~~~~~~~~~~~~~~~~
net = caffe.Net(model, weights, caffe.TEST)
caffe.set_mode_gpu()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~SETUP INPUT/OTPUT~~~~~~~~~~~~~~~~~~~~~
input_shape = net.blobs['data'].data.shape
output_shape = net.blobs['argmax'].data.shape
label_colours = cv2.imread(colours).astype(np.uint8)
cap = cv2.VideoCapture(video)  # Change this to your webcam ID, or file name for your video file
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if cap.isOpened():  # try to get the first frame
	rval, frame = cap.read()
else:
	rval = False
i = 1
while rval:
	print 'FRAME = '+str(i)
	i=i+1
	#READ AN IMAGE
	start = time.time()
	rval, frame = cap.read()
	frame = cv2.GaussianBlur(frame.astype(np.uint8),(3,3),0)#MEDIUM BLUR
	end = time.time()
	print '%30s' % 'Grabbed camera frame in ', str((end - start) * 1000), 'ms'

	#PROCESS INPUT FOR DETECTION
	start = time.time()
	frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
	input_image = frame.transpose((2, 0, 1))
	input_image = input_image[(2, 1, 0), :, :]
	input_image = np.asarray([input_image])
	end = time.time()
	print '%30s' % 'Resized image in ', str((end - start) * 1000), 'ms'

	#DETECT
	start = time.time()
	out = net.forward_all(data=input_image)
	end = time.time()
	print '%30s' % 'Executed SegNet in ', str((end - start) * 1000), 'ms'

	#PROCESS RESULT
	start = time.time()
	segmentation_ind = np.squeeze(net.blobs['argmax'].data)
	segmentation_ind_3ch = np.resize(segmentation_ind, (3, input_shape[2], input_shape[3]))
	segmentation_ind_3ch = segmentation_ind_3ch.transpose(1, 2, 0).astype(np.uint8)
	segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)
	cv2.LUT(segmentation_ind_3ch, label_colours, segmentation_rgb)
	segmentation_rgb2 = segmentation_rgb
	segmentation_rgb = segmentation_rgb.astype(float) / 255.0
	end = time.time()
	print '%30s' % 'Processed results in ', str((end - start) * 1000), 'ms\n'


	########LABEL CHANGE-->POINT OF INTEREST
	seg_backup=segmentation_ind.copy()
	##ROAD ONLY
	ROAD_ONLY=np.zeros_like(segmentation_rgb[0:-95,:,:])
	segmentation_ind[segmentation_ind>3]=0
	segmentation_ind[segmentation_ind<2.5]=0
	ROAD_ONLY[:,:,1]=segmentation_ind[0:-95,:]*70
	##CAR AND OTHERS 'tree(5)','signs(6)','fence(7)','vehicle(8)','pedes(9)','bike(10)','void(11)'
	segmentation_ind=seg_backup.copy()
	CAR_ONLY=np.zeros_like(segmentation_rgb[0:-95,:,:])
	segmentation_ind[segmentation_ind==0]=0
	segmentation_ind[segmentation_ind==1]=0
	segmentation_ind[segmentation_ind==2]=0
	segmentation_ind[segmentation_ind==5]=0
	segmentation_ind[segmentation_ind==6]=0
	segmentation_ind[segmentation_ind==7]=0
	segmentation_ind[segmentation_ind==8]=0
	segmentation_ind[segmentation_ind==9]=0
	segmentation_ind[segmentation_ind==10]=0
	segmentation_ind[segmentation_ind==11]=0


	# segmentation_ind[segmentation_ind<7.5]=0
	CAR_ONLY[:,:,1]=segmentation_ind[0:-95,:]*30

	###VIEW INPUT AND OUTPUT
	cv2.imshow("Input", frame)
	cv2.imshow("SegNet", segmentation_rgb)

	#Post Processing Road detection
	FRAME=frame[0:-95,:,:]
	mask1=np.zeros([180,480])
	mask2=np.ones([85,480])
	mask=np.vstack((mask1,mask2))

	A1=cv2.cvtColor(FRAME.astype(np.uint8),cv2.COLOR_RGB2GRAY)
	maskgray = cv2.inRange(A1, 70, 120)
	A1 = cv2.bitwise_and(A1, A1, mask = maskgray)
	A1=cv2.erode(A1,np.ones((3,3)))
	A2 = cv2.GaussianBlur(A1.astype(np.uint8),(3,3),0)#MEDIUM BLUR
	A3 = cv2.GaussianBlur(A1.astype(np.uint8),(5,5),0)#FULL BLUR

	EDGES_2ch= cv2.Canny(A1,10,200)
	EDGES_2ch = cv2.bitwise_and(EDGES_2ch,EDGES_2ch,mask = mask.astype(np.uint8))
	A1=EDGES_2ch.copy()#NO BLUR

	EDGES_2ch= cv2.Canny(A2,10,200)
	EDGES_2ch = cv2.bitwise_and(EDGES_2ch,EDGES_2ch,mask = mask.astype(np.uint8))
	A2=EDGES_2ch.copy()#MEDIUM BLUR

	EDGES_2ch= cv2.Canny(A3,10,200)
	EDGES_2ch = cv2.bitwise_and(EDGES_2ch,EDGES_2ch,mask = mask.astype(np.uint8))
	A3=EDGES_2ch.copy()#FULL BLUR


	##NOW REMOVE CARS FROM EDGE MAP FOR ACCURACY
	carmap = cv2.GaussianBlur(CAR_ONLY.astype(np.uint8),(5,5),0)
	carmap=cv2.cvtColor(carmap,cv2.COLOR_RGB2GRAY)
	ret,carmap=cv2.threshold(carmap.copy(),10,255,cv2.THRESH_BINARY)
	A1 = cv2.bitwise_and(A1,A1,mask = carmap.astype(np.uint8))
	A2 = cv2.bitwise_and(A2,A2,mask = carmap.astype(np.uint8))
	A3 = cv2.bitwise_and(A3,A3,mask = carmap.astype(np.uint8))

	##ADD CARS TO IMAGE
	segmentation_ind=seg_backup.copy()
	CAR_ONLY=np.zeros_like(segmentation_rgb[0:-95,:,:])
	segmentation_ind[segmentation_ind!=8]=0
	CAR_ONLY[:,:,1]=segmentation_ind[0:-95,:]*30
	carmap = cv2.GaussianBlur(CAR_ONLY.astype(np.uint8),(5,5),0)
	carmap=cv2.cvtColor(carmap,cv2.COLOR_RGB2GRAY)
	carmap = cv2.bitwise_and(carmap.astype(np.uint8),carmap.astype(np.uint8),mask = mask.astype(np.uint8))
	cnt,hire=cv2.findContours(carmap.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	dummy=np.zeros_like(carmap)
	cv2.drawContours(dummy,cnt,-1,255,thickness=1)

	##ATTACH SEGNET ROAD to FRAME
	ROAD_ONLY_GRAY=cv2.cvtColor(ROAD_ONLY.astype(np.uint8),cv2.COLOR_RGB2GRAY)
	ROAD_ONLY_GRAY = cv2.bitwise_and(ROAD_ONLY_GRAY.copy(),ROAD_ONLY_GRAY.copy(),mask = mask.astype(np.uint8))
	ret,ROAD_ONLY_GRAY=cv2.threshold(ROAD_ONLY_GRAY.copy(),10,255,cv2.THRESH_BINARY)
	A1_ROAD_GRAY=ROAD_ONLY_GRAY.copy()
	A2_ROAD_GRAY = cv2.GaussianBlur(ROAD_ONLY_GRAY.astype(np.uint8),(3,3),0)#MEDIUM BLUR
	A3_ROAD_GRAY = cv2.GaussianBlur(ROAD_ONLY_GRAY.astype(np.uint8),(5,5),0)#FULL BLUR
	#Contour Generation
	contours1, hierarchy = cv2.findContours(A1_ROAD_GRAY,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	contours2, hierarchy = cv2.findContours(A2_ROAD_GRAY,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	contours3, hierarchy = cv2.findContours(A3_ROAD_GRAY,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#Contours Post processing
	polypnt1=[]
	polypnt2=[]
	polypnt3=[]
	pixelpoint1=[]
	pixelpoint2=[]
	pixelpoint3=[]
	[polypnt1.append(cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)) for cnt in contours1 if cv2.contourArea(cnt)>20]
	[polypnt2.append(cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)) for cnt in contours2 if cv2.contourArea(cnt)>20]
	[polypnt3.append(cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)) for cnt in contours3 if cv2.contourArea(cnt)>20]
	[pixelpoint1.append(cnt) for cnt in contours1 if cv2.contourArea(cnt)>3]
	[pixelpoint2.append(cnt) for cnt in contours2 if cv2.contourArea(cnt)>3]
	[pixelpoint3.append(cnt) for cnt in contours3 if cv2.contourArea(cnt)>3]
	polypnt1 = sorted(polypnt1, key = cv2.contourArea, reverse = True)[:10]
	polypnt2 = sorted(polypnt2, key = cv2.contourArea, reverse = True)[:10]
	polypnt3 = sorted(polypnt3, key = cv2.contourArea, reverse = True)[:10]
	pixelpoint1 = sorted(pixelpoint1, key = cv2.contourArea, reverse = True)[:10]
	pixelpoint2 = sorted(pixelpoint2, key = cv2.contourArea, reverse = True)[:10]
	pixelpoint3 = sorted(pixelpoint3, key = cv2.contourArea, reverse = True)[:10]

	##SELECT EDGE TO CHECK
	edgeimg=A2.copy()
	AA1=cv2.addWeighted(edgeimg,1,dummy,1,0)
	edgeimg=AA1.copy()
	#cv2.imshow('test3',AA1)
	nPointsparam=40

	cv2.imshow("EDGES", edgeimg)
	ret,thresh=cv2.threshold(edgeimg,20,255,cv2.THRESH_BINARY_INV)
	lines = cv2.HoughLines(edgeimg,1,np.pi/180,nPointsparam)
	if(lines!=None):
		XYLIST=[]
		L=lines[0]
		X2=((np.cos(L[:,1])*L[:,0])+1000*(np.sin(L[:,1])))
		Y2=((np.sin(L[:,1])*L[:,0])-1000*(np.cos(L[:,1])))
		Y1=((np.sin(L[:,1])*L[:,0])+1000*(np.cos(L[:,1])))
		X1=((np.cos(L[:,1])*L[:,0])-1000*(np.sin(L[:,1])))
		ANGLE=np.arctan2(Y2-Y1,X2-X1)*(180/np.pi)
		if len(ANGLE)>2:
			bins=np.arange(min(ANGLE), max(ANGLE), 5)
			ind = np.digitize(ANGLE, bins)
			ANGLE_IND=bins[ind-1]
			bins=np.arange(min(L[:,0]), max(L[:,0]), 30)
			ind = np.digitize(L[:,0], bins)
			RHOS_IND=bins[ind-1]
			VALS= list(zip(ANGLE_IND,RHOS_IND))
			SET=list(set(zip(ANGLE_IND,RHOS_IND)))
			for s in SET:
				indices = [ii for ii, x in enumerate(VALS) if x ==s]
				x1=np.mean(X1[indices])
				x2=np.mean(X2[indices])
				y1=np.mean(Y1[indices])
				y2=np.mean(Y2[indices])
				angle=np.arctan2(y2-y1,x2-x1)*(180/np.pi)
				if((angle>-25 and angle <-5) or (angle >5 and angle<25)):
					for k in np.arange(0.1,1,0.1):
						c=k*(x2)+(1-k)*(x1)
						r=k*(y2)+(1-k)*(y1)
						for cnt in polypnt1:
							dist=cv2.pointPolygonTest(cnt,(c,r),True)
							if(dist>=0):
								#cv2.line(FRAME, (x1,y1), (x2,y2), (0,0,255), 1)
								XYLIST.append([(x1,y1),(x2,y2)])
								break
		else:
			for j in range(0,len(X1)):
				angle=np.arctan2(Y2[j]-Y1[j],X2[j]-X1[j])*(180/np.pi)
				if((angle>-25 and angle <-5) or (angle >5 and angle<25)):
					for k in np.arange(0.1,1,0.1):
						c=k*(X2[j])+(1-k)*(X1[j])
						r=k*(Y2[j])+(1-k)*(Y1[j])
						for cnt in polypnt1:
							if(cv2.pointPolygonTest(cnt,(c,r),False)>=0):
								#cv2.line(FRAME, (X1[j],Y1[j]), (X2[j],Y2[j]), (0,0,255), 1)
								XYLIST.append([(X1[j],Y1[j]),(X2[j],Y2[j])])
								break


	else:
		continue



	A3_3ch = np.zeros_like(FRAME)
	#####DRAW CONTOURS
	# cv2.drawContours(A3_3ch,polypnt1,-1,(255,0,0),thickness=1)
	# cv2.drawContours(A3_3ch,polypnt2,-1,(255,0,0),thickness=1)
	# cv2.drawContours(A3_3ch,polypnt3,-1,(255,0,0),thickness=1)
	cv2.drawContours(A3_3ch,pixelpoint1,-1,(0,100,100),thickness=-1)
	cv2.drawContours(A3_3ch,pixelpoint2,-1,(0,100,100),thickness=-1)
	cv2.drawContours(A3_3ch,pixelpoint3,-1,(0,100,100),thickness=-1)

	if(XYLIST!=[]):
		angle=[np.arctan2(X[1][1]-X[0][1],X[1][0]-X[0][0])*(180/np.pi) for X in XYLIST ]
		bins=np.linspace(min(angle), max(angle), 2)
		ind = np.digitize(angle, bins)
		ANGLE_IND=bins[ind-1]
		SET=list(set(ANGLE_IND))

		for s in SET:
			indices = [ii for ii, x in enumerate(ANGLE_IND) if x ==s]
			x1=0
			x2=0
			y1=0
			y2=0
			for kkk in indices:
				line1=XYLIST[kkk]
				x1=x1+line1[0][0]
				x2=x2+line1[1][0]
				y1=y1+line1[0][1]
				y2=y2+line1[1][1]
		# line1=XYLIST[np.array(angle).argmin()]
		# line2=XYLIST[np.array(angle).argmax()]
		# cv2.line(FRAME, (line1[0][0],line1[0][1]), (line1[1][0],line1[1][1]), (0,0,255), 2)
		# cv2.line(FRAME, (line2[0][0],line2[0][1]), (line2[1][0],line2[1][1]), (0,0,255), 2)
		# for kk in range(0,len(XYLIST)):
			#######DRAW LINES
			#cv2.line(FRAME, (int(x1/len(indices)),int(y1/len(indices))), (int(x2/len(indices)),int(y2/len(indices))), (0,0,180), 2)
	#
	xlist=list()
	mlist=list()
	# for cnt in pixelpoint1:
	font = cv2.FONT_HERSHEY_SIMPLEX
	for cnt in pixelpoint1[0:5]:
		print len(cnt)
		if len(cnt)>5:
			(x,y),(M,m),a = cv2.fitEllipse(cnt)
			xlist.append(x)
			###########DRAW ELLIPSE
			#cv2.ellipse(FRAME,((x,y),(M,m),a),(0,255,0),2)
			#cv2.putText(FRAME, str(m), (int(x),int(y)), font, 0.5, (0, 0, 0), 2)
			if(m>350):
				mlist.append((m,x,y))
	if(mlist!=[]):
		cv2.putText(FRAME, str('Intersection'), (150,20), font, 0.5, (0, 0, 0), 2)
		#cv2.putText(FRAME, str(mlist[0][1]), (150,40), font, 0.5, (0, 0, 0), 2)


	#A3_3ch_GRAY=cv2.cvtColor(A3_3ch.astype(np.uint8),cv2.COLOR_RGB2GRAY)

	AA1=cv2.addWeighted(FRAME,1,A3_3ch,1,0)
	cv2.imshow('hough',AA1)




#HIDDEN PREVIOUS VERSION


	#Visualise EDGES(Uncomment)
	# A1_3ch = np.zeros_like(FRAME)
	# cv2.drawContours(A1_3ch,pixelpoint1,-1,(80,0,0),thickness=1)
	# cv2.drawContours(A1_3ch,pixelpoint2,-1,(150,0,0),thickness=1)
	# cv2.drawContours(A1_3ch,pixelpoint3,-1,(255,0,0),thickness=1)
	# cv2.imshow('working1',A1_3ch)
	#
	# A2_3ch = np.zeros_like(FRAME)
	# cv2.drawContours(A2_3ch,polypnt1,-1,(80,0,0),thickness=1)
	# cv2.drawContours(A2_3ch,polypnt2,-1,(150,0,0),thickness=1)
	# cv2.drawContours(A2_3ch,polypnt3,-1,(255,0,0),thickness=1)
	# cv2.imshow('working2',A2_3ch)

	#FOCUS ON MAX BLUR alone
	# C=ROAD_ONLY[:,:,1]
	# ret,thresh= cv2.threshold(C.astype(np.uint8),100,1,cv2.THRESH_BINARY)
	# index=np.nonzero(thresh)
	# I=np.squeeze(index,2)
	# row=thresh.shape[0]
	# col=thresh.shape[1]
	# cntlisttofill=[]
	# lines = cv2.HoughLines(EDGES_2ch,1,np.pi/180,100)
	# A3_3ch = np.zeros_like(FRAME)
	# for rho,theta in lines[0]:
	# 	if(theta<np.pi/20. or theta >19.*np.pi/20.0):
	# 		a = np.cos(theta)
	# 		b = np.sin(theta)
	# 		x0 = a*rho
	# 		y0 = b*rho
	# 		x1 = int(x0 + 1000*(-b))
	# 		y1 = int(y0 + 1000*(a))
	# 		x2 = int(x0 - 1000*(-b))
	# 		y2 = int(y0 - 1000*(a))
	# 		cv2.line(FRAME,(x1,y1),(x2,y2),(0,0,255),2)
	#
	# cv2.imshow('tets',FRAME)
	# # for cnt in pixelpoint3:
	# # 	csum=0#no of pixels within conour
	# # 	fsum=0#number of pixels within contour that are filled
	# # 	for r in range(0,row):
	# # 		for c in range(0,col):
	# # 			if(cv2.pointPolygonTest(cnt,(c,r),False)>=0):
	# # 				csum=csum+1
	# # 	for i in range(0,I.shape[1]):
	# # 		if(cv2.pointPolygonTest(cnt,(I[0,i],I[1,i]),False)>=0):
	# # 			fsum=fsum+1
	# # 	if(fsum/csum>0.5):
	# # 		cntlisttofill.append(cnt)
	# #
	# # A3_3ch = np.zeros_like(FRAME)
	# # cv2.drawContours(A3_3ch,cntlisttofill,-1,(255,0,0),thickness=-1)
	# # cv2.imshow('working3',A3_3ch)

	# V=np.zeros_like(FRAME)
	# V[:,:,0]=EDGES2
	# V[:,:,1]=EDGES2
	# V[:,:,2]=EDGES2
	#
	#
	#
	#
	#
	#
	#
	#
	# blur = cv2.GaussianBlur(ROAD_ONLY.astype(np.uint8),(5,5),0)
	# blur=cv2.cvtColor(blur,cv2.COLOR_RGB2GRAY)
	# edges = cv2.Canny(blur,100,200)
	# #REMOVE SKY PREDICTIONS
	# mask1=np.zeros([145,480])
	# mask2=np.ones([145,480])
	# mask=np.vstack((mask1,mask2))
	# EDGES = cv2.bitwise_and(edges,edges,mask = mask.astype(np.uint8))
	# BLUR= cv2.bitwise_and(blur,blur,mask = mask.astype(np.uint8))
	#
	#
	# Dummy=ROAD_ONLY.copy()
	# contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# contours2, hierarchy = cv2.findContours(EDGES,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# contours3, hierarchy = cv2.findContours(BLUR,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#
	# pnt=[]
	# pnt2=[]
	# pnt3=[]
	# for cnt in contours:
	# 	if(cv2.contourArea(cnt)>10):
	# 		pnt.append(cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True))
	# for cnt in contours2:
	# 	if(cv2.contourArea(cnt)>10):
	# 		pnt2.append(cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True))
	# for cnt in contours3:
	# 	if(cv2.contourArea(cnt)>10):
	# 		#pnt3.append(cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True))
	# 		pnt3.append(cnt)
	#
	# cv2.drawContours(Dummy,pnt,-1,(255,0,0),-1)
	# cv2.imshow('tets2',Dummy)
	# cv2.drawContours(FRAME,pnt2,-1,(255,0,0),-1)
	# cv2.imshow('OUTPUT',FRAME)
	# cv2.drawContours(EDGES2,pnt2,-1,(255,0,0),-1)
	# cv2.imshow('OUTPUT2',EDGES2)
	# # cv2.drawContours(V,pnt3,-1,(125,0,0),thickness=-1)
	# # cv2.imshow('OUTPUT3',V)
	# BLUR2=np.zeros_like(V)
	# BLUR2[:,:,0]=(BLUR*255).astype(np.uint8)
	# ADDWEIGHTED=cv2.addWeighted(V,1,BLUR2,0.6,0)
	# cv2.imshow('OUTPUT4',ADDWEIGHTED)
	key = cv2.waitKey(5)
	if key == 27:  # exit on ESC
		break
	#time.sleep(1)

cap.release()
cv2.destroyAllWindows()
