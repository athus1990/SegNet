import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import cv2
import sys
import time
import scipy.io as sio



class Camera:
	def __init__(self,name,height,width,z,alpha,theta):
		self.name=name
		self.height=height #image height
		self.width=width #image width
		self.h=z #xamera height from ground
		self.alpha=alpha*np.pi/180.0 #veiwing angle in radians
		self.theta=theta*np.pi/180.0  # camera tilt from ground in radians

	def createpixelstoworld(self):
		den=np.sqrt(np.power(self.m-1,2)+np.power(self.n-1,2))
		alpha_u=np.arctan2(((self.n-1)/den )*(np.tan(self.alpha)))
		alpha_v=np.arctan2(((self.m-1)/den )*(np.tan(self.alpha)))

		rHorizon=np.ceil((self.m-1)/2*(1-np.tan(self.theta)/np.tan(alpha_v)))+np.ceil(self.m*0.05)
		mcropped=self.m-rHorizon+1

		xmap=np.zeros((mcropped,self.n))
		ymap=np.zeros_like(xmap)

		for r in range(0,mcropped):
			rorig=r+rHorizon+1
			rfactor=(1-2*(rorig-1)/(self.m-1))*np.tan(alpha_v)
			num=1+rfactor*np.tan(self.theta)
			den2=np.tan(self.theta)-rfactor
			xmap[r,0:self.n]=self.h*(num/den2)

			for c in range(0,self.n):
				num3=(1-2*(self.c-1)/(self.n-1))*np.tan(alpha_u)
				den3=np.sin(self.theta)-rfactor*np.cos(self.theta)
				ymap[r,c]=self.h*(num3/den3)


		self.xmap=xmap
		self.ymap=ymap





if __name__ == "__main__":
	mat_contents = sio.loadmat('dataInterMap.mat', struct_as_record=False, squeeze_me=True)
	a=mat_contents['data']
	interpMap=a.interpMap
	[nRows, nCols, dummy] = interpMap.pixels.shape
	p=interpMap.pixels
	w=interpMap.weights
	I=cv2.imread('testimg.png')
	Wc = np.zeros([nRows, nCols,3])
	for c in reversed(range(0,3)):
		Id = np.double(I[:,:,c])/255.0
		# for ii in range(0,nRows):
		# 	for jj in range(0,nCols):
		# 		pp=p[ii,jj]
		[idxr,idxc]=np.unravel_index(p-1,(Id.shape[0],Id.shape[1]),order='F')
		Wc[:,:,c]= np.sum(np.multiply(Id[idxr,idxc],w),2)
	Wc = np.minimum(Wc, 1.0000)
	Wc = np.maximum(Wc, 0.0000)
	Wg = np.mean(Wc, 2)


	print "Done ===== Done "