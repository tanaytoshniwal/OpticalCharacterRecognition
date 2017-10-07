import cv2
import numpy as np
import pytesseract
from resizeimage import resizeimage
from PIL import Image
import os

src_path='/home/alphabat69/Desktop/AlphaBAT69/Python/ocr/'
#cv2.imshow('image',img)

def get(path):
	img	=	cv2.imread(path)
	img	=	cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
	
	#removing noise
	kernel	=	np.ones((1,1) , np.uint8)
	img	=	cv2.dilate(img , kernel , iterations=1)
	img	=	cv2.erode(img , kernel , iterations=1)
	cv2.imwrite(src_path+'res/a.png' , img)
	
	#applying threshold
	img	=	cv2.adaptiveThreshold(img , 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY , 11 , 2)
	cv2.imwrite(src_path+'res/b.png' , img)
	
	#recognition using tesseract engine
	result 	=	pytesseract.image_to_string(Image.open(src_path + 'res/b.png'))
	return result

print(get(src_path+'img1.png'))
