import cv2
import numpy as np
import pytesseract
from PIL import Image

cap=cv2.VideoCapture(0)
while True:
	_,fr	=	cap.read()
	frame	=	cv2.cvtColor(fr , cv2.COLOR_BGR2GRAY)
	
	
	
	#kernel	=	np.ones((1,1) , np.uint8)
	#frame	=	cv2.dilate(frame , kernel , iterations=1)
	#frame	=	cv2.erode(frame , kernel , iterations=1)
	#frame	=	cv2.adaptiveThreshold(frame , 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY , 9 , 2)
	
	_,frame	=	cv2.threshold(frame , 100 , 255 , cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	frame	=	cv2.medianBlur(frame , 3)
	
	
	#frame2	=	cv2.adaptiveThreshold(frame , 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY , 11 , 2)
	result	=	pytesseract.image_to_string(Image.fromarray(frame) , lang='eng')
	if len(result) is not 0:
		print(result)
	cv2.imshow('Frame',frame)
	#scv2.imshow('fff',frame2)
	if cv2.waitKey(5) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break
