import cv2
import matplotlib.pyplot as plt
import numpy as np
data= np.loadtxt('data/letter-recognition.data', dtype= 'float32', delimiter = ',', converters= {0: lambda ch: ord(ch)-ord('A')})
responses, trainData = np.hsplit(data,[1])
knn=cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
#with np.load('data.npz') as data:
#    tdata=data['trainData']
#    tres=data['responses']
#ret, result, neighbours, dist = knn.findNearest(testData, k=5)
#knn.train(tdata, cv2.ml.ROW_SAMPLE, tres)
img = cv2.imread('data/test1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5,5), 0)
img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
a, b, c = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#ret, result, neighbours, dist = knn.findNearest(img, k=5)
print(responses)
print('--------')
string = ''
for b in b:
    x,y,w,h = cv2.boundingRect(b)
    area = cv2.contourArea(b)
    imgR = img[y : y + h,x : x + w]
    imgR = cv2.resize(imgR, (16,20000))
    npa = imgR.reshape((20000, 16))
    #imgR.shape = (2,3,4)
    npa = np.float32(npa)
    ret, result, neighbours, dist = knn.findNearest(npa, k=1)
    string = string +' '+ str(int(result[0][0]))
    #if np.any(result)==np.any(responses):
    #    print('hi')
    print('----------------------------------   ')
#print(str(x)+' '+str(y)+' '+str(w)+' '+str(h)+' '+str(area))
print(string)
#print(tdata)
#print('--------------------------')
#print(tdata.shape)#20000,16
#print(npa.shape)#30,20
#cv2.imshow('img',img1)
#cv2.imshow('image',img)
#cv2.waitKey(0)
