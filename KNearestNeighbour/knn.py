import cv2
import numpy as np
import matplotlib.pyplot as plt

#DATASET TRAINING
data= np.loadtxt('/home/alphabat69/OpenCV/samples/data/letter-recognition.data', dtype= 'float32', delimiter = ',', converters= {0: lambda ch: ord(ch)-ord('A')})
train, test = np.vsplit(data,2)
#train = np.vsplit(data,1)
responses, trainData = np.hsplit(data,[1])
labels, testData = np.hsplit(test,[1])
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)

ret, result, neighbours, dist = knn.findNearest(testData, k=5)
print(result)
correct = np.count_nonzero(result == labels)
accuracy = correct*100.0/10000
print( accuracy )
#print(result)
#print('--------------------------------------')
#print(neighbours)
#print('--------------------------------------')
#print(dist)
#SAVING DATA
np.savez('data.npz',trainData=trainData, responses=responses)
with np.load('data.npz') as data:
    print( data.files )
    train = data['trainData']
    train_labels = data['responses']

#CONT
#print(train_labels)

#ret, result, neighbours, dist = knn.findNearest(testData, k=5)
#print('--------------------------------------')
#print(result)
#print('--------------------------------------')
#print(neighbours)
#print('--------------------------------------')
#print(np.hsplit(test,[1]))
print(data)
