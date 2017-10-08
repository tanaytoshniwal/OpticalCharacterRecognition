import cv2
import numpy as np
import matplotlib.pyplot as plt
data= np.loadtxt('/home/alphabat69/OpenCV/samples/data/letter-recognition.data', dtype= 'float32', delimiter = ',', converters= {0: lambda ch: ord(ch)-ord('A')})
train, test = np.vsplit(data,2)
responses, trainData = np.hsplit(train,[1])
labels, testData = np.hsplit(test,[1])
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
ret, result, neighbours, dist = knn.findNearest(testData, k=5)
correct = np.count_nonzero(result == labels)
accuracy = correct*100.0/10000
print( accuracy )
np.savez('knn_data_handwritten.npz',trainData=trainData, responses=responses)
with np.load('knn_data_handwritten.npz') as data:
    print( data.files )
    train = data['trainData']
    train_labels = data['responses']
