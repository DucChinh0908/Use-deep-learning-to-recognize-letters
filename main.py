import numpy as np
import csv
import random
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers import regression
from tflearn.data_utils import to_categorical

ATC_SIZE = 32
IMG_SIZE = 28
N_CLASSES = 4
LR = 0.001
N_EPOCHS = 50

tf.compat.v1.reset_default_graph()
network = input_data(shape=[None,IMG_SIZE,IMG_SIZE,1])
network = conv_2d(network,32,3,activation='relu')
network = max_pool_2d(network,2)
network = conv_2d(network,64,3,activation='relu')
network = max_pool_2d(network,2)
network = conv_2d(network,32,3,activation='relu')
network = max_pool_2d(network,2)
network = conv_2d(network,64,3,activation='relu')
network = max_pool_2d(network,2)
network = conv_2d(network,32,3,activation='relu')
network = max_pool_2d(network,2)
network = conv_2d(network,64,3,activation='relu')
network = max_pool_2d(network,2)
network = fully_connected(network,N_CLASSES,activation='softmax')
network = regression(network)
model = tflearn.DNN(network)
data = []
s = 1
with open("DataTraining.csv","r") as csv_file:
    res = csv.reader(csv_file)
    for d in res:
        s += 1
        data.append(d)

trainData = []
trainLabel = []
for d in data:
    if(d[0] == '0') or (d[0] == '1') or (d[0] == '2') or (d[0] == '3'):
        x = np.array([int(j) for j in d[1:]])
        x = x.reshape(28,28)
        trainData.append(x)
        trainLabel.append(int(d[0]))
    else:
        break
rvs = list(range(len(trainLabel)))
random.shuffle(rvs)

trainData = np.array(trainData)
trainLabel = np.array(trainLabel)

trainData = trainData[rvs]
trainLabel = trainLabel[rvs]

trainx = trainData[:50000]
trainy = trainLabel[:50000]
valx = trainData[50000:53000]
valy = trainLabel[50000:53000]
testx = trainData[53000:]
testy = trainLabel[53000:]

BATC_SIZE = 32
IMG_SIZE = 28
N_CLASSES = 4
LR = 0.001
N_EPOCHS = 50

tf.compat.v1.reset_default_graph()

network = input_data(shape=[None,IMG_SIZE,IMG_SIZE,1])
network = conv_2d(network,32,3,activation='relu')
network = max_pool_2d(network,2)
network = conv_2d(network,64,3,activation='relu')
network = max_pool_2d(network,2)
network = conv_2d(network,32,3,activation='relu')
network = max_pool_2d(network,2)
network = conv_2d(network,64,3,activation='relu')
network = max_pool_2d(network,2)
network = conv_2d(network,32,3,activation='relu')
network = max_pool_2d(network,2)
network = conv_2d(network,64,3,activation='relu')
network = max_pool_2d(network,2)
network = fully_connected(network,N_CLASSES,activation='softmax')
network = regression(network)

model = tflearn.DNN(network)

trainx = trainx.reshape(-1,IMG_SIZE,IMG_SIZE,1)
valx = valx.reshape(-1,IMG_SIZE,IMG_SIZE,1)
testx = testx.reshape(-1,IMG_SIZE,IMG_SIZE,1)

original_test_y = testy

trainy = to_categorical(trainy,N_CLASSES)
valy = to_categorical(valy,N_CLASSES)
testy = to_categorical(testy,N_CLASSES)

model.fit(trainx,trainy,n_epoch=N_EPOCHS,validation_set=(valx,valy),show_metric=True)

model.save('ai.tflearn')

