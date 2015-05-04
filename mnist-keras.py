# Author: Hussein Al-barazanchi
# reading and saving data are based on the code from Andy

# import numpy and pandas for array manipulationa and csv files
import numpy as np
import pandas as pd


# import keras necessary classes
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils


# Creating the model which consists of 3 conv layers followed by
# 2 fully conntected layers
print('creating the model')

# Sequential wrapper model
model = Sequential()

# first convolutional layer
model.add(Convolution2D(32,1,2,2))
model.add(Activation('relu'))


# second convolutional layer
model.add(Convolution2D(16, 32, 2, 2))
model.add(Activation('relu')) 

model.add(MaxPooling2D(poolsize=(2,2)))

# third convolutional layer
model.add(Convolution2D(8, 16, 2, 2))
model.add(Activation('relu'))

model.add(MaxPooling2D(poolsize=(2,2)))


# convert convolutional filters to flatt so they can be feed to 
# fully connected layers
model.add(Flatten())


# first fully connected layer
model.add(Dense(8*6*6, 32, init='lecun_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.25))


# last fully connected layer which output classes
model.add(Dense(32, 10, init='lecun_uniform'))
model.add(Activation('softmax'))

# setting sgd optimizer parameters
sgd = SGD(lr=0.09, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)



print('read data')
# reading training data
training = pd.read_csv('/home/mnist/train.csv')

# split training labels and pre-process them
training_targets = training.ix[:,0].values.astype('int32')
training_targets = np_utils.to_categorical(training_targets) 

# split training inputs
training_inputs = (training.ix[:,1:].values).astype('float32')

# read testing data
testing_inputs = (pd.read_csv('/home/mnist/test.csv').values).astype('float32')



# pre-process training and testing data
max_value = np.max(training_inputs)
training_inputs /= max_value
testing_inputs /= max_value

mean_value = np.std(training_inputs)
training_inputs -= mean_value
testing_inputs -= mean_value

# reshaping training and testing data so it can be feed to convolutional layers
training_inputs = training_inputs.reshape(training_inputs.shape[0], 1, 28, 28)
testing_inputs = testing_inputs.reshape(testing_inputs.shape[0], 1, 28, 28)





print("Starting training")
model.fit(training_inputs, training_targets, nb_epoch=10, batch_size=1000, validation_split=0.1, show_accuracy=True)


print("Generating predections")
preds = model.predict_classes(testing_inputs, verbose=0)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

print('Saving predictions')
write_preds(preds, "keras-mlp.csv")
