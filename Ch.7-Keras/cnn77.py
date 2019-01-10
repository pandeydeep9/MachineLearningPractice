#now cnn, finally with keras
import numpy as np 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import plot_model as plot

#image size
img_rows, img_cols = 28,28

#filter
nb_filters = 32

# Pooling
maxPool_size = (2,2)

#Kernel
kernel_size = (3,3)

#prepare the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
print(x_test.shape)
x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)

ip_sahpe = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255 #(in range 0-1)
x_test = x_test / 255

nb_classes = 10
y_train = np_utils.to_categorical(y_train,nb_classes)
y_test = np_utils.to_categorical(y_test,nb_classes)

#the cnn
model = Sequential()
model.add(Convolution2D(nb_filters,kernel_size[0],kernel_size[1],border_mode='valid', input_shape = ip_sahpe))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters,kernel_size[0],kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= maxPool_size))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#compile
model.compile(loss = 'categorical_crossentropy',optimizer = 'adadelta',metrics = ['accuracy'])

#train
b_size = 64
epoch = 1
model.fit(x_train,y_train,batch_size= b_size,nb_epoch = epoch, verbose = 1, validation_data = (x_test,y_test))

#evaluate
score = model.evaluate(x_test,y_test,verbose = 0)
print("TEst Metrics: ", model.metrics_names, score)
plot(model, to_file = 'cnn77.png',show_shapes = True)

