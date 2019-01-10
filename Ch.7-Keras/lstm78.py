import numpy as np 
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation,Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.utils import plot_model as plot

max_features = 20000
maxLen = 80
b_Size =32

#prepare the dataset
(x_train, y_train),(x_test,y_test) = imdb.load_data(nb_words=max_features)
x_train = sequence.pad_sequences(x_train,maxLen)
x_test = sequence.pad_sequences(x_test,maxLen)

#lstm
model = Sequential()
model.add(Embedding(max_features,128,dropout = 0.2))
model.add(LSTM(128,dropout_W = 0.2,dropout_U = 0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#compile the model
model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

#train
model.fit(x_train,y_train,batch_size = b_Size, verbose = 1,nb_epoch = 1,validation_data= (x_test,y_test))

#evaluate
score = model.evaluate(x_test,y_test,batch_size = b_Size)
print("test Metrics: ", model.metrics_names, score)
plot(model,to_file = 'lstm78.png',show_shapes = True)
