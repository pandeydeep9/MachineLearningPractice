import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model#/ as plot

#We are making a sequential model
model = Sequential()
#single layer NN. one dense layer with 500 input units 1 output unit
model.add(Dense(1, input_dim = 500))
#use a sigmoid activation on the output to classify the input
model.add(Activation(activation = 'sigmoid'))
#how to improve the model with training
model.compile(optimizer='rmsprop',loss = 'binary_crossentropy',metrics = ['accuracy'])

#1000 examples to look at
data = np.random.random((1000,500))
labels = np.random.randint(2, size = (1000,1))


score = model.evaluate(data,labels,verbose = 0)
print("Before Training: ", zip(model.metrics_names,score))

#train
model.fit(data, labels, nb_epoch=10,batch_size=32,verbose=0)

score = model.evaluate(data,labels,verbose=0)
print("After Training: ", zip(model.metrics_names,score) )

#save the model diagram
plot_model(model,to_file="model71.png",show_shapes = True)