#multiclass classification
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model as plot
from keras.utils import to_categorical as toCat

model = Sequential()
model.add(Dense(32, input_dim = 500))
model.add(Activation(activation = 'relu'))
model.add(Dense(10))
model.add(Activation(activation = 'softmax'))
model.compile(optimizer='rmsprop',loss = 'categorical_crossentropy',metrics = ['categorical_accuracy'])

data = np.random.random((1000,500))
labels = toCat(np.random.randint(10, size = (1000,1)))


score = model.evaluate(data,labels,verbose = 0)
print("Before Training: ", zip(model.metrics_names,score))

#compile train and evaluate the model
model.fit(data, labels, epochs=10,batch_size=32,verbose=0)

score = model.evaluate(data,labels,verbose=0)
print("After Training: ", zip(model.metrics_names,score) )

#save the model diagram
plot(model,to_file="model73.png",show_shapes = True)