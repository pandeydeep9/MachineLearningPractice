#Now the options for activation functions
#choices of optimization algorithhms in keras
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense,Activation

def activation_options(act):
	model = Sequential()
	model.add(Dense(1,input_dim=500))
	model.add(Activation(activation = act))
	model.compile(optimizer = "sgd", loss = 'binary_crossentropy',metrics = ['accuracy'])

	data = np.random.random((1000,500))
	labels = np.random.randint(2,size = (1000,1))

	score = model.evaluate(data,labels,verbose = 0)
	print("Activation: ", act)
	print("Before Training: ", model.metrics_names,score)
	model.fit(data,labels,epochs = 10,batch_size = 64, verbose=0)
	score = model.evaluate(data,labels,verbose = 0)
	print ("After Training: ", model.metrics_names,score, '\n')

activation_options("relu")
activation_options("tanh")
activation_options("sigmoid")
activation_options("hard_sigmoid")
activation_options("linear")
