#choices of optimization algorithhms in keras
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense,Activation

def optimizer_options(opt):
	model = Sequential()
	model.add(Dense(1,input_dim=500))
	model.add(Activation(activation = 'sigmoid'))
	model.compile(optimizer = opt, loss = 'binary_crossentropy',metrics = ['accuracy'])

	data = np.random.random((1000,500))
	labels = np.random.randint(2,size = (1000,1))

	score = model.evaluate(data,labels,verbose = 0)
	print("Optimizer: ", opt)
	print("Before Training: ", model.metrics_names,score)
	model.fit(data,labels,epochs = 10,batch_size = 64, verbose=0)
	score = model.evaluate(data,labels,verbose = 0)
	print ("After Training: ", model.metrics_names,score, '\n')

optimizer_options("sgd")
optimizer_options("rmsprop")
optimizer_options("adagrad")
optimizer_options("adadelta")
optimizer_options("adam")
optimizer_options("adamax")
optimizer_options("nadam")