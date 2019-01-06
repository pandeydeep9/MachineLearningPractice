#logistic regression -> neural network with a single unit
import numpy as np
import theano 
import theano.tensor as T
import sklearn.metrics

#l2 regularization
def l2(x):
	return T.sum(x**2)

examples = 1000
features = 100
lambda_val = 0.01

Dataset = (np.random.randn(examples,features),np.random.randint(size = examples,low = 0,high = 2))
training_steps = 500

x = T.dmatrix('x')
y = T.dvector('y')
w = theano.shared(np.random.randn(features),name = 'w')
b = theano.shared(0.,name = 'b')

sig = 1 / (1 + T.exp(-T.dot(x,w) - b)) # sigmoid function
error = T.nnet.binary_crossentropy(sig,y)
loss = error.mean() + lambda_val * l2(w)
prediction = sig >0.5
gw, gb = T.grad(loss, [w,b])

alpha = 0.1
train = theano.function(inputs = [x,y],outputs = [sig,error], updates = ((w,w - alpha * gw),(b, b- alpha * gb)))
predict = theano.function(inputs = [x],outputs = prediction)

print("Accuracy before training: ", sklearn.metrics.accuracy_score( Dataset[1], predict( Dataset[0] ) ) ) 

#train
for i in xrange(training_steps):
	prediction, error = train(Dataset[0],Dataset[1])

print("Accuracy after training: ", sklearn.metrics.accuracy_score(Dataset[1], predict( Dataset[0] ) ) )

#Experiment

import pylab
pylab.plot(predict(Dataset[0][0:20]),'r')
pylab.plot((Dataset[1][0:20]),'g')

pylab.show()