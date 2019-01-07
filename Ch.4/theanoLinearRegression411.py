import numpy as np
import theano 
import theano.tensor as T 
import sklearn.metrics

def l2(x):
	return T.sum(x**2)

def squared_error(x,y):
	return (x - y)**2

examples = 1000
features = 100

Dataset = (np.random.randn(examples,features),np.random.randn(examples))
training_steps = 1000

x = T.dmatrix('x')
y = T.dvector('y')
w = theano.shared(np.random.randn(features),name = 'w')
b = theano.shared(0.0,name = 'b')

lambdaVal = 0.01
pred = T.dot(x,w) + b
error = (squared_error(pred,y))
loss = error.mean() + lambdaVal * l2(w)
gw, gb = T.grad(loss,[w,b])

train = theano.function(inputs = [x,y], outputs = [pred,error], updates = ((w,w-0.01*gw),(b,b-0.01*gb) ))
predict = theano.function(inputs = [x],outputs = pred)

print("Accuracy before training: ", sklearn.metrics.mean_squared_error( Dataset[1], predict( Dataset[0] ) ) ) 

#train
for i in xrange(training_steps):
	prediction, error = train(Dataset[0],Dataset[1])

print("Accuracy after training: ", sklearn.metrics.mean_squared_error(Dataset[1], predict( Dataset[0] ) ) )
