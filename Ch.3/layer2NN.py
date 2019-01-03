#Ch. 3
import autograd.numpy as np 
import autograd.numpy.random as npr
#autograd effeciently computes derivatives
from autograd import grad
import sklearn.metrics
import pylab

#generating dataset
examples = 1000
features = 100
Dataset = (npr.randn(examples,features), npr.randn(examples))

#Network details
layer1_units = 10
layer2_units = 1

w1 = npr.rand(features,layer1_units)
b1 = npr.rand(1,layer1_units)

w2 = npr.rand(layer1_units,layer2_units)
b2 = 0.0#npr.rand(1,layer2_units)

theta = (w1,b1,w2,b2)

#loss function 
def squared_loss(y,y_hat):
	a = y-y_hat
	return np.dot( a,a )

#Output layer
def binary_cross_entropy(y,y_hat):
	a = y * np.log(y_hat)
	b = (1 - y) * np.log(1 - y_hat)
	return np.sum( -(a + b) )

#the Neural Network
def neural_network(x,theta):
	#all the parameters
	w1,b1,w2,b2 = theta
	#layer 1
	xW1 = np.dot(x,w1)
	#activation
	theTanH = np.tanh(xW1+ b1)
	#layer 2
	secondxW2 = np.dot(theTanH, w2)
	return np.tanh(secondxW2 + b2)

#wrapper function around the objective function to be optimized
def objective(theta, idx):
	return squared_loss(Dataset[1][idx], neural_network(Dataset[0][idx], theta))

#update
def update_theta(theta, delta, alpha):
	w1,b1,w2,b2 = theta
	w1_delta,b1_delta,w2_delta,b2_delta = delta
	w1_new = w1 - alpha * w1_delta
	b1_new = b1 - alpha * b1_delta
	w2_new = w2 - alpha * w2_delta
	b2_new = b2 - alpha * b2_delta
	new_theta = (w1_new,b1_new,w2_new,b2_new)
	return new_theta

#compute gradient
grad_objective = grad(objective)

#Training the neural network
#how many times to go over the training data
epochs = 10
print ("RMSE before training", sklearn.metrics.mean_squared_error(Dataset[1],neural_network(Dataset[0],theta)))
rmse =[]
for i in xrange(0,epochs):
	for j in xrange(0, examples): 
		delta = grad_objective(theta,j)
		theta = update_theta(theta, delta,0.01)
	rmse.append(sklearn.metrics.mean_squared_error(Dataset[1],neural_network(Dataset[0],theta) ))

print("RMSE after training", sklearn.metrics.mean_squared_error(Dataset[1],neural_network(Dataset[0],theta) ) )

#print(rmse)
pylab.plot(rmse,'g')
pylab.show()















