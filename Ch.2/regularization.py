#trying to penalize the higher degree parameters such that the model fits the best way possible to the data
import pylab
import numpy
import numpy as np

x= np.linspace(-1,1,100)

signal =2 + x - 2 * x**3
noise = np.random.normal(0,0.2,100)

y = signal + noise

x_train = x[0:80]
y_train = y[0:80]

train_rmse =[]
test_rmse = []

degree = 80


x_train = np.column_stack([np.power(x_train,i) for i in xrange(0,degree)])

lambda_reg_values = np.linspace(0.01,0.99,100)
#print(lambda_reg_values)
for lambda_reg in lambda_reg_values:
	# a regularized model
	 model = np.dot(np.dot(np.linalg.inv(np.dot(x_train.transpose(),x_train) + lambda_reg * np.identity(degree)), x_train.transpose()), y_train)
	 predicted = np.dot(model,[np.power(x,i) for i in xrange(0,degree)])
	 train_rmse.append(np.sqrt(np.sum(np.dot(y[0:80] - predicted[0:80], y_train - predicted[0:80]))))
	 test_rmse.append(np.sqrt(np.sum(np.dot(y[80:] - predicted[80:], y[80:] - predicted[80:]))))


pylab.plot(lambda_reg_values,train_rmse)
pylab.plot(lambda_reg_values,test_rmse)
pylab.xlabel(r'$\lambda$')
pylab.ylabel("RMSE")
pylab.legend(["Train","Test"],loc = 2)
pylab.show()

#Looking at the curve best value of lambda is 0.08
lambda_reg = 0.08
model = np.dot(np.dot(np.linalg.inv(np.dot(x_train.transpose(),x_train) + lambda_reg * np.identity(degree)), x_train.transpose()), y_train)
predicted = np.dot(model,[np.power(x,i) for i in xrange(0,degree)])

pylab.plot(x,y,'g')
pylab.xlabel('x')
pylab.ylabel('y')

pylab.plot(x, predicted,'r')
pylab.legend(["Actual Data","Prediction with lambda_reg = 0.08"], loc = 1)
pylab.show()