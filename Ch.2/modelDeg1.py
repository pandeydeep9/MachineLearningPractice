import pylab
import numpy
import numpy as np

#100 random variables between -1 and 1
x= np.linspace(-1,1,100)

signal = 2 + x - 2 * x**3

#100 random and normally distributed variables with 0 mean and sd 0.1
noise = np.random.normal(0,0.1,100)

y = signal + noise

#Train with the first 80 data
x_train = x[0:80]
y_train = y[0:80]



#model with degree 1
degree =2

#print(x_train.shape)
#x train is an single column of length 80
x_train = np.column_stack([np.power(x_train,i) for i in xrange(0,degree)])
# xtrain is now an two column matrix with data 1 in column 1 and data x_train in column 2
#print(x_train)

# The model is ( ( (X.T * X) inv ) * X.T ) * Y ) 
model = np.dot(np.dot(np.linalg.inv(np.dot(x_train.transpose(),x_train)), x_train.transpose()), y_train)


#this is our input data plot in green
pylab.plot(x,y,'g')
pylab.xlabel('x')
pylab.ylabel('y')

#this is our prediction from the 1 degree model
predicted = np.dot(model,[np.power(x,i) for i in xrange(0,degree)])
pylab.plot(x, predicted,'r')
pylab.legend(["Actual Data","Prediction with First Degree"], loc = 1)

pylab.show() #The prediction of first degree equation

#Now we see how accurate the model is
train_rmse1 = np.sqrt(np.sum(np.dot(y[0:80] - predicted[0:80], y_train - predicted[0:80])))
test_rmse1 = np.sqrt(np.sum(np.dot(y[80:] - predicted[80:], y[80:] - predicted[80:])))
print("Train rmse(degree = 1) ", train_rmse1)
print("test rmse(degree = 1) ", test_rmse1)


#model with degree 2
degree =3

x_train = x[0:80]
y_train = y[0:80]
x_train = np.column_stack([np.power(x_train,i) for i in xrange(0,degree)])

model = np.dot(np.dot(np.linalg.inv(np.dot(x_train.transpose(),x_train)), x_train.transpose()), y_train)

pylab.plot(x,y,'g')
pylab.xlabel('x')
pylab.ylabel('y')
predicted = np.dot(model,[np.power(x,i) for i in xrange(0,degree)])
pylab.plot(x, predicted,'r')
pylab.legend(["Actual Data","Prediction of Second Degree"], loc = 1)
pylab.show()

train_rmse1 = np.sqrt(np.sum(np.dot(y[0:80] - predicted[0:80], y_train - predicted[0:80])))
test_rmse1 = np.sqrt(np.sum(np.dot(y[80:] - predicted[80:], y[80:] - predicted[80:])))
print("Train rmse(degree = 2) ", train_rmse1)
print("test rmse(degree = 2) ", test_rmse1)

#model with degree 3
degree =[4,9]

x_train = x[0:80]
y_train = y[0:80]
x_train_deg3 = np.column_stack([np.power(x_train,i) for i in xrange(0,degree[0] )])
x_train_deg8 = np.column_stack([np.power(x_train,i) for i in xrange(0,degree[1] )])

x_train = x_train_deg3
model_deg3 = np.dot(np.dot(np.linalg.inv(np.dot(x_train.transpose(),x_train)), x_train.transpose()), y_train)
x_train = x_train_deg8
model_deg8 = np.dot(np.dot(np.linalg.inv(np.dot(x_train.transpose(),x_train)), x_train.transpose()), y_train)

pylab.plot(x,y,'g')
pylab.xlabel('x')
pylab.ylabel('y')

predicted_deg3 = np.dot(model_deg3,[np.power(x,i) for i in xrange(0,degree[0])])
predicted_deg8 = np.dot(model_deg8,[np.power(x,i) for i in xrange(0,degree[1])])

pylab.plot(x, predicted_deg3,'r')
pylab.plot(x, predicted_deg8,'y')

pylab.legend(["Actual Data","Prediction of Third Degree","Prediction of Degree 8(overfit)"], loc = 1)
pylab.show()

train_rmse1 = np.sqrt(np.sum(np.dot(y[0:80] - predicted[0:80], y_train - predicted[0:80])))
test_rmse1 = np.sqrt(np.sum(np.dot(y[80:] - predicted[80:], y[80:] - predicted[80:])))
print("Train rmse(degree = 2) ", train_rmse1)
print("test rmse(degree = 2) ", test_rmse1)