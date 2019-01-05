#theano has many loss functions in nnet package

import theano.tensor as T
from theano import function

#Binary cross entropy
a1 = T.dmatrix('a1')
a2 = T.dmatrix('a2')
# crossentropy(t,o) = - ( t * log(o) + (1 - t) * log(1 - o) )
f_a = T.nnet.binary_crossentropy(a1,a2).mean()
f_sigmoid = function([a1,a2],[f_a])
print("BInary Cross Entropy: [[0.01,0.01]],[[0.99,0.01]]: ",
	f_sigmoid([[0.01,0.01]],[[0.99,0.01]]) )

#what is binary cross entropy loss?
'''Binray cross entropy of a given strategy is the expected number of questions to gusss the class under that 
strategy [Decision Tree]
sum(pi*log (1/pi) ) where pi is probability of the class. log(1/pi) represents how many questions does it take
to reach to the level of that class
https://www.quora.com/Whats-an-intuitive-way-to-think-of-cross-entropy
'''
#Categorical cross entropy
b1 = T.dmatrix('b1')
b2 = T.dmatrix('b2')
# categorical_crossentropy(p,q) = - sumx ( p(x) log( q(x) ))
f_b = T.nnet.categorical_crossentropy(b1,b2).mean()
f_sigmoid = function([b1,b2],[f_b])
print("Categorical Cross Entropy: [[0.01,0.01]],[[0.99,0.01]]: ",
	f_sigmoid([[0.01,0.01]],[[0.99,0.01]]) )

'''
http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#theano.tensor.nnet.nnet.categorical_crossentropy
'''

#squared error
def squared_error(x,y):
	return(x - y)**2

c1 = T.dmatrix('c1')
c2 = T.dmatrix('c2')
# we can implement squared error by ourself
f_c = squared_error(c1,c2).mean()
f_squared_error = function([c1,c2],[f_c])
print("Simple Square Error: [[0.01,0.01]],[[0.99,0.01]]: ",
	f_squared_error([[0.01,0.01]],[[0.99,0.01]]) )