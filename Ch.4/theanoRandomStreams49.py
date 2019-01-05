import theano.tensor as T
from theano import function
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import pylab

random = RandomStreams(seed = 34)

a = random.normal((1,3))
b = T.dmatrix('b')

f1 = a * b

g1 = function([b], f1)

print("Invoke once: ", g1( [[1,2,2]] ) )  
print("Invoke two:  ", g1( np.ones((1,3)) ) )
print("Invoke three:", g1( np.ones((1,3)) ) )

#Experimental Section :)
c = random.normal()
d = T.dmatrix('d')

f2 = c * d
g2 = function([d],f2)

# the value 3 in g2 signifies that the most data will lie between -3 and 3 maybe the sd(IDK)
print("so: ", g2([[3]]))
apple= []
for i in xrange(0,1000):
	apple.append(g2([[30]])[0] ) 


pylab.plot(apple,'b')

#apple is collection of 100 datas
apple.sort()

pylab.plot(apple,'g')


pylab.plot(apple,'g')
pylab.xlabel('x')
pylab.ylabel('y')
pylab.legend(["The distribution"], loc = 1)

pylab.show()