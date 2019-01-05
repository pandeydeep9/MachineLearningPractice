#now with vectors
import numpy as np
import theano.tensor as T
from theano import function

#matxis defined now before they can be used, every matrix given a unique name
a = T.dmatrix('a')
b = T.dmatrix('b')
c = T.dmatrix('c')
d = T.dmatrix('d')
#e = T.dmatrix('e')

e = (a + b - c) * d
f = function([a,b,c,d],e)#((a - b + c) * d) / e

a_data = np.array([[1,2],[4,3]]) #Just a simple matrix
b_data = np.array([[5,6],[7,8]])
c_data = np.array([[9,10],[11,12]])
d_data = np.array([[13,14],[15,16]])
#print(a_data[1][0])

normally = (a_data + b_data - c_data) * d_data
fromTheano = f(a_data,b_data,c_data,d_data)

print("Expected: ", normally)
print ("Theano:" , fromTheano)
#print("Expected: ((1 - 2 + 3)* 4) / 5.0 =", ((1 - 2 + 3)* 4) / 5.0)
#print("Theano: ((1 - 2 + 3)* 4) / 5.0 =", g(1,2,3,4,5) )

