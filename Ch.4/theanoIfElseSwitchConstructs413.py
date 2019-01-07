'''
certain functions we may need if/else and switch constructs.
Such is readily available in theano
'''
import numpy as np 
import theano
import theano.tensor as T 
from theano.ifelse import ifelse

#This is a simple function which returns the value of x+y if x+y>0, else it returns 0
def hinge_a(x,y):
	return T.max([0, x+y])

#T.lt(x+y,0*x) is the condition(similar to if() does first. else does second)
#The condition is is x+y less than x??
def hinge_b(x,y):
	#print(T.lt(x,y), x, " a ", y)
	return ifelse(T.lt(x+y,x), x, x+y)

#exactly same logic aas iflelse. I don't see any difference here. 
def hinge_c(x,y):
	return T.switch(T.lt(x,y),x,x*y)

hinge_b(1,1)
hinge_b(-1,1)
x = T.dscalar('x')
y = T.dscalar('y')
z1 = hinge_a(x, y)
z2 = hinge_b(x, y)
z3 = hinge_c(x, y)
f1 = theano.function([x,y], z1)
f2 = theano.function([x,y], z2)
f3 = theano.function([x,y], z3)
print "f(-2, 1) =",f1(-2, 1), f2(-2, 1), f3(-2, 1)
print "f(-1,1 ) =",f1(-1, 1), f2(-1, 1), f3(-1, 1)
print "f(0,1) =",f1(0, 1), f2(0, 1), f3(0, 1)
print "f(1, 1) =",f1(1, 1), f2(1, 1), f3(1, 1)
print "f(2, 1) =",f1(2, 1), f2(2, 1), f3(2, 1)