#simple script to generate a dataset
import pylab
import numpy

# 100 data points between -1 and 1
x= numpy.linspace(-1,1,100)

signal = 2 + x - 2 * x**3
noise = numpy.random.normal(0,0.2,100)

y = signal + noise

pylab.plot(signal,'b')
pylab.plot(y,'g')
pylab.plot(noise,'r')

pylab.xlabel('x')
pylab.ylabel('y')
pylab.legend(["Without noise", "With Noise", "Noise"], loc = 1)

pylab.show()