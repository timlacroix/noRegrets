from graphs import *
import pylab


n_arms = 500
m = 1
m0 = 2

degrees = np.zeros(n_arms)

n_estimate = 500

g = BAGraph(arms=n_arms, m=m, m0=m0, r=1)

for _ in range(n_estimate):
    degree = sum(g.getObserved(0))-1
    degrees[degree] += 1

pylab.plot(degrees)
pylab.show()