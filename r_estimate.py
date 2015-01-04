from graphs import *
from joblib import Parallel, delayed
from learners import *
from numpy import random
import numpy as np

def get_M_BA(n_arms, m, m0):
    g = BAGraph(arms=n_arms, m=m, m0=m0, r=1)
    g. makeGraph()
    vector = np.append(g.getObserved(0),1)
    return np.argmax(vector[1:])+1

def get_R_ER(n_arms, r, n_estimate):
    return 1./np.mean(
        np.minimum(random.geometric(r, n_estimate), n_arms)
    )

if __name__=="__main__":

    n_estimate = 20000
    n_arms = 500
    m = 1
    m0 = 2

    M_BA = Parallel(n_jobs=8, verbose=5)(delayed(get_M_BA)(
        n_arms=n_arms, m=m, m0=m0) for i in range(n_estimate)
    )

    g = BAGraph(arms=n_arms, m=m, m0=m0, r=1)
    g.makeGraph()

    real_r = sum(sum(g.adjacency))/(float(n_arms)*(n_arms-1))

    r_er = get_R_ER(n_arms=n_arms, r=real_r, n_estimate=n_estimate)

    print 'R_ER : {}'.format(r_er)
    print 'R_BA : {}'.format(1./np.mean(M_BA))

    print 'Real_R : {}'.format(real_r)
    l=BAEXP3(0,0,arms=n_arms)

    print 'Graph K : {}'.format(l.K)

    t = n_arms-m0
    formula_r = 2.*float(m*t)/((m0+t)*(m0+t-1))

    normalization = sum([k**(-3) for k in range(1, n_arms)])
    probas = [k**(-3)/normalization for k in range(1, n_arms)]
    formula_r_BA = n_arms*sum([x/(i+2) for (i,x) in enumerate(probas)])
    formula_r_BA = 1./formula_r_BA

    print 'Formula R : {}'.format(formula_r)
    print 'Formula R_BA : {}'.format(formula_r_BA)