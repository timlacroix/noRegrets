import numpy as np
from numpy.random import rand, choice
from random import shuffle


class ERGraph:
    def __init__(self, arms, r):
        self.arms = arms
        self.r = r

    def getObserved(self, pulled):
        observed = np.zeros(self.arms)
        observed[pulled] = 1
        return np.minimum(observed + (rand(self.arms) < self.r), 1)


class BAGraph:
    def __init__(self, arms, m, m0, r=0.3):
        """
        arms : total number of nodes
        m : number of connections each new node creates
        m0 : initial graph size
        r : initial graph sparsity
        """
        if m > m0:
            raise Exception("m > m0")
        if m0 > arms:
            raise Exception("m0 > arms")
        self.arms = arms
        self.m = m
        self.m0 = m0
        self.r=r


    def addEdge(self, i, j):
        self.adjacency[i,j] = 1
        self.adjacency[j,i] = 1
        self.nb_neighbors[i] += 1
        self.nb_neighbors[j] += 1

    def makeGraph(self):
        self.nodes = range(self.arms)
        shuffle(self.nodes)

        self.adjacency = np.zeros((self.arms, self.arms))
        self.nb_neighbors = np.zeros(self.arms)
        
        # create a random path first to connect the first m_0 nodes
        for i in range(self.m0-1):
            self.addEdge(i, i+1)
        # add each edges with proba r
        for i in range(self.m0):
            for j in range(i+2,self.m0):
                if rand() <= self.r:
                    self.addEdge(i,j)
        # add the others arms-m0 nodes
        for cur in range(self.m0, self.arms):
            chosen = choice(
                cur,
                size=self.m,
                p=(self.nb_neighbors /np.sum(self.nb_neighbors))[:cur]
            )
            for i in chosen:
                self.addEdge(cur, i)

        # SHUFFLE
        self.adjacency = self.adjacency[self.nodes].T[self.nodes]

    def getObserved(self, pulled):
        self.makeGraph()
        observed = self.adjacency[pulled,:]
        observed[pulled] = 1
        return observed
