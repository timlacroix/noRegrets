import numpy as np
from numpy.random import rand, multinomial
from random import sample


class ERGraph:
    def __init__(self, arms, r):
        self.arms = arms
        self.r = r

    def getObserved(self, pulled):
        observed = np.zeros(self.arms)
        observed[pulled] = 1
        return np.minimum(observed + (rand(self.arms) < self.r), 1)


class BAGraph:
    def __init__(self, arms, m):
        self.arms = arms
        self.m = m

    def makeGraph(self):
        nodes = range(self.arms)
        adjacency = np.zeros((1, 2))
        nb_neighbors = np.zeros(self.arms)
        initiation = sample(nodes, self.m+1)
        for i in initiation:
            nodes.pop(nodes.index(i))
        for i in range(self.m):
            adjacency = np.vstack((adjacency, (initiation[i], initiation[self.m]),
                                  (initiation[self.m], initiation[i])))
        nb_neighbors[initiation] = 1
        nb_neighbors[initiation[self.m]] = self.m
        while len(np.nonzero(nb_neighbors)[0]) < self.arms:
            new_node = sample(nodes, 1)[0]
            nodes.pop(nodes.index(new_node))
            chosen = np.where(multinomial(self.m, nb_neighbors /
                                          np.sum(nb_neighbors)))[0][0:self.m]
            nb_neighbors[chosen] += 1
            nb_neighbors[new_node] = self.m
            for i in chosen:
                adjacency = np.vstack((adjacency, (i, new_node), (new_node, i)))
        self.adjacency = adjacency

    def getObserved(self, pulled):
        observed = np.zeros(self.arms)
        observed[pulled] = 1
        observed[list(self.adjacency[self.adjacency[:, 0] == pulled][:, 1])] = 1
        return observed
