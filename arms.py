from numpy.random import rand

class Bernoulli:
    def __init__(self, mean):
        self.mean = mean

    def getLoss(self):
        return int(rand(1)[0]<=self.mean)
