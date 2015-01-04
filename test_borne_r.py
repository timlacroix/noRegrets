from igraph import *
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt


sizes = [25, 50, 75]
ms = [1, 2, 5, 10]

repeats = 150

for n_nodes in sizes:
    pp = PdfPages('independence_number{}.pdf'.format(n_nodes))
    print(n_nodes)
    for m in ms:
        tirages = list()
        for i in range(repeats):
            print(i)
            g = Graph.Barabasi(n_nodes, m)
            r = 2*g.ecount()/(float(n_nodes)*(n_nodes-1))
            tirages.append(g.independence_number())
        plt.figure()
        plt.hist(tirages, normed=1)
        plt.title('{} nodes and m = {}, r = {}, 1/r = {}'.format(n_nodes, m, np.round(r, decimals=1), np.round(1/r, decimals=2)), fontsize=25)
        pp.savefig()
        plt.close()

    pp.close()
