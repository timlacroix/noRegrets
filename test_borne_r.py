from igraph import *
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# sizes = [25, 50, 75]
# ms = [1, 2, 5, 10]

# repeats = 150

# for n_nodes in sizes:
#     pp = PdfPages('independence_number{}.pdf'.format(n_nodes))
#     print(n_nodes)
#     for m in ms:
#         tirages = list()
#         for i in range(repeats):
#             print(i)
#             g = Graph.Barabasi(n_nodes, m)
#             r = 2*g.ecount()/(float(n_nodes)*(n_nodes-1))
#             tirages.append(g.independence_number())
#         plt.figure()
#         plt.hist(tirages, normed=1)
#         plt.title('{} nodes and m = {}, r = {}, 1/r = {}'.format(n_nodes, m, np.round(r, decimals=1), np.round(1/r, decimals=2)), fontsize=25)
#         pp.savefig()
#         plt.close()

#     pp.close()

# sizes = [25, 50, 75]    
# rs = [0.05, 0.2, 0.5, 0.7]

# repeats = 300

# for n_nodes in sizes:
#     pp = PdfPages('independence_number_ER_{}.pdf'.format(n_nodes))
#     print(n_nodes)
#     for r in rs:
#         tirages = list()
#         for i in range(repeats):
#             print(i)
#             g = Graph.Erdos_Renyi(n_nodes, r)
#             tirages.append(g.independence_number())
#         plt.figure()
#         plt.hist(tirages, normed=1)
#         plt.title('{} nodes and r = {}, r = {}, 1/r = {}'.format(n_nodes, r, np.round(r, decimals=1), np.round(1/r, decimals=2)), fontsize=25)
#         pp.savefig()
#         plt.close()

#     pp.close()

# n_nodes = 10
# rs = [0.1, 0.2, 0.5, 0.8]
# for r in rs:
#     er=nx.erdos_renyi_graph(n_nodes,r)
#     nx.draw(er)
#     plt.savefig('ER_graph_{}.pdf'.format(10*r))
#     plt.close() 


# n_nodes = 10
# ms = [1, 2, 5]
# for m in ms:
#     ba=nx.barabasi_albert_graph(n_nodes, m)
#     nx.draw(ba)
#     plt.savefig('BA_graph_{}.pdf'.format(m))
#     plt.close() 

rep = 40
n_nodes = 50
r_list = list()
big_ba = list()
ba_std = list()
big_er = list()
er_std = list()
for m in range(1, 15):
    ba_alpha = list()
    er_alpha = list()
    for i in range(rep):
        g = Graph.Barabasi(n_nodes, m)
        r = 2*g.ecount()/(float(n_nodes)*(n_nodes-1))
        g_er = Graph.Erdos_Renyi(n_nodes, r)
        ba_alpha.append(g.independence_number())
        er_alpha.append(g_er.independence_number())
    r_list.append(r)
    big_ba.append(np.mean(ba_alpha))
    big_er.append(np.mean(er_alpha))
    ba_std.append(np.std(ba_alpha))
    er_std.append(np.std(er_alpha))

plt.errorbar(r_list, big_ba, yerr=ba_std, fmt='o', color='m', label='BA graph', linewidth=2)
plt.errorbar(r_list, big_er, yerr=er_std, fmt='o', color='b', label='ER graph', linewidth=2)
plt.legend(fontsize=20)
plt.xlabel('p', fontsize=20)
plt.ylabel('independence number', fontsize=20)
plt.savefig('independance_number_com.pdf')
plt.close()

"""
Random graph from given degree sequence.
Draw degree rank plot and graph with matplotlib.
"""
__author__ = """Aric Hagberg <aric.hagberg@gmail.com>"""
import networkx as nx
import matplotlib.pyplot as plt
G = nx.barabasi_albert_graph(200, 20)

degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
#print "Degree sequence", degree_sequence
dmax=max(degree_sequence)

plt.hist(np.log(degree_sequence), normed=1, log=True)
#plt.show()

plt.savefig("degree_histogram.png")
