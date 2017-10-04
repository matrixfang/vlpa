import networkx as nx
import numpy as np
import scipy as sp
import inputdata
import draw
import matplotlib.pyplot as plt
import vlpa
from sklearn.metrics.cluster import normalized_mutual_info_score


def nmi(labels, labels_real):
    # normalized mutual information
    list_cal = []
    list_real = []
    for node in labels:
        list_cal.append(labels[node])
        list_real.append(labels_real[node])
    return normalized_mutual_info_score(list_cal, list_real)

G, community_label = inputdata.read_lfr(128)


# n = float(len(G.nodes()))
# k_ave = float(sum(G.degree().values())) / n
#
# pro = vlpa.Propragation(G)
# label = pro.run()
#
# draw.draw_group(G, community_label, label)
# plt.show()
# plt.close()
# print(nmi(community_label,label))


a = vlpa.vlabel()
b = vlpa.vlabel({1:2,3:4,4:4})
c = vlpa.vlabel({i:i for i in xrange(10)})

print(a.normalize())



