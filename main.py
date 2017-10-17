import networkx as nx
import numpy as np
import scipy as sp
import draw
import inputdata
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import vlpa
from sklearn.metrics.cluster import normalized_mutual_info_score
import community




def nmi(labels, labels_real):
    # normalized mutual information
    list_cal = []
    list_real = []
    for node in labels:
        list_cal.append(labels[node])
        list_real.append(labels_real[node])
    return normalized_mutual_info_score(list_cal, list_real)


def compare():
    nmi_a = []
    nmi_b = []
    nmi_c = []
    nmi_d = []
    #x = [0.0, 0.1]
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for num in x:
        g, real_label = inputdata.read_lfr(num)
        a = vlpa.vlpa(g)
        b = vlpa.lpa(g)
        c = vlpa.clusting_infomap(g)
        d = community.best_partition(g)
        nmi_a.append(nmi(real_label, a))
        nmi_b.append(nmi(real_label, b))
        nmi_c.append(nmi(real_label, c))
        nmi_d.append(nmi(real_label, d))
    # plot
    plt.figure(1)
    plt.plot(x, nmi_a, label='vlpa')
    plt.plot(x, nmi_b, label='lpa')
    plt.plot(x, nmi_c, label='infomap')
    plt.plot(x, nmi_d, label='louvain')
    plt.legend(loc='upper right')
    plt.savefig('compare.png')


# G, community_label = inputdata.read_lfr(0.7)
#
# print(community.best_partition(G))

compare()
