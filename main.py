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
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for num in x:
        g, real_label = inputdata.read_lfr(num)
        a = vlpa.vlpa(g)
        b = vlpa.lpa(g)
        c = vlpa.clusting_infomap(g)
        nmi_a.append(nmi(real_label, a))
        nmi_b.append(nmi(real_label, b))
        nmi_c.append(nmi(real_label, c))
    # plot
    plt.figure(1)
    plt.plot(x, nmi_c, color='b', label='infomap')
    plt.plot(x, nmi_a, color='r', label='vlpa')
    plt.plot(x, nmi_b, color='g', label='lpa')
    plt.legend(loc='upper right')
    plt.savefig('compare.png')


# G, community_label = inputdata.read_lfr(0.7)
# a = vlpa.clusting_infomap(G)
# a = vlpa.vlpa(G)
# print(nmi(community_label,a))
# draw.draw_group(G, community_label, a)
# plt.show()
# plt.close()



compare()


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
#

# a = vlpa.vlabel()
# b = vlpa.vlabel({1:2,3:4,4:4})
# c = vlpa.vlabel({i:i for i in xrange(10)})
#
# print(a.normalize())
