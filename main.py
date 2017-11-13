import networkx as nx
import numpy as np
import scipy as sp
import draw
import pickle
import inputdata
import matplotlib.pyplot as plt
import vlpa
from sklearn.metrics.cluster import normalized_mutual_info_score
import community
import matplotlib as mpl
mpl.use("Agg")


def nmi(labels_real, labels):
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
    nmi_e = []
    nmi_f = []
    # x = [0.0, 0.1]
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for num in x:
        g, real_label = inputdata.read_lfr(num)

        a = vlpa.vlpa(g)
        b = vlpa.lpa(g)
        c = vlpa.clusting_infomap(g)
        d = community.best_partition(g)
        # e = vlpa.vlpa2(g)
        # f = vlpa.vlpa3(g)
        nmi_a.append(nmi(real_label, a))
        nmi_b.append(nmi(real_label, b))
        nmi_c.append(nmi(real_label, c))
        nmi_d.append(nmi(real_label, d))
        # nmi_e.append(nmi(real_label, e))
        # nmi_f.append(nmi(real_label, f))
    # plot
    plt.figure(1)
    plt.plot(x, nmi_a, label='vlpa')
    plt.plot(x, nmi_b, label='lpa')
    plt.plot(x, nmi_c, label='infomap')
    plt.plot(x, nmi_d, label='louvain')
    # plt.plot(x, nmi_e, label='vlpa2')
    # plt.plot(x, nmi_f, label='vlpa3')
    plt.legend(loc='upper left')
    plt.savefig('compare.png')


def shrink_compare():
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m1 = vlpa.method(withshrink=True)
    m2 = vlpa.method(withshrink=False)
    nmi_a = []
    nmi_b = []
    for num in x:
        g, real_label = inputdata.read_lfr(num)
        nmi_a.append(nmi(real_label, m1(g)))
        nmi_b.append(nmi(real_label, m2(g)))

    plt.figure(1)
    plt.plot(x, nmi_a, label='withshrink')
    plt.plot(x, nmi_b, label='withoutshrink')
    plt.legend(loc='upper left')
    plt.savefig('compare.png')


def gamma_compare():
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    methods = dict()
    nmi_dic = dict()
    gamma_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    for gamma in gamma_list:
        methods[gamma] = vlpa.method(withshrink=False, gamma=gamma)
        nmi_dic[gamma] = []

    for num in x:
        g, real_label = inputdata.read_lfr(num)
        for gamma in gamma_list:
            nmi_dic[gamma].append(nmi(real_label, methods[gamma](g)))

    plt.figure(1)
    for gamma in gamma_list:
        plt.plot(x, nmi_dic[gamma], label='gamma=' + str(gamma))
    plt.legend(loc='upper left')
    plt.savefig('gamma_compare.png')


def pos_compare():
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m1 = vlpa.method(posshrink=True)
    m2 = vlpa.method(posshrink=False)
    nmi_a = []
    nmi_b = []
    for num in x:
        g, real_label = inputdata.read_lfr(num)
        nmi_a.append(nmi(real_label, m1(g)))
        nmi_b.append(nmi(real_label, m2(g)))

    with open('pos_compare.dat', 'wb') as f:
        pickle.dump(x, f)
        pickle.dump(nmi_a, f)
        pickle.dump(nmi_b, f)


def convergence_test():
    g, real_label = inputdata.read_lfr(0.6)
    mod = vlpa.convergence_vlpa(g, gamma=0.6, mod='both')
    opt_value = community.modularity(real_label, g)
    log_values = [np.log(abs(v - opt_value)) for v in mod]
    log_k = [np.log(k + 1) for k in xrange(len(log_values))]
    with open('convergence_test.dat', 'wb') as f:
        pickle.dump(log_k, f)
        pickle.dump(log_values, f)


def test():
    g, real_label = inputdata.read_lfr(0.6)
    label = vlpa.clustering_infomap(g)
    label2 = vlpa.basic_vlpa(g)
    print('nmi of vlpa is', nmi(real_label, label2))
    print('modularity of vlpa is', community.modularity(label2, g))
    print('nmi of infomap is', nmi(real_label, label))
    print('modularity of infomap is ', community.modularity(label, g))


def method_adjust():
    # x = [0.0, 0.1]
    mod_a = []
    mod_b = []
    mod_c = []
    g, real_label = inputdata.read_lfr(0.6)
    opt_value = community.modularity(real_label, g)
    mod_a = vlpa.convergence_vlpa(g, gamma=0.5, mod='both')
    mod_b = vlpa.convergence_vlpa(g, gamma=0.5, mod='nothing')
    mod_c = vlpa.convergence_vlpa(g, gamma=0.5, mod='normalize')
    mod_d = vlpa.convergence_vlpa(g, gamma=0.9, mod='normalize')
    log_a_values = [np.log(abs(v - opt_value)) for v in mod_a]
    log_b_values = [np.log(abs(v - opt_value)) for v in mod_b]
    log_c_values = [np.log(abs(v - opt_value)) for v in mod_c]
    log_d_values = [np.log(abs(v - opt_value)) for v in mod_d]

    with open('method_adjust.dat', 'wb') as f:
        pickle.dump(log_a_values, f)
        pickle.dump(log_b_values, f)
        pickle.dump(log_c_values, f)
        pickle.dump(log_d_values, f)


method_adjust()

test()
