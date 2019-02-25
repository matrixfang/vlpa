import networkx as nx
import numpy as np
import scipy as sp
import cProfile
import re

import pickle
from collections import namedtuple
import inputdata
import matplotlib.pyplot as plt
import vlpa
from sklearn.metrics.cluster import normalized_mutual_info_score
import community
import plot
import matplotlib as mpl
import time
import pandas as pd
#pd.set_option('precision', 4)
def nmi(labels_real, labels):
    # normalized mutual information
    list_cal = []
    list_real = []
    for node in labels:
        list_cal.append(labels[node])
        list_real.append(labels_real[node])
    return normalized_mutual_info_score(list_cal, list_real)

def decide_de():
    g, real_label = inputdata.read_lfr(0.6)
    vecs = vlpa.de_vpa(g, k=15)
    with open('vecs.dat', 'wb') as f:
        pickle.dump(vecs, f)
    pass

def decide_de_plot():
    with open('vecs.dat', 'r') as f:
        vecs = pickle.load(f)

    norms1 = []
    norms1_mean =[]
    dims1 = range(1,16)
    for d in dims1:
        array = np.array([vecs[node].useful_norm(d) for node in vecs])
        norms1.append(array)
        norms1_mean.append(array.mean())

    dims2 = []
    norms2 = np.arange(0.5, 1.01, 0.05)
    for norm in norms2:
        dims2.append(np.array([vecs[node].useful_dims(p=norm) for node in vecs]))
    #fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(4,9))
    np.random.seed(20181010)
    #plt.plot(dims1, norms1_mean)
    plt.figure()
    plt.violinplot(norms1[0:10],showmeans=True,showmedians=False)
    plt.plot(dims1[0:10],norms1_mean[0:10])
    plt.scatter([6],[0.95],color='red')
    plt.annotate('cutting dimension = 6\n cutting norm=0.95',xy=(6,0.95),xycoords='data',
                 xytext=(+1,-80),textcoords='offset points', fontsize=10,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.xlabel('cutting dimension $d$')
    plt.ylabel('rest norm')
    plt.savefig('useful_dimensions.svg')
    plt.show()
    plt.close()
    pass

def NMI_compare_1():
    nmi_a = []
    nmi_b = []
    nmi_c = []
    nmi_d = []
    nmi_e = []
    nmi_f = []
    # x = [0.0, 0.1]
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for num in x:
        g, real_label = inputdata.read_lfr(num)

        a = vlpa.vpa(g,ifrecord=False).labels
        b = vlpa.vpas(g,ifrecord=False).labels
        c = vlpa.lpa(g,ifrecord=False).labels
        d = vlpa.clustering_infomap(g).labels
        e = vlpa.louvain(g).labels

        nmi_a.append(nmi(real_label, a))
        nmi_b.append(nmi(real_label, b))
        nmi_c.append(nmi(real_label, c))
        nmi_d.append(nmi(real_label, d))
        nmi_e.append(nmi(real_label, e))
        # nmi_f.append(nmi(real_label, f))
    # plot
    NMI_result = [x, nmi_a, nmi_b, nmi_c, nmi_d, nmi_e]
    with open('NMI_compare.dat', 'wb') as f:
        pickle.dump(NMI_result, f)

    with open('NMI_compare.dat', 'r') as f:
        result = pickle.load(f)
    plt.figure(1)
    plt.plot(result[0], result[1], label='VLPA')
    plt.plot(result[0], result[2], label='sVLPA')
    plt.plot(result[0], result[3], label='lpa')
    plt.plot(result[0], result[4], label='infomap')
    plt.plot(result[0], result[5], label='louvain')


    plt.legend(loc='lower left')
    plt.xlabel('$\mu$')
    plt.ylabel('NMI')
    plt.savefig('NMI_compare.eps')
    pass

def NMI_compare_2():

    with open('NMI_compare.dat', 'r') as f:
        result = pickle.load(f)
    plt.figure(1)
    plt.plot(result[0], result[1], label='VLPA')
    plt.plot(result[0], result[2], label='sVLPA')
    plt.plot(result[0], result[3], label='lpa')
    plt.plot(result[0], result[4], label='infomap')
    plt.plot(result[0], result[5], label='louvain')


    plt.legend(loc='lower left')
    plt.xlabel('$\mu$')
    plt.ylabel('NMI')
    plt.savefig('NMI_compare.eps')
    pass


def modularity_compare():
    x = [0.0,0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9]
    mod1 = []
    mod2 = []
    mod3 = []
    mod4 = []
    mod5 = []
    mod_reals = []
    for num in x:
        g, real_label = inputdata.read_lfr(num)
        result1 = vlpa.vpa(g,ifrecord=False)
        result2 = vlpa.vpas(g,ifrecord=False)
        result3 = vlpa.lpa(g)
        result4 = vlpa.clustering_infomap(g)
        result5 = vlpa.louvain(g)
        mod_real = community.modularity(real_label, g)

        mod1.append(result1.mod)
        mod2.append(result2.mod)
        mod3.append(result3.mod)
        mod4.append(result4.mod)
        mod5.append(result5.mod)
        mod_reals.append(mod_real)

    # plot
    result = [x, mod1, mod2, mod3, mod4, mod5, mod_reals]
    with open('modularity_compare.dat', 'wb') as f:
        pickle.dump(result, f)
    pass


def modularity_compare_plot():
    with open('modularity_compare.dat', 'r') as f:
        result = pickle.load(f)
        plt.figure(1)
        index = [4,5,6,7,8,9]
        plt.plot(result[0][5:10], np.array(result[6][5:10]), '--', label='benchmark')
        plt.plot(result[0][5:10], np.array(result[1][5:10]), label='VLPA')
        plt.plot(result[0][5:10], np.array(result[2][5:10]), label='sVLPA')
        plt.plot(result[0][5:10], np.array(result[3][5:10]), label='LPA')

        plt.plot(result[0][5:10], np.array(result[4][5:10]), label='infomap')
        plt.plot(result[0][5:10], np.array(result[5][5:10]), label='louvain')


        #plt.plot(result[0], np.log(1+np.array(result[4])), label='infomap')

        plt.legend()
        plt.xlabel('$\mu$')
        plt.ylabel('Modularity')
        plt.savefig('relative_modularity_compare.eps')
        plt.show()


def convergence_test():
    # x = [0.0, 0.1]
    g, real_label = inputdata.read_lfr(0.9)
    opt_value = community.modularity(real_label, g)
    result1 = vlpa.vpa(g)
    result2 = vlpa.vpas(g)
    results = [result1, result2,]

    with open('convergence_test.dat', 'wb') as f:
        pickle.dump(results, f)


def convergence_test_plot():
    with open('convergence_test.dat', 'r') as f:
        results = pickle.load(f)

    input = {}
    for result in results:
        input[result.algorithm] = result.mods

    opt_value = result.mod
    convergence_gamma_plot(input, opt_value)

def convergence_gamma_plot(input, opt_value):
    plt.figure()
    eps = 0.0001
    k_min = 0
    key_min = None
    k_max = -float('Inf')
    key_max = None

    for key in input:
        mods = input[key]
        print(mods)
        logmods = []
        x = np.log10(range(1, len(mods) + 1))
        for v in mods:
            logmods.append(np.log10(abs(v - opt_value) + eps))
        if key in [key_max, key_min]:
            pass
        else:
            plt.plot(x,logmods)

    plt.legend(loc='lower left')
    plt.xlabel('$\log_{10} (step)$')
    plt.ylabel('$\log_{10}(\|Q-Q_{best}\|)$')
    plt.savefig('convergence_plot.eps')
    plt.close()
    pass


def form_modularity_compare():
    data = {'vpa':[],'vpas':[],'louvain':[],'lpa':[],'infomap':[],'benckmark':[]}
    for x in [0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9]:
        g, real_label = inputdata.read_lfr(x)
        result1 = vlpa.vpa(g,ifrecord=False)
        result2 = vlpa.vpas(g,ifrecord=False)
        result3 = vlpa.louvain(g,ifrecord=False)
        result4 = vlpa.lpa(g,ifrecord=False)
        result5 = vlpa.clustering_infomap(g, ifrecord=False)
        data['vpa'].append(result1.mod)
        data['vpas'].append(result2.mod)
        data['louvain'].append(result3.mod)
        data['lpa'].append(result4.mod)
        data['infomap'].append(result5.mod)
        opt_value = community.modularity(real_label, g)
        data['benckmark'].append(opt_value)
    with open('form_modularity_compare10.dat', 'wb') as f:
        pickle.dump(data, f)


def form_modularity_compare_plot():
    dict ={'vpa':{},'vpas':{},'louvain':{},'lpa':{},'informap':{},'benchmark':{}}
    df_all = pd.DataFrame()
    for num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        file_name = 'form_modularity_compare' + str(num) + '.dat'
        #print(file_name + ' loaded!')
        with open(file_name, 'r') as f:
            data = pickle.load(f)
        df_cal = pd.DataFrame(data)
        df_cal['mu'] = [0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9]
        df_all = pd.concat([df_all,df_cal])
    print(df_all)
    df_final = (df_all.groupby(df_all['mu']).mean())
    col = list(df_final.columns)
    print(col)
    col.insert(1, col.pop(3))
    print(col)
    print(df_final.loc[:,col].to_latex())

modularity_compare_plot()