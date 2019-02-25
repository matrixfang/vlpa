import networkx as nx
import numpy as np
import scipy as sp
import draw
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
mpl.use("Agg")

def all_tests():
    results_all = []
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for num in x:
        g, real_label = inputdata.read_lfr(num)

        a = vlpa.vpa(g)
        b = vlpa.vpas(g)
        c = vlpa.lpa(g)
        d = vlpa.clustering_infomap(g)
        e = vlpa.louvain(g)

        mod_real = community.modularity(real_label, g)

        result = [num,real_label,a,b,c,d,e,mod_real]
        results_all.append(result)
        # nmi_f.append(nmi(real_label, f))
    with open('results_all.dat', 'wb') as f:
        pickle.dump(results_all, f)

def modularity_compare_plot():
    with open('results_all.dat', 'r') as f:
        results = pickle.load(f)
    x = []
    a_plots = []
    b_plots = []
    c_plots = []
    d_plots = []
    e_plots = []
    real_plots = []
    for result in results:
        num = result[0]
        real_label = result[1]
        a = result[2]
        b = result[3]
        c = result[4]
        d = result[5]
        e = result[6]
        mod_real = result[7]
        if num>0.0:
            x.append(num)
            a_plots.append(a.mod/e.mod)
            b_plots.append(b.mod/e.mod)
            c_plots.append(c.mod/e.mod)
            d_plots.append(d.mod/e.mod)
            e_plots.append(e.mod/e.mod)
            #real_plots.append(mod_real/e.mod)
        print(x)

    plt.figure(1)
    plt.plot(x, a_plots, label='VLPA')
    plt.plot(x, b_plots, label='sVLPA')
    plt.plot(x, c_plots, label='lpa')
    plt.plot(x, d_plots, label='infomap')
    plt.plot(x, e_plots, label='Louvain')
    #plt.plot(x, real_plots, label='real')

    plt.legend(loc='lower left')
    plt.xlabel('$\mu$')
    plt.ylabel('relative Modularity')
    plt.savefig('relative_modularity_compare.eps')


modularity_compare_plot()