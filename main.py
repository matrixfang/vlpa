import networkx as nx
import numpy as np

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
from collections import Counter
mpl.use("Agg")


def nmi(labels_real, labels):
    # normalized mutual information
    list_cal = []
    list_real = []
    for node in labels:
        list_cal.append(labels[node])
        list_real.append(labels_real[node])
    return normalized_mutual_info_score(list_cal, list_real)


"""first_compare"""


def first_compare():
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

        a = vlpa.sgd_vlpa(g).labels
        b = vlpa.lpa(g).labels
        c = vlpa.clustering_infomap(g).labels
        d = vlpa.louvain(g).labels
        # f = vlpa.vlpa3(g)
        nmi_a.append(nmi(real_label, a))
        nmi_b.append(nmi(real_label, b))
        nmi_c.append(nmi(real_label, c))
        nmi_d.append(nmi(real_label, d))
        # nmi_f.append(nmi(real_label, f))
    # plot
    compare_result = [x, nmi_a, nmi_b, nmi_c, nmi_d,nmi_e]
    with open('first_compare.dat', 'wb') as f:
        pickle.dump(compare_result, f)


def first_compare_plot():
    with open('first_compare.dat', 'r') as f:
        result = pickle.load(f)
    plt.figure(1)
    plt.plot(result[0], result[1], label='vlpa')
    plt.plot(result[0], result[2], label='lpa')
    plt.plot(result[0], result[3], label='infomap')
    plt.plot(result[0], result[4], label='louvain')
    # plt.plot(x, nmi_e, label='vlpa2')
    # plt.plot(x, nmi_f, label='vlpa3')
    plt.legend(loc='lower left')
    plt.xlabel('$\mu$')
    plt.ylabel('NMI')
    plt.savefig('first_compare.eps')
    pass


"""
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
"""


def convergence_test():
    # x = [0.0, 0.1]
    g, real_label = inputdata.read_lfr(0.5)
    opt_value = community.modularity(real_label, g)
    result1 = vlpa.first_vlpa(g, gamma=0.1)
    result2 = vlpa.first_vlpa(g, gamma=0.2)
    result3 = vlpa.first_vlpa(g, gamma=0.4)
    result4 = vlpa.first_vlpa(g, gamma=0.6)
    result5 = vlpa.first_vlpa(g, gamma=0.8)
    result6 = vlpa.first_vlpa(g, gamma=1.0)
    results = [result1, result2, result3, result4, result5, result6]

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
    for key in input: # get the key_max and key_min
        mods = input[key]
        logmods = []
        x = np.log10(range(1, len(mods) + 1))
        for v in mods:
            logmods.append(np.log10(abs(v - opt_value) + eps))
        fit = np.polyfit(x[10:30], logmods[10:30], 1)
        if fit[0] < k_min:
            k_min = fit[0]
            key_min = key
        else:
            pass
        if fit[0]> k_max:
            k_max = fit[0]
            key_max = key
        else:
            pass

    for key in input:
        mods = input[key]
        logmods = []
        x = np.log10(range(1, len(mods) + 1))
        for v in mods:
            logmods.append(np.log10(abs(v - opt_value) + eps))
        if key in [key_max, key_min]:
            pass
        else:
            plt.plot(x,logmods)


    print(k_min)
    print(k_max)
    logmods_min = []
    for v1 in input[key_min]:
        logmods_min.append(np.log10(abs(v1 - opt_value) + eps))
    real_key = re.findall(r"\d+\.?\d*", key_min)
    plt.plot(x, logmods_min,label = '$\gamma=$' + str(real_key[0]))
    plot.fit_plot(x[10:30], logmods_min[10:30], upmove=-0.1)


    logmods_max = []
    for v2 in input[key_max]:
        logmods_max.append(np.log10(abs(v2 - opt_value) + eps))
    real_key = re.findall(r"\d+\.?\d*", key)
    plt.plot(x, logmods_max, label='$\gamma=$' + str(real_key[0]))
    plot.fit_plot(x[10:30], logmods_max[10:30], upmove=+0.1)
    plt.legend(loc='lower left')
    plt.xlabel('$\log_{10} (step)$')
    plt.ylabel('$\log_{10}(\|Q-Q_{best}\|)$')
    plt.savefig('convergence_plot.eps')
    plt.close()
    pass


def useful_dim_test():
    g, real_label = inputdata.read_lfr(0.6)
    result,vecs = vlpa.dim_test_vlpa(g,k=12)
    with open('useful_dim_test.dat', 'wb') as f:
        pickle.dump(vecs, f)
    pass


def useful_dim_test_plot():
    def num(label, p):
        list_sorted = sorted(label, key=lambda x: label[x], reverse=True)
        num = 0
        value = 0.0
        for i in list_sorted:
            if value < np.sqrt(p):
                value += (label[i])**2
                num += 1
        return num

    with open('useful_dim_test.dat', 'r') as f:
        vecs = pickle.load(f)
        degree = []
        dims_num = []
        for k in vecs:
            degree.append(len(vecs[k]))
            dims_num.append(num(vecs[k], 0.95))
            print(len(vecs[k]), num(vecs[k], 0.95))
        plt.figure()
        plt.subplot(121)
        plt.hist(degree)
        plt.xlabel("Degree of node")
        plt.ylabel("Frequency")
        plt.subplot(122)
        plt.hist(dims_num)
        plt.xlabel(' "Useful" dimensions of node')
        plt.savefig('useful_dims.eps')
        plt.show()


def form_modularity_compare():
    data = {'first_vlpa': [], 'fixed_zeronorm_vlpa': [], 'sgd_vlpa': [], 'random_vlpa': [],'louvain':[]}
    for x in [0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9]:
        g, real_label = inputdata.read_lfr(x)
        result1 = vlpa.first_vlpa(g,ifrecord=False)
        result2 = vlpa.fixed_zeronorm_vlpa(g,ifrecord=False)
        result3 = vlpa.sgd_vlpa(g,ifrecord=False)
        result4 = vlpa.random_vlpa(g,ifrecord=False)
        result5 = vlpa.louvain(g, ifrecord=False)
        data['first_vlpa'].append(result1.mod)
        data['fixed_zeronorm_vlpa'].append(result2.mod)
        data['sgd_vlpa'].append(result3.mod)
        data['random_vlpa'].append(result4.mod)
        data['louvain'].append(result5.mod)
    with open('form_modularity_compare.dat', 'wb') as f:
        pickle.dump(data, f)


def form_modularity_compare_plot():
    with open('form_modularity_compare.dat', 'r') as f:
        data = pickle.load(f)

    df = pd.DataFrame(data)
    print(df.to_latex())


def final_compare():
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

        a = vlpa.first_vlpa(g,ifrecord=False).labels
        b = vlpa.sgd_vlpa(g,ifrecord=False).labels
        c = vlpa.random_vlpa_best(g,ifrecord=False).labels
        d = vlpa.louvain(g,ifrecord=False).labels
        e = vlpa.fixed_zeronorm_vlpa(g,ifrecord=False).labels
        # f = vlpa.vlpa3(g)
        nmi_a.append(nmi(real_label, a))
        nmi_b.append(nmi(real_label, b))
        nmi_c.append(nmi(real_label, c))
        nmi_d.append(nmi(real_label, d))
        nmi_e.append(nmi(real_label, e))
        # nmi_f.append(nmi(real_label, f))
    # plot
    compare_result = [x, nmi_a, nmi_b, nmi_c, nmi_d,nmi_e]
    with open('final_compare.dat', 'wb') as f:
        pickle.dump(compare_result, f)


def final_compare_plot():
    with open('final_compare.dat', 'r') as f:
        result = pickle.load(f)
    plt.figure(1)
    plt.plot(result[0], result[1], label='vlpa')
    plt.plot(result[0], result[2], label='sgd_vlpa')
    plt.plot(result[0], result[3], label='sgd_all')
    plt.plot(result[0], result[4], label='louvain')
    #plt.plot(result[0], result[5], label='fixed_zeronorm_vlpa')
    # plt.plot(x, nmi_e, label='vlpa2')
    # plt.plot(x, nmi_f, label='vlpa3')
    plt.legend(loc='lower left')
    plt.xlabel('$\mu$')
    plt.ylabel('NMI')
    plt.savefig('final_compare.png')
    pass



"""temporal unused"""

def louvain_compare():
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

        a = vlpa.first_vlpa(g)
        b = vlpa.lpa(g)
        c = vlpa.clustering_infomap(g)
        d = vlpa.louvain(g).labels
        e = vlpa.louvain_vlpa(g).labels
        # f = vlpa.vlpa3(g)
        nmi_a.append(nmi(real_label, a))
        nmi_b.append(nmi(real_label, b))
        nmi_c.append(nmi(real_label, c))
        nmi_d.append(nmi(real_label, d))
        nmi_e.append(nmi(real_label, e))
        # nmi_f.append(nmi(real_label, f))
    # plot
    compare_result = [x, nmi_a, nmi_b, nmi_c, nmi_d, nmi_e]
    with open('louvian_compare.dat', 'wb') as f:
        pickle.dump(compare_result, f)


def louvain_compare_plot():
    with open('louvian_compare.dat', 'r') as f:
        result = pickle.load(f)
    plt.figure(1)
    plt.plot(result[0], result[1], label='vlpa')
    plt.plot(result[0], result[2], label='lpa')
    plt.plot(result[0], result[3], label='infomap')
    plt.plot(result[0], result[4], label='louvain')
    plt.plot(result[0], result[5], label='louvian_vlpa')
    # plt.plot(x, nmi_f, label='vlpa3')
    plt.legend(loc='lower left')
    plt.xlabel('$\mu$')
    plt.ylabel('NMI')
    plt.savefig('louvian_compare.png')
    pass


def modularity_compare():
    x = [0.0,0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9]
    mod1 = []
    mod2 = []
    mod3 = []
    mod4 = []
    mod5 = []
    for num in x:
        g, real_label = inputdata.read_lfr(num)
        result1 = vlpa.vpa(g,ifrecord=False)
        result2 = vlpa.vpas(g,ifrecord=False)
        result3 = vlpa.lpa(g)
        result4 = vlpa.clustering_infomap(g)
        result5 = vlpa.louvain(g)
        mod_real = community.modularity(real_label, g)

        mod1.append(result1.mod/result5.mod)
        mod2.append(result2.mod/result5.mod)
        mod3.append(result3.mod/result5.mod)
        mod4.append(result4.mod/result5.mod)
        mod5.append(mod_real/result5.mod)

    # plot
    result = [x, mod1, mod2, mod3, mod4, mod5]
    with open('modularity_compare.dat', 'wb') as f:
        pickle.dump(result, f)
    pass


def modularity_compare_plot():
    with open('modularity_compare.dat', 'r') as f:
        result = pickle.load(f)
        plt.figure(1)
        plt.plot(result[0], result[1], label='VPA')
        plt.plot(result[0], result[2], label='VPAs')
        plt.plot(result[0], result[3], label='lpa')
        plt.plot(result[0], result[4], label='infomap')
        plt.plot(result[0], result[5], label='real')

        plt.legend(loc='lower left')
        plt.xlabel('$\mu$')
        plt.ylabel('relative Modularity')
        plt.savefig('relative_modularity_compare.png')


#modularity_compare_plot()

def louvain_modularity_compare():
    x = [0.5, 0.6, 0.7, 0.8, 0.9]
    mod1 = []
    mod2 = []
    mod3 = []
    mod4 = []
    for num in x:
        g, real_label = inputdata.read_lfr(num)
        result1 = vlpa.first_vlpa(g)
        result2 = vlpa.louvain(g)
        result3 = vlpa.fixed_pos_louvain_vlpa(g, 5)
        benchmark = community.modularity(real_label, g)

        mod1.append(nmi(result1.labels, real_label))
        mod2.append(nmi(result2.labels, real_label))
        mod3.append(nmi(result3.labels, real_label))
        mod4.append(benchmark)

    # plot
    result = [x, mod1, mod2, mod3, mod4]
    with open('louvain_modularity_compare.dat', 'wb') as f:
        pickle.dump(result, f)
    pass


def louvain_modularity_compare_plot():
    with open('louvain_modularity_compare.dat', 'r') as f:
        result = pickle.load(f)
        plt.figure(1)
        plt.plot(result[0], result[1], label='vlpa')
        plt.plot(result[0], result[2], label='louvain')
        plt.plot(result[0], result[3], label='vlpa_sgd')
        plt.plot(result[0], result[4], '--', label='benchmark')

        plt.legend(loc='upper left')
        plt.xlabel('$\mu$')
        plt.ylabel('Modularity')
        plt.savefig('louvain_modularity_compare.png')


def several_idea_compare():
    mod_a = []
    mod_b = []
    mod_c = []
    mod_d = []
    mod_e = []
    mod_f = []
    mod_s = []
    x = [0.1]
    #x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9]
    for num in x:
        g, real_label = inputdata.read_lfr(num)

        a = vlpa.first_vlpa(g)
        b = vlpa.lpa(g)
        c = vlpa.clustering_infomap(g)
        d = vlpa.louvain(g).labels
        e = vlpa.fixed_pos_vlpa(g, 5)
        f = vlpa.fixed_pos_vlpa(g, 10)
        # f = vlpa.vlpa3(g)
        mod_a.append(community.modularity(a, g))
        mod_b.append(community.modularity(b, g))
        mod_c.append(community.modularity(c, g))
        mod_d.append(community.modularity(d, g))
        mod_e.append(community.modularity(e, g))
        mod_f.append(community.modularity(f, g))
        mod_s.append(community.modularity(real_label, g))
        # nmi_f.append(nmi(real_label, f))
    # plot
    compare_result = [x, mod_a, mod_b, mod_c, mod_d, mod_e, mod_f, mod_s]
    with open('several_idea_compare.dat', 'wb') as f:
        pickle.dump(compare_result, f)


def several_idea_compare_plot():
    with open('several_idea_compare.dat', 'r') as f:
        result = pickle.load(f)
    plt.figure(1)
    plt.plot(result[0], result[1], label='vlpa')
    plt.plot(result[0], result[2], label='no_renorm')
    plt.plot(result[0], result[3], label='renorm')
    # plt.plot(x, nmi_f, label='vlpa3')
    print(result[2])
    print(result[3])
    plt.legend(loc='lower left')
    plt.xlabel('\mu')
    plt.ylabel('NMI')
    plt.savefig('renorm_compare.png')
    pass





def time_test(method):
    x = [800, 1200, 1600, 2000]
    plus = []
    times = []
    for num in x:
        g, real_label = inputdata.read_lfr(num)
        t1 = time.time()
        result = method(g,ifrecord=False)
        t2 = time.time()
        print('benchmark modularity is %f' % (community.modularity(real_label, g)))
        plus.append(np.log10(len(g.edges()) + len(g.nodes())))
        times.append(np.log10(t2-t1))
    return plus, times


def time_plot(method):
    x, y = time_test(method)
    plot.fit_plot(x, y)
    plt.savefig('set_compare.png')


def time_complexity_plot():
    data = {}
    medthods = [vlpa.first_vlpa, vlpa.fixed_pos_vlpa, vlpa.fixed_pos_louvain_vlpa]
    for method in methods:
        x, y = time_test(method)
        data[method] = [x, y]
        pass


def plot_graph(g, label):
    node_color = []
    edge_color = []
    node_size = []
    for n in g.nodes():
        node_color.append(label[n])
        if g.degree(n) != 0:
            node_size.append(25 * g.degree(n))
        else:
            node_size.append(40)

    for edge in g.edges():
        if edge[0] == edge[1]:
            edge_color.append(float(label[edge[0]]))
        else:
            label1 = float(label[edge[1]])
            label2 = float(label[edge[0]])
            edge_color.append((label1 + label2) / 2)

    f = plt.figure()

    pos = nx.fruchterman_reingold_layout(g)
    nx.draw_networkx_edges(g, pos=pos, alpha=0.3, edge_color=edge_color, width=2, style='solid')
    nx.draw_networkx_nodes(g, pos=pos, alpha=0.5, with_labels=False,
                           node_size=node_size, node_color=node_color)
    f.set_rasterized(True)
    plt.show()
    plt.close()
    # plt.savefig("network"+str(num)+".eps",rasterized=True,dpi=300)


def modularity_test(method):
    x = [200, 400, 800, 1200, 1600]
    mod_real = []
    mod_opt = []
    for num in x:
        g, real_label = inputdata.read_lfr(num)
        result = method(g,ifrecord=False)
        print('benchmark modularity is %f' % (community.modularity(real_label, g)))
        mod_real.append(community.modularity(real_label, g))
        mod_opt.append(community.modularity(result.labels, g))
    return mod_real, mod_opt


def modularity_test_plot():
    plt.figure()
    m1, opt = modularity_test(vlpa.louvian)
    m2, opt2 = modularity_test(vlpa.louvian_vlpa)
    plt.plot(range(len(opt)), m1, label='louvian')
    plt.plot(range(len(opt)), m2, label='louvian_vlpa')
    # plt.plot(range(len(opt)),opt,label='bench_mark')
    plt.legend()
    plt.show()
    plt.close()
    pass


def louvain_test():
    g, real_label = inputdata.read_lfr(0.6)
    result1 = vlpa.louvain_vlpa(g, gamma=0.2)
    result2 = vlpa.louvain_vlpa(g, gamma=0.5)
    result3 = vlpa.louvain_vlpa(g, gamma=0.7)
    opt_value = community.modularity(real_label, g)

    print(opt_value)
    print('louvian method', vlpa.louvain(g).mod)
    print(result1.vmods)

    print(set(real_label.values()))
    draw.draw_group(g, real_label, result2.labels)
    plt.show()
    plt.close()
    convergence_rate_plot({0: result1.mods, 1: result2.mods, 2: result3.mods}, opt_value)
    pass


def convergence_rate_plot(input, opt_value):
    max_number = 0
    plt.figure()
    for key in input:
        mods = input[key]
        x = range(len(mods))
        if len(mods) > max_number:
            max_number = len(mods)
        plt.plot(x, mods, label=key)

    standard_values = []
    for i in xrange(max_number):
        standard_values.append(opt_value)

    plt.plot(range(max_number), standard_values, label='best')
    plt.legend()
    plt.show()
    plt.close()


def just_modularity_test():
    """
    just test any algorithm
    :return:
    """
    g, real_label = inputdata.read_lfr(2000)
    #g = nx.erdos_renyi_graph(1000,0.02)

    #result1 = vlpa.fixed_pos_vlpa(g,4)
    #result2 = vlpa.louvain_vlpa(g)
    #result3 = vlpa.vlpa(g)
    #result4 = vlpa.louvain(g)
    #result5 = vlpa.clustering_infomap(g)
    #result6 = vlpa.lpa(g)
    #result7 = vlpa.fixed_pos_louvain_vlpa(g,4)
    result8 = vlpa.fixed_pos_louvain_vlpa(g, gamma=1.0)

    result = vlpa.louvain(g)

    opt_value = community.modularity(result.labels, g)
    print(result.labels)
    print(set(result8.labels.values()))
    print(opt_value)

    print(len(set(result.labels.values())), len(set(result8.labels.values())))

    # print(result1.mod)
    #draw.draw_group(g, real_label, result.labels)

    # plt.show()
    # plt.close()
    convergence_rate_plot({'mods': result8.mods, 'vmods': result8.vmods}, opt_value)
    pass


def just_speed_test():
    g, real_label = inputdata.read_lfr(0.6)
    # result2 = vlpa.real_final_agg_louvain(g,5)
    # result1 = vlpa.fixed_pos_louvain_vlpa(g,5)
    # result3 = vlpa.final_agg_louvain(g,5)
    #result6 = vlpa.final_vlpa(g)
    result2 = vlpa.first_vlpa(g, gamma=0.2)
    result3 = vlpa.first_vlpa(g, gamma=0.4)
    result4 = vlpa.first_vlpa(g, gamma=0.6)
    result5 = vlpa.first_vlpa(g, gamma=0.8)
    opt_value = community.modularity(real_label, g)
    #opt_value = vlpa.louvain(g).mod
    plot_dict = {result2.algorithm: result2.mods,result3.algorithm:result3.mods,
                 result4.algorithm: result5.mods,result5.algorithm: result5.mods}
    convergence_rate_plot(plot_dict, opt_value)
    pass


def temporal_test():
    g, real_label = inputdata.read_lfr(0.5)

    result1 = vlpa.different_init_vpa(g)
    #result2 = vlpa.mixed_vpas(g)
    result3 = vlpa.vpa(g)
    result4 = vlpa.louvain(g)

    opt_value = community.modularity(result4.labels, g)
    #opt_value = vlpa.louvain(g).mod
    print(set(real_label.values()))
    print('\n''*****')
    print(sorted(dict(Counter(result1.labels.values())).values()))
    #print(set(result2.labels.values()))
    print(set(result3.labels.values()))
    print(sorted(dict(Counter(result4.labels.values())).values()))

    plot_dict = {result1.algorithm: result1.mods,
                 #result2.algorithm: result2.mods,
                 result3.algorithm: result3.mods
                 }

    convergence_rate_plot(plot_dict, opt_value)
    pass



def run_method(method):
    g, real_label = inputdata.read_lfr(0.9)
    method(g,ifrecord=False)

def time_length():
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9]
    T = []
    for num in x:
        g, real_label = inputdata.read_lfr(num)
        t = vlpa.time_sgd_vlpa(g)
        T.append(t)
    plt.figure()
    plt.plot(T)
    plt.show()

"""paper_v1.1"""


def all_tests():
    results_all = {}
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for num in x:
        g, real_label = inputdata.read_lfr(num)

        a = vlpa.vpa(g)
        b = vlpa.vpas(g)
        c = vlpa.lpa(g)
        d = vlpa.clustering_infomap(g)
        e = vlpa.louvain(g)
        results_all[num] = [real_label,a,b,c,d,e]
        # nmi_f.append(nmi(real_label, f))
    with open('results_all.dat', 'wb') as f:
        pickle.dump(results_all, f)

def NMI_compare_plot():
    with open('results_all.dat', 'r') as f:
        results = pickle.load(f)
    nmi_a = []
    nmi_b = []
    nmi_c = []
    nmi_d = []
    nmi_e = []
    for num in results.keys():
        real_label = results[num][0]
        result_a = results[num][1]
        result_b = results[num][2]
        result_c = results[num][3]
        result_d = results[num][4]
        result_e = results[num][5]
        nmi_a.append(nmi(real_label, result_a.labels))
        nmi_b.append(nmi(real_label, result_b.labels))
        nmi_c.append(nmi(real_label, result_c.labels))
        nmi_d.append(nmi(real_label, result_d.labels))
        nmi_e.append(nmi(real_label, result_e.labels))
    print(nmi_a)
    print(nmi_b)
    print(nmi_c)
    print(nmi_d)
    print(nmi_e)
    plt.figure(1)
    plt.plot(results.keys(), nmi_a, label='VLPA')
    plt.plot(results.keys(), nmi_b, label='sVLPA')
    plt.plot(results.keys(), nmi_c, label=result_c.algorithm)
    plt.plot(results.keys(), nmi_d, label=result_d.algorithm)
    plt.plot(results.keys(), nmi_e, label=result_e.algorithm)
    plt.legend(loc='lower left')
    plt.xlabel('$\mu$')
    plt.ylabel('NMI')
    plt.savefig('NMI_compare.eps')
    pass

def essential_dimension():
    plt.figure(1)
    x = [0.7, 0.9]
    d_all = [2,4,6,8,10]
    for num in x:
        plots = []
        for d in d_all:
            g, real_label = inputdata.read_lfr(num)
            result = vlpa.vpa(g,k=d)
            plots.append(result.mod)
        print(plots)
        plt.plot(d_all,plots,label = '\mu = '+str(num))
    plt.legend(loc='lower left')
    plt.xlabel('$\mu$')
    plt.ylabel('Modularity')
    plt.savefig('essential_dimension.eps')
    pass



def method_test():
    for x in [0.9,0.8,0.7,0.6,0.5,0.4]:
        g, real_label = inputdata.read_lfr(x)
        result1 = vlpa.synchronous_vpa(g,ifrecord=False)
        result2 = vlpa.vpa(g,ifrecord=False)
        resutl3 = vlpa.louvain(g)
    pass

NMI_compare_plot()