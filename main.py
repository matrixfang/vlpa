import networkx as nx
import numpy as np
import scipy as sp
import draw
import cProfile


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
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9]
    for num in x:
        g, real_label = inputdata.read_lfr(num)

        a = vlpa.first_vlpa(g).labels
        b = vlpa.lpa(g).labels
        c = vlpa.clustering_infomap(g).labels
        d = vlpa.louvain(g).labels
        # e = vlpa.vlpa2(g)
        # f = vlpa.vlpa3(g)
        nmi_a.append(nmi(real_label, a))
        nmi_b.append(nmi(real_label, b))
        nmi_c.append(nmi(real_label, c))
        nmi_d.append(nmi(real_label, d))
        # nmi_e.append(nmi(real_label, e))
        # nmi_f.append(nmi(real_label, f))
    # plot
    compare_result = [x,nmi_a,nmi_b,nmi_c,nmi_d]
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
    plt.savefig('first_compare.png')
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
    result1 = vlpa.convergence_vlpa(g, gamma=0.1, mod='nothing')
    result2 = vlpa.convergence_vlpa(g, gamma=0.2, mod='nothing')
    result3 = vlpa.convergence_vlpa(g, gamma=0.4, mod='nothing')
    result4 = vlpa.convergence_vlpa(g, gamma=0.6, mod='nothing')
    result5 = vlpa.convergence_vlpa(g, gamma=0.8, mod='nothing')
    result6 = vlpa.convergence_vlpa(g, gamma=1.0, mod='nothing')
    results = [result1,result2,result3,result4,result5,result6]

    with open('convergence_test.dat', 'wb') as f:
        pickle.dump(results, f)


def convergence_test_plot():
    with open('convergence_test.dat', 'r') as f:
        results = pickle.load(f)

    input  ={}
    for result in results:
        input[result.algorithm] = result.mods


    opt_value = result.mod
    convergence_gamma_plot(input,opt_value)


def useful_dim_test():
    g, real_label = inputdata.read_lfr(0.6)
    vecs = vlpa.dim_test_vlpa(g)
    with open('useful_dim_test.dat', 'wb') as f:
        pickle.dump(vecs, f)
    pass


def useful_dim_test_plot():
    def num(label,p):
        list_sorted = sorted(label,key=lambda x:label[x],reverse=True)
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
        plt.savefig('useful_dims.png')
        plt.show()


def form_modularity_compare():
    data = {'vlpa':[],'louvain':[],'louvain_vlpa':[],'sgd_vlpa':[]}
    for x in [0.5,0.6,0.7,0.8, 0.9]:
        g, real_label = inputdata.read_lfr(x)
        result1 = vlpa.first_vlpa(g)
        result2 = vlpa.louvain(g)
        result3 = vlpa.louvain_vlpa(g)
        result4 = vlpa.fixed_pos_louvain_vlpa(g)
        data['vlpa'].append(result1.mod)
        data['louvain'].append(result2.mod)
        data['louvain_vlpa'].append(result3.mod)
        data['sgd_vlpa'].append(result4.mod)
    with open('form_modularity_compare.dat', 'wb') as f:
        pickle.dump(data, f)


def form_modularity_compare_plot():
    with open('form_modularity_compare.dat', 'r') as f:
        data = pickle.load(f)
    for key in data:
        print(key)
        print(data[key])


def louvain_compare():
    nmi_a = []
    nmi_b = []
    nmi_c = []
    nmi_d = []
    nmi_e = []
    nmi_f = []
    # x = [0.0, 0.1]
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9]
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
    compare_result = [x,nmi_a,nmi_b,nmi_c,nmi_d,nmi_e]
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
    x = [0.5, 0.6, 0.7, 0.8, 0.9]
    mod1 = []
    mod2 = []
    mod3 = []
    mod4 = []
    mod5 = []
    for num in x:
        g, real_label = inputdata.read_lfr(num)
        result1 = vlpa.first_vlpa(g)
        result2 = vlpa.lpa(g)
        result3 = vlpa.clustering_infomap(g)
        result4 = vlpa.louvain(g)

        mod1.append(result1.mod)
        mod2.append(result2.mod)
        mod3.append(result3.mod)
        mod4.append(result4.mod)
        mod5.append(community.modularity(real_label,g))

    # plot
    result = [x, mod1, mod2, mod3, mod4,mod5]
    with open('modularity_compare.dat', 'wb') as f:
        pickle.dump(result, f)
    pass


def modularity_compare_plot():
    with open('modularity_compare.dat', 'r') as f:
        result = pickle.load(f)
        plt.figure(1)
        plt.plot(result[0], result[1], label='vlpa')
        plt.plot(result[0], result[2], label='lpa')
        plt.plot(result[0], result[3], label='infomap')
        plt.plot(result[0], result[4], label='louvain')
        plt.plot(result[0], result[5],label='benchmark')

        plt.legend(loc='lower left')
        plt.xlabel('$\mu$')
        plt.ylabel('Modularity')
        plt.savefig('modularity_compare.png')


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
        result3 = vlpa.fixed_pos_louvain_vlpa(g,5)
        benchmark = community.modularity(real_label,g)

        mod1.append(nmi(result1.labels,real_label))
        mod2.append(nmi(result2.labels,real_label))
        mod3.append(nmi(result3.labels,real_label))
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
        plt.plot(result[0], result[3],label='vlpa_sgd')
        plt.plot(result[0], result[4], '--',label='benchmark')


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
        e = vlpa.fixed_pos_vlpa(g,5)
        f = vlpa.fixed_pos_vlpa(g,10)
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
    compare_result = [x,mod_a,mod_b,mod_c,mod_d,mod_e,mod_f,mod_s]
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
        a = vlpa.first_vlpa(g)
        b = vlpa.lpa(g)
        c = vlpa.clustering_infomap(g)
        d = community.best_partition(g)
        e = vlpa.basic_vlpa(g)
        #e = vlpa.final_vlpa(g, gamma=0.9, mod='normalize', pos_shrink='True')
        f = vlpa.nonmodel(g)
        nmi_a.append(nmi(real_label, a))
        nmi_b.append(nmi(real_label, b))
        nmi_c.append(nmi(real_label, c))
        nmi_d.append(nmi(real_label, d))
        nmi_e.append(nmi(real_label, e))
        nmi_f.append(nmi(real_label, f))
    # plot
    plt.figure(1)
    plt.plot(x, nmi_a, label='vlpa')
    plt.plot(x, nmi_b, label='lpa')
    plt.plot(x, nmi_c, label='infomap')
    plt.plot(x, nmi_d, label='louvain')
    plt.plot(x, nmi_e, label='final_vlpa')
    plt.plot(x, nmi_f, label='nonmodel')
    plt.legend(loc='upper left')
    plt.savefig('final_compare.png')


def time_test(method):
    x = [200, 400, 800, 1200, 1600,2000]
    plus = []
    times = []
    for num in x:
        g, real_label = inputdata.read_lfr(num)
        t1 = time.time()
        result = method(g)
        t2 = time.time()
        print('benchmark modularity is %f'%(community.modularity(real_label, g)))
        plus.append(np.log10(len(g.edges())+len(g.nodes()) ))
        times.append(np.log10(abs(t2-t1)))
    return plus, times


def time_plot():
    x,y = time_test(vlpa.louvain_vlpa)
    plot.fit_plot(x,y)
    plt.savefig('set_compare.png')

def time_complexity_plot():
    data = {}
    medthods = [vlpa.first_vlpa, vlpa.fixed_pos_vlpa, vlpa.fixed_pos_louvain_vlpa]
    for method in methods:
        x,y =time_test(method)
        data[method] = [x,y]
        pass

def plot_graph(g, label):
    node_color = []
    edge_color = []
    node_size = []
    for n in g.nodes():
        node_color.append(label[n])
        if g.degree(n)!=0:
            node_size.append(25 * g.degree(n))
        else:
            node_size.append(40)


    for edge in g.edges():
        if edge[0]==edge[1]:
            edge_color.append(float(label[edge[0]]))
        else:
            label1 = float(label[edge[1]])
            label2 = float(label[edge[0]])
            edge_color.append((label1+label2)/2)


    f = plt.figure()

    pos = nx.fruchterman_reingold_layout(g)
    nx.draw_networkx_edges(g, pos=pos, alpha=0.3, edge_color = edge_color,width=2, style='solid')
    nx.draw_networkx_nodes(g, pos=pos, alpha=0.5, with_labels=False, node_size=node_size, node_color=node_color)
    f.set_rasterized(True)
    plt.show()
    plt.close()
    #plt.savefig("network"+str(num)+".eps",rasterized=True,dpi=300)

def modularity_test(method):
    x = [200, 400, 800, 1200, 1600]
    mod_real = []
    mod_opt = []
    for num in x:
        g, real_label = inputdata.read_lfr(num)
        result = method(g)
        print('benchmark modularity is %f' % (community.modularity(real_label, g)))
        mod_real.append(community.modularity(real_label, g))
        mod_opt.append(community.modularity(result.labels, g))
    return mod_real, mod_opt

def modularity_test_plot():
    plt.figure()
    m1,opt=modularity_test(vlpa.louvian)
    m2,opt2=modularity_test(vlpa.louvian_vlpa)
    plt.plot(range(len(opt)),m1, label = 'louvian')
    plt.plot(range(len(opt)),m2,label='louvian_vlpa')
    #plt.plot(range(len(opt)),opt,label='bench_mark')
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

def convergence_gamma_plot(input,opt_value):
    plt.figure()
    eps = 0.0001
    k_min = 0
    key_min = None
    for key in input:
        mods = input[key]
        logmods = []
        x = np.log10(range(1,len(mods)+1))
        for v in mods:
            logmods.append(np.log10(abs(v - opt_value)+eps))
        plt.plot(x, logmods, label=key)
        fit = np.polyfit(x[10:30],logmods[10:30],1)
        if fit[0]<k_min:
            k_min = fit[0]
            key_min = key
        else:
            pass
    print(k_min)
    plot.fit_plot(x[10:30], logmods[10:30], upmove=0.1)
    logmods = []
    for v in input[key_min]:
        logmods.append(np.log10(abs(v - opt_value) + eps))

    plot.fit_plot(x[10:30],logmods[10:30], upmove=-0.1)


    plt.legend()
    plt.savefig('convergence_plot.png')
    plt.close()
    pass

def convergence_rate_plot(input,opt_value):
    max_number = 0
    plt.figure()
    for key in input:
        mods = input[key]
        x = range(len(mods))
        if len(mods)>max_number:
            max_number = len(mods)
        plt.plot(x,mods,label=key)

    standard_values = []
    for i in xrange(max_number):
        standard_values.append(opt_value)

    plt.plot(range(max_number),standard_values,label='best')
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
    result8 = vlpa.fixed_pos_louvain_vlpa(g,gamma=1.0)

    result = vlpa.louvain(g)

    opt_value = community.modularity(result.labels, g)
    print(result.labels)
    print(set(result8.labels.values()))
    print(opt_value)

    print(len(set(result.labels.values())),len(set(result8.labels.values())))


    #print(result1.mod)
    #draw.draw_group(g, real_label, result.labels)

    #plt.show()
    #plt.close()
    convergence_rate_plot({'mods':result8.mods,'vmods':result8.vmods}, opt_value)
    pass

def just_speed_test():
    g, real_label = inputdata.read_lfr(0.7)
    # result2 = vlpa.real_final_agg_louvain(g,5)
    # result1 = vlpa.fixed_pos_louvain_vlpa(g,5)
    # result3 = vlpa.final_agg_louvain(g,5)
    #result6 = vlpa.final_vlpa(g)
    result2 = vlpa.convergence_vlpa(g, gamma=0.2, mod='nothing')
    result3 = vlpa.convergence_vlpa(g, gamma=0.4, mod='nothing')
    result4 = vlpa.convergence_vlpa(g, gamma=0.6, mod='nothing')
    result5 = vlpa.convergence_vlpa(g, gamma=0.8, mod='nothing')
    result6 = vlpa.convergence_vlpa(g, gamma=1.0, mod='nothing')
    opt_value = community.modularity(result4.labels, g)
    #opt_value = vlpa.louvain(g).mod
    plot_dict = {result2.algorithm:result2.mods,result3.algorithm:result3.mods,
                 result4.algorithm:result4.mods,result5.algorithm:result5.mods,
                 result6.algorithm:result6.mods}
    convergence_rate_plot(plot_dict, opt_value)
    pass

def temporal_test():
    g, real_label = inputdata.read_lfr(0.9)
    # result2 = vlpa.real_final_agg_louvain(g,5)
    # result1 = vlpa.fixed_pos_louvain_vlpa(g,5)
    # result3 = vlpa.final_agg_louvain(g,5)
    #result6 = vlpa.final_vlpa(g)
    #result2 = vlpa.mixed_method(g)
     #result3 = vlpa.fixed_pos_louvain_vlpa_dot(g)
    result1 = vlpa.first_vlpa(g)
    result2 = vlpa.fixed_zeronorm_vlpa(g)
    result3 = vlpa.fixed_zeronorm_vlpa_dot(g)
    result4 = vlpa.louvain(g)

    opt_value = community.modularity(result4.labels, g)
    #opt_value = vlpa.louvain(g).mod
    print(set(real_label.values()))
    #print(set(result2.labels.values()))
    print(set(result1.labels.values()))
    print(set(result4.labels.values()))

    plot_dict = {result1.algorithm:result1.mods,
                 result2.algorithm: result2.mods,
                 result3.algorithm: result3.mods}
    convergence_rate_plot(plot_dict, opt_value)
    pass

def run_method(method):
    g,real_label = inputdata.read_lfr(0.9)
    method(g,ifrecord=False)
    pass

first_compare()