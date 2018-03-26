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

        a = vlpa.vlpa(g)
        b = vlpa.lpa(g)
        c = vlpa.clustering_infomap(g)
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
    #label = vlpa.clustering_infomap(g)
    #label2 = vlpa.basic_vlpa(g)
    label3 = vlpa.ada_vlpa(g)
    print
    #print('nmi of vlpa is', nmi(real_label, label2))
    #print('modularity of vlpa is', community.modularity(label2, g))
    #print('nmi of infomap is', nmi(real_label, label))
    #print('modularity of infomap is ', community.modularity(label, g))


def method_adjust():
    # x = [0.0, 0.1]
    g, real_label = inputdata.read_lfr(0.6)
    opt_value = community.modularity(real_label, g)
    mod_a = vlpa.convergence_vlpa(g, gamma=0.5, mod='both')
    mod_b = vlpa.convergence_vlpa(g, gamma=0.9, mod='both')
    mod_c = vlpa.convergence_vlpa(g, gamma=0.5, mod='nothing')
    mod_d = vlpa.convergence_vlpa(g, gamma=0.9, mod='nothing')
    mod_e = vlpa.convergence_vlpa(g, gamma=0.5, mod='normalize')
    mod_f = vlpa.convergence_vlpa(g, gamma=0.9, mod='normalize')


    log_a_values = [np.log10(abs((v - opt_value)/opt_value)) for v in mod_a]
    log_b_values = [np.log10(abs((v - opt_value)/opt_value)) for v in mod_b]
    log_c_values = [np.log10(abs((v - opt_value)/opt_value)) for v in mod_c]
    log_d_values = [np.log10(abs((v - opt_value)/opt_value)) for v in mod_d]
    log_e_values = [np.log10(abs((v - opt_value)/opt_value)) for v in mod_e]
    log_f_values = [np.log10(abs((v - opt_value)/opt_value)) for v in mod_f]

    with open('method_adjust.dat', 'wb') as f:
        pickle.dump(log_a_values, f)
        pickle.dump(log_b_values, f)
        pickle.dump(log_c_values, f)
        pickle.dump(log_d_values, f)
        pickle.dump(log_e_values, f)
        pickle.dump(log_f_values, f)


def pos_adjust():
    g, real_label = inputdata.read_lfr(0.6)
    vecs = vlpa.information_vlpa(g)
    with open('pos_adjust.dat', 'wb') as f:
        pickle.dump(vecs, f)

def pos_plot():
    def num(label,p):
        list_sorted = sorted(label,key=lambda x:label[x],reverse=True)
        num = 0
        value = 0.0
        for i in list_sorted:
            if value < np.sqrt(p):
                value += (label[i])**2
                num += 1
        return num

    with open('pos_adjust.dat', 'r') as f:
        vecs = pickle.load(f)

    degree = []
    pos_num = []
    for k in vecs:
        degree.append(len(vecs[k]))
        pos_num.append(num(vecs[k], 0.90))
        print(len(vecs[k]), num(vecs[k], 0.90))
    plt.figure(1)
    plt.scatter(degree, pos_num)
    plt.savefig('num_shrink.png')
    plt.show()


def pos_change_adjust():
    g, real_label = inputdata.read_lfr(0.6)
    a = vlpa.no_pos_vlpa(g)
    b = vlpa.pos_vlpa(g)
    print(a)
    print(b)
    print(community.modularity(real_label,g))


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

        a = vlpa.vlpa(g)
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
    plt.xlabel('\mu')
    plt.ylabel('NMI')
    plt.savefig('louvian_compare.png')
    pass


def renorm_compare():
    nmi_a = []
    nmi_b = []
    nmi_c = []
    nmi_d = []
    nmi_e = []
    nmi_f = []
    #x = [0.1,0.2]
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9]
    for num in x:
        g, real_label = inputdata.read_lfr(num)

        a = vlpa.vlpa(g)
        b = vlpa.time_vlpa(g,if_renorm='False').labels
        c = vlpa.time_vlpa(g,if_renorm='True').labels
        # f = vlpa.vlpa3(g)
        nmi_a.append(nmi(real_label, a))
        nmi_b.append(nmi(real_label, b))
        nmi_c.append(nmi(real_label, c))
        # nmi_f.append(nmi(real_label, f))
    # plot
    compare_result = [x,nmi_a,nmi_b,nmi_c,]
    with open('renorm_compare.dat', 'wb') as f:
        pickle.dump(compare_result, f)


def renorm_compare_plot():
    with open('renorm_compare.dat', 'r') as f:
        result = pickle.load(f)
    plt.figure(1)
    plt.plot(result[0], result[1], label='vlpa')
    plt.plot(result[0], result[2], label='no_renorm')
    plt.plot(result[0], result[3], label='renorm')
    # plt.plot(x, nmi_f, label='vlpa3')
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
        a = vlpa.vlpa(g)
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
    x = [200, 400, 800, 1200, 1600]
    plus = []
    times = []
    for num in x:
        g, real_label = inputdata.read_lfr(num)
        t1 = time.time()
        result = method(g)
        t2 = time.time()
        print('benchmark modularity is %f'%(community.modularity(real_label, g)))
        plus.append(np.log10(len(g.edges())))
        times.append(np.log10(abs(t2-t1)))
    return plus, times


def set_compare():
    x,y = time_test(vlpa.clustering_infomap)
    plot.fit_plot(x,y)
    plt.savefig('set_compare.png')


def timeit_profile():
    """
    check running time bottleneck
    """
    g, real_label = inputdata.read_lfr(0.6)
    vlpa.louvian_vlpa(g)
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


def convergence_rate_plot(input,opt_value):
    step = []
    mod1 = []
    mod2 = []
    best_value = []
    for i in xrange(len(input[0])):
        step.append(i)
        mod1.append(input[0][i])
        mod2.append(input[1][i])
        best_value.append(opt_value)
    plt.figure()
    plt.plot(step,input[0],label='$\gamma=0.2$')
    plt.plot(step,input[1],label='$\gamma=0.5$')
    plt.plot(step,input[2],label='$\gamma=0.7$')
    plt.plot(step,best_value)
    plt.legend()
    plt.show()
    plt.close()
    pass

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

def louvian_test():
    g, real_label = inputdata.read_lfr(0.6)
    result1 = vlpa.louvian_vlpa_opt(g, gamma=0.2)
    result2 = vlpa.louvian_vlpa_opt(g, gamma=0.5)
    result3 = vlpa.louvian_vlpa_opt(g, gamma=0.7)
    opt_value = community.modularity(real_label, g)

    print(opt_value)
    print('louvian method', vlpa.louvian(g).after_mod)
    print(result1.vmods)

    print(set(real_label.values()))
    draw.draw_group(g, real_label, result2.labels)
    plt.show()
    plt.close()
    convergence_rate_plot({0: result1.mods, 1: result2.mods, 2: result3.mods}, opt_value)
    pass
#cProfile.run("timeit_profile()", filename="result.out")
#louvain_compare()
renorm_compare()
renorm_compare_plot()
