import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations


def method_plot(G, d, index_condition):
    # plot partition result according to Graph G, weighted
    # values of edges and some conditions along the algorithm iteration
    """
    :param G: Graph in networkx
    :param d: the dictionary of edges and corresponded weighted values,
    d = {(node1,node2):values}
    :param index_condition : the condition index that we wish it converge to 0
    :return: plot the Graph wi th aboved average weighted edges and histogram of
    weighted edges
    """
    # T = fang.threashold(d.values())
    T = 1
    plt.figure(1)
    plt.subplot(221)
    plt.hist(d.values())
    plt.scatter(T, 0, 50, color ='r')
    plt.subplot(222)
    plt.plot(index_condition)
    plt.subplot(223)
    plt.subplot(224)


def draw_group(g, real_label, label):
    set_label = list(set(real_label.values()))
    num_label = len(set_label)
    g_label = nx.complete_graph(num_label)
    pos_label = nx.spring_layout(g_label)
    pos = dict()
    for i in xrange(num_label):
        move = pos_label[i]
        nodes = [k for k, v in real_label.iteritems() if v == set_label[i]]
        g_sub = g.subgraph(nodes)
        pos_g_sub = dict([(k, v + 0.8 * num_label * move) for k, v in nx.spring_layout(g_sub).iteritems()])
        pos = dict(pos, **pos_g_sub)

    node_color = []
    for node in g.nodes():
        node_color.append(float(label[node]))

    edge_color = []
    for edge in g.edges():
        if label[edge[0]] != label[edge[1]]:
            edge_color.append(0.0)
        else:
            edge_color.append(1.0)

    node_size = 12800/float(len(g.nodes()))

    plt.figure()
    nx.draw_networkx_nodes(g, pos=pos, node_color=node_color, node_size=node_size)
    nx.draw_networkx_edges(g, pos=pos, style='dotted', alpha=0.2)
    #nx.draw_networkx_labels(g, pos=pos, labels =label)


def draw_network(G):
    nx.draw_networkx(G, node_size=1.0, with_labels=False)


def method_core_plot(G, d):
    plt.figure()
    plt.subplot(121)
    plt.hist(d.values())
    plt.scatter(1.0, 0, 50, color='r')
    plt.subplot(122)
    nx.draw_networkx(G, edge_color=d.values(), node_size=1.0, with_labels=False)

