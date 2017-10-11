import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations


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


def draw_network(g, real_label):
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
        node_color.append(float(real_label[node]))


    node_size = 12800 / float(len(g.nodes()))
    pos = nx.spring_layout(g)

    plt.figure()
    nx.draw_networkx_nodes(g, pos=pos, node_color=node_color, node_size=node_size)
    nx.draw_networkx_edges(g, pos=pos, style='dotted', alpha=0.2)
    # nx.draw_networkx_labels(g, pos=pos, labels =label)

