import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations


def draw_group(g, real_label, label):
    set_label = list(set(real_label.values()))
    num_label = len(set_label)
    g_label = nx.complete_graph(num_label)
    pos_label = nx.fruchterman_reingold_layout(g_label)
    pos = dict()
    for i in xrange(num_label):
        move = pos_label[i]
        nodes = [k for k, v in real_label.iteritems() if v == set_label[i]]
        g_sub = g.subgraph(nodes)
        pos_g_sub = dict([(k, v + 0.8 * num_label * move) for k, v in nx.fruchterman_reingold_layout(g_sub).iteritems()])
        pos = dict(pos, **pos_g_sub)

    node_size = []
    node_color = []
    for n in g.nodes():
        node_color.append(label[n])
        if g.degree(n)!=0:
            node_size.append(25 * g.degree(n))
        else:
            node_size.append(60)

    edge_color = []
    for edge in g.edges():
        if edge[0]==edge[1]:
            edge_color.append(float(label[edge[0]]))
        else:
            label1 = float(label[edge[1]])
            label2 = float(label[edge[0]])
            edge_color.append((label1+label2)/2)

    f = plt.figure()

    nx.draw_networkx_edges(g, pos=pos, alpha=0.3, edge_color=edge_color, width=2, style='solid')
    nx.draw_networkx_nodes(g, pos=pos, alpha=0.5, with_labels=False, node_size=node_size, node_color=node_color)
    f.set_rasterized(True)
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

