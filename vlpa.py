import networkx as nx
import numpy as np
import heapq

def to_vlabel(dic):
    vec = vlabel()
    for key in dic:
        vec[key] = dic[key]
    return vec

def to_vlabels(dic):
    vecs = vlabels()
    for key in dic:
        vecs[key] = dic[key]
    return vecs

class vlabel(dict):
    # structure of vlabel is like {1:0.2, 2:0.3, 3:0.5}
    def __init__(self):
        # initialization
        self.name = 'vlabel'

    def fromdic(self, dic):
        for k in dic:
            self[k] = dic[k]

    def copy(self):
        # make a copy of a vlabel
        copyed = vlabel()
        for k in self:
            copyed[k] = self[k]
        return copyed

    def __add__(self, other):
        added = self.copy()
        for key in other:
            if key in self:
                added[key] = self[key] + other[key]
            else:
                added[key] = other[key]
        return added

    def __mul__(self, num):
        scaled = vlabel()
        for k in self:
            scaled[k] = num * self[k]
        return scaled

    def scale(self, a):
        # return a scaled vlabel
        scaled = vlabel()
        for k in self:
            scaled[k] = a * self[k]
        return scaled

    def nlarg(self, n):
        # get first n largest items
        nlarged = vlabel()
        if len(self) <= n:
            nlarged.fromdic(self)
        else:
            for key in heapq.nlargest(n, self, key=self.get):
                nlarged[key] = self[key]
        return nlarged

    def main(self):
        # get only the maximam key in the vlabel
        key_max = max(self, key=self.get)
        mained = vlabel()
        mained[key_max] = 1.0
        return mained

    def shrink(self, v):
        """
        delete all the itoms that is smaller than v
        :param v:
        :return:
        """
        shrinked = vlabel()
        for key in self:
            if self[key] > v:
                shrinked[key] = self[key] - v
        return shrinked

    def normalize(self):
        # make the norm of self is 1.0
        normalized = vlabel()
        if len(self) > 0:
            norm = float(sum(self.values()))
            for key in self:
                normalized[key] = float(self[key]) / norm
            return normalized
        else:
            print "the vlabel is empty"


class vlabels(dict):
    # structure of mlabels is like {node1:vlabel1, node2:vlabel2, ...}
    def __init__(self, g):
        self.name = 'vlabels'
        self.graph = g
        self.pos = g.degree()
        for node in g.nodes():
            self[node] = vlabel()

    def initialization(self, g):
        for node in g.nodes():
            label = vlabel()
            num = float(g.degree(node))
            for neigh in g.neighbors(node):
                label[neigh] = 1.0 / num
            self[node] = label

    def print_all(self):
        print(self)

    # operations related to graph

    def __add__(self, other):
        added = vlabels(self.graph)
        for node in self:
            added[node] = self[node] + other[node]
        return added

    def __mul__(self, num):
        muled = vlabels(self.graph)
        for node in self:
            muled[node] = self[node] * num
        return muled

    def nlarg(self):
        nlarged = vlabels(self.graph)
        for node in self:
           nlarged[node] = self[node].nlarg(self.pos[node])
        return nlarged

    def normalize(self):
        normalized = vlabels(self.graph)
        for node in self:
            normalized[node] = self[node].normalize()
        return normalized

    def to_labels(self):
        labels = dict()
        for node in self:
            labels[node] = self[node].main().keys()[0]

        symbols = list(set(labels.values()))

        for key in labels:
            labels[key] = symbols.index(labels[key])
        return labels


class Propragation(object):
    def __init__(self, g):
        self.graph = g

    def run(self):
        # initiazaiton
        vectors = vlabels(self.graph)
        vectors.initialization(self.graph)
        # propagation step
        n = float(len(self.graph.nodes()))
        m = float(len(self.graph.edges()))
        k_ave = float(sum(self.graph.degree().values())) / n
        for step in xrange(60):
            vectors_grad = vlabels(self.graph)
            vec_all = vlabel()
            for node in self.graph.nodes():
                vec_all = vec_all + vectors[node]
                for neigh in self.graph.neighbors(node):
                    vectors_grad[node] = vectors_grad[node] + vectors[neigh]
            vecs_all = vlabels(self.graph)
            for node in self.graph.nodes():
                vecs_all[node] = vec_all * (- k_ave * k_ave / (2 * m))
            vectors_grad = (vectors_grad + vecs_all).nlarg().normalize()
            vectors = (vectors * 0.4 + vectors_grad * 0.6).nlarg().normalize()

        return vectors.to_labels()

    def run2(self):
        # initiazaiton
        vectors = vlabels(self.graph)
        vectors.initialization(self.graph)
        # propagation step
        n = float(len(self.graph.nodes()))
        m = float(len(self.graph.edges()))
        k_ave = float(sum(self.graph.degree().values())) / n
        for step in xrange(60):
            vectors_grad = vlabels(self.graph)
            vec_all = vlabel()
            for node in self.graph.nodes():
                vec_all = vec_all + vectors[node] * self.graph.degree(node)
                for neigh in self.graph.neighbors(node):
                    vectors_grad[node] = vectors_grad[node] + vectors[neigh]
            vecs_all = vlabels(self.graph)
            for node in self.graph.nodes():
                vecs_all[node] = vec_all * (- float(self.graph.degree(node)) /(2 * m))

            vectors_grad = (vectors_grad + vecs_all).nlarg().normalize()

            vectors = (vectors * 0.4 + vectors_grad * 0.6).nlarg().normalize()

        return vectors.to_labels()