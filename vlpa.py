import networkx as nx
import numpy as np
import heapq


class vlabel(dict):
    # structure of vlabel is like {1:0.2, 2:0.3, 3:0.5}
    def __init__(self):
        # initialization
        self.name = 'vlabel'

    def get_from(self, dic):
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

    def scale(self, a):
        # return a scaled mlabel
        scaled = vlabel()
        for k in self:
            scaled[k] = a * self[k]
        return scaled

    def nlarg(self, n):
        # get first n largest items
        nlarged = vlabel()
        if len(self) <= n:
            nlarged.get_from(self)
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
    # structure of mlabels is like {node1:mlabel1, node2:mlabel2, ...}
    def __init__(self, g):
        self.name = 'vlabels'
        for node in g.nodes():
            self[node] = vlabel()

    def initialization(self, g):
        for node in g.nodes():
            label = vlabel()
            for neigh in g.neighbors(node):
                label[neigh] = 1.0 / g.degree(node)
            self[node] = label

    def print_all(self):
        print(self)

    def __add__(self,other):
        added = vlabels(self.graph)
        for node in self:
            added[node] = self[node] + other[node]
        return added

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
        for step in xrange(15):
            vectors_grad = vlabels(self.graph)
            for node in self.graph.nodes():
                for neigh in self.graph.neighbors(node):
                    vectors_grad[node] = vectors_grad[node] + vectors[neigh]
                vectors_grad[node] = vectors_grad[node].nlarg(self.graph.degree(node)).normalize()
            for node in self.graph.nodes():
                vectors[node] = (vectors[node].scale(0.4) + vectors_grad[node].scale(0.6)).nlarg(self.graph.degree(node)).normalize()

        return vectors.to_labels()
