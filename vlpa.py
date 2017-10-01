import networkx as nx
import numpy as np
import heapq
from random import choice

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
        # add other to self by sparsity adding
        added = self.copy()
        for key in other:
            if key in self:
                added[key] = self[key] + other[key]
            else:
                added[key] = other[key]
        return added

    def __sub__(self, other):
        # return self - other by doing sparsity subsection
        added = self.copy()
        for key in other:
            if key in self:
                added[key] = self[key] - other[key]
            else:
                added[key] = -other[key]
        return added

    def __mul__(self, num):
        scaled = vlabel()
        for k in self:
            scaled[k] = num * self[k]
        return scaled

    def norm(self, n=2):
        return np.linalg.norm(self.values(), n)

    def error(self, other):
        error = self - other
        return error.norm / other.norm

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
        # get the key respect to maximum value in the vlabel
        # if there are some key has the same maximum value then randomly choose one
        max_value = max(self.values())
        mained = vlabel()
        keys = [k for k in self if self[k]==max_value]
        key = choice(keys)
        mained = vlabel()
        mained[key] = 1.0
        return mained

    def all_max_keys(self):
        max_value = max(self.values())
        keys = [k for k in self if self[k] == max_value]
        return keys

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

    def normalize(self, n=1):
        # make the norm of self is 1.0
        normalized = vlabel()

        if len(self) > 0:
            norm = self.norm(n)
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

    def __sub__(self, other):
        subed = vlabels(self.graph)
        for node in self:
            subed[node] = self[node] - other[node]
        return subed

    def __mul__(self, num):
        muled = vlabels(self.graph)
        for node in self:
            muled[node] = self[node] * num
        return muled

    def nlarg(self, pos=None):
        if pos == None:
            pos = self.graph.degree()
        else:
            pass

        nlarged = vlabels(self.graph)
        for node in self:
           nlarged[node] = self[node].nlarg(pos[node])
        return nlarged

    def normalize(self,n=1):
        normalized = vlabels(self.graph)
        for node in self:
            normalized[node] = self[node].normalize(n)
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
        pos = self.graph.degree()
        k_ave = float(sum(self.graph.degree().values())) / n
        for step in xrange(100):
            vectors_grad = vlabels(self.graph)
            vec_all = vlabel()
            for node in self.graph.nodes():
                vec_all = vec_all + vectors[node]
                for neigh in self.graph.neighbors(node):
                    vectors_grad[node] = vectors_grad[node] + vectors[neigh]
            vecs_all = vlabels(self.graph)
            for node in self.graph.nodes():
                vecs_all[node] = vec_all * (- k_ave/ (2 * m))

            vectors_grad = (vectors_grad + vecs_all).nlarg(pos).normalize()
            vectors_new = (vectors * 0.4 + vectors_grad * 0.6).nlarg(pos).normalize()

            for node in self.graph.nodes():
                if vectors[node].error(vectors_new[node]) < 0.1:
                    pos[node] = max(pos[node] - 1, 1)
                else:
                    pass
            vectors = vectors_new

        return vectors.to_labels()

    def run2(self):
        # initiazaiton
        vecs = vlabels(self.graph)
        vecs.initialization(self.graph)
        # propagation step
        n = float(len(self.graph.nodes()))
        m = float(len(self.graph.edges()))
        k_ave = float(sum(self.graph.degree().values())) / n
        for step in xrange(60):
            vecs_grad = vlabels(self.graph)
            vec_all = vlabel()
            for node in self.graph.nodes():
                vec_all = vec_all + vecs[node] * self.graph.degree(node)
                for neigh in self.graph.neighbors(node):
                    vecs_grad[node] = vecs_grad[node] + vecs[neigh]
            vecs_all = vlabels(self.graph)
            for node in self.graph.nodes():
                vecs_all[node] = vec_all * (- float(self.graph.degree(node)) /(2 * m))

            vecs_grad = (vecs_grad + vecs_all).nlarg().normalize()

            vecs = (vecs * 0.4 + vecs_grad * 0.6).nlarg().normalize()

        return vecs.to_labels()

    def lpa(self):
        def estimate_stop_cond():
            for node in self.graph.nodes():
                vec = vlabel()
                for neigh in self.graph.neighbors(node):
                    vec = vec + vecs[neigh]
                if vecs[node] in vec.all_max_keys():
                    return False
            return True

        vecs = vlabels(self.graph)
        for node in self.graph.nodes():
            vec = vlabel()
            vec[node] = 1.0
            vecs[node] = vec

        loop_count = 0
        while estimate_stop_cond():
            loop_count += 1
            for node in self.graph.nodes():
                vec = vlabel()
                for neigh in self.graph.neighbors(node):
                    vec = vec + vecs[neigh]
                vecs[node] = vec.main()
            if estimate_stop_cond() is True or loop_count >= 20:
                break

        return vecs.to_labels()


