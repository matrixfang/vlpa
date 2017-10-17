import networkx as nx
import numpy as np
import heapq
from random import choice
import sys
sys.path.append('../infomap/examples/python/infomap')
import infomap


class vlabel(dict):
    # structure of vlabel is like {1:0.2, 2:0.3, 3:0.5}
    def __init__(self, *args, **kwargs):
        # initialization
        dict.__init__(self, *args, **kwargs)
        self.name = 'vlabel'

    def copy(self):
        # make a copy of a vlabel
        result = vlabel()
        for k in self:
            result[k] = self[k]
        return result

    def __add__(self, other):
        # add other to self by sparsity adding
        result = self.copy()
        for key in other:
            if key in self:
                result[key] = self[key] + other[key]
            else:
                result[key] = other[key]
        return result

    def __sub__(self, other):
        # return self - other by doing sparsity subsection
        result = self.copy()
        for key in other:
            if key in self:
                result[key] = self[key] - other[key]
            else:
                result[key] = - other[key]
        return result

    def __mul__(self, num):
        scaled = vlabel()
        for k in self:
            scaled[k] = num * self[k]
        return scaled

    def norm(self, n=1):
        return np.linalg.norm(self.values(), n)

    def error(self, other):
        error = self - other
        return error.norm() / other.norm()

    def nlarg(self, n):
        # get first n largest items
        nlarged = vlabel()
        if len(self) <= n:
            return self.copy()
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
            elif self[key] < -v:
                shrinked[key] = self[key] + v
            else:
                pass
        return shrinked

    def normalize(self, n=1):
        # make the norm of self is 1.0
        # n is the dimension of norm
        # n = 1 means |x_1| + ... + |x_d| = 1
        # n = 2 means x_1^2 + ... + x_d^2 = 1
        result = vlabel()
        if len(self)== 0:
            raise Exception("the vlabel is empty, and can't be normalized")

        norm = self.norm(n)
        for key in self:
            result[key] = float(self[key]) / norm
        return result


class vlabels(dict):
    # structure of mlabels is like {node1:vlabel1, node2:vlabel2, ...}
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.name = 'vlabels'

    def initialization(self, g):
        for node in g.nodes():
            self[node] = vlabel({neigh: 1.0/g.degree(node) for neigh in g.neighbors(node)})

    def print_all(self):
        print(self)

    # operations related to graph

    def __add__(self, other):
        if set(self.keys()) != set(other.keys()):
            raise Exception("index of vlabels are not the same")
        return vlabels({node: self[node] + other[node] for node in self})

    def __sub__(self, other):
        if set(self.keys()) != set(other.keys()):
            raise Exception("index of vlabels are not the same")
        return vlabels({node: self[node] - other[node] for node in self})

    def __mul__(self, num):
        return vlabels({node: self[node] * num for node in self})

    def sum(self):
        result = vlabel()
        for node in self:
            result = result + self[node]
        return result

    def shrink(self, v):
        return vlabels({node: self[node].shrink(v) for node in self})

    def nlarg(self, pos):
        if set(self.keys()) != set(pos.keys()):
            raise Exception("index of vlabels and position are not the same")
        return vlabels({node: self[node].nlarg(pos[node]) for node in self})

    def normalize(self,n=1):
        return vlabels({node: self[node].normalize(n) for node in self})

    def to_labels(self):
        labels = dict()
        for node in self:
            labels[node] = self[node].main().keys()[0]

        symbols = list(set(labels.values()))

        for key in labels:
            labels[key] = symbols.index(labels[key])
        return labels


def vlpa(g):
    # initiazaiton
    vecs = vlabels()
    vecs.initialization(g)
    # propagation step
    n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = g.degree()
    k_ave = float(sum(g.degree().values())) / n
    for step in xrange(15):
        vec_all = vecs.sum()
        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        vecs_all = vlabels()
        for node in g.nodes():
            vecs_all[node] = vec_all * (-k_ave * k_ave / (2 * m))

        vecs_grad = (vecs_grad + vecs_all).nlarg(pos).normalize()
        vecs = (vecs * 0.4 + vecs_grad * 0.6).nlarg(pos).normalize()

    return vecs.to_labels()


def vlpa2(g):
    # initiazaiton
    vecs = vlabels()
    vecs.initialization(g)
    # propagation step
    n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = g.degree()
    k_ave = float(sum(g.degree().values())) / n
    for step in xrange(15):
        vec_all = vecs.sum()
        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        vecs_all = vlabels()
        for node in g.nodes():
            vecs_all[node] = vec_all * (-k_ave * k_ave / (2 * m))

        vecs_grad = (vecs_grad + vecs_all).nlarg(pos).normalize()
        vecs_new = (vecs * 0.4 + vecs_grad * 0.6).nlarg(pos).normalize()

        for node in g.nodes():
            if vecs[node].error(vecs_new[node]) < 0.1:
                pos[node] = max(pos[node] - 1, 1)
            else:
                pass
        vecs = vecs_new

    return vecs.to_labels()


def vlpa3(g):
    # initiazaiton
    vecs = vlabels()
    vecs.initialization(g)
    # propagation step
    n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = g.degree()
    k_ave = float(sum(g.degree().values())) / n
    for step in xrange(15):
        vec_all = vecs.sum()
        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        vecs_all = vlabels()
        for node in g.nodes():
            vecs_all[node] = vec_all * (-k_ave * k_ave / (2 * m))

        vecs_grad = (vecs_grad + vecs_all).nlarg(pos).normalize()
        vecs = (vecs * 0.4 + vecs_grad * 0.6).shrink(0.05).nlarg(pos).normalize()

    return vecs.to_labels()


def lpa(g):
    def estimate_stop_cond():
        global node
        global vec

        for node in g.nodes():
            vec = vlabel()
            for neigh in g.neighbors(node):
                vec = vec + vecs[neigh]
            if vecs[node] in vec.all_max_keys():
                return False
        return True

    vecs = vlabels()
    for node in g.nodes():
        vec = vlabel()
        vec[node] = 1.0
        vecs[node] = vec

    loop_count = 0
    while estimate_stop_cond():
        loop_count += 1
        for node in g.nodes():
            vec = vlabel()
            for neigh in g.neighbors(node):
                vec = vec + vecs[neigh]
            vecs[node] = vec.main()
        if estimate_stop_cond() is True or loop_count >= 15:
            break

    return vecs.to_labels()


def clusting_infomap(G):
    """
    	Partition network with the Infomap algorithm.
    	Annotates nodes with 'community' id and return number of communities found.
    	"""

    infomapWrapper = infomap.Infomap("--two-level")

    print("Building Infomap network from a NetworkX graph...")
    for e in G.edges_iter():
        infomapWrapper.addLink(*e)

    print("Find communities with Infomap...")
    infomapWrapper.run();

    tree = infomapWrapper.tree

    print("Found %d top modules with codelength: %f" % (tree.numTopModules(), tree.codelength()))

    communities = {}
    for node in tree.leafIter():
        communities[node.originalLeafIndex] = node.moduleIndex()

    return communities
