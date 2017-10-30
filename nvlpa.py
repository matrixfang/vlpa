import scipy as sp
from scipy.sparse import *
import inputdata
import heapq
import math

g = inputdata.read_network('/Users/fangwenyi/Documents/Data_set/network/network_12/12_network.dat')

class nvlabels(lil_matrix):

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        lil_matrix.__init__(self, arg1, shape, dtype, copy)

    def initialization(self, g):
        def n2i(n):
            return g.nodes().index(n)

        for node in g.nodes():
            for neigh in g.neighbors(node):
                self[n2i(node), n2i(neigh)] = 1.0 / g.degree(node)

    def shrink(self, v):
        for i in xrange(self.shape[0]):
            old = self[i, :]
            positive_index = (old > v).nonzero()[1]
            negative_index = (old < -v).nonzero()[1]
            if len(positive_index) == 0:
                new = lil_matrix(old.shape, dtype='float32')
                max_index = max(old.nonzero()[1], key=lambda x: old[0, x])
                new[0, max_index] = self[i, max_index]
                self[i,:] = new
            else:
                new = lil_matrix(old.shape, dtype='float32')
                for j in positive_index:
                    new[0, j] = self[i, j] - v
                for j in negative_index:
                    new[0, j] = self[i, j] + v
                self[i, :] = new
        return self

    def print_all(self):
        print(self)

    def nlarg(self, pos):
        if self.shape[0] != len(pos):
            raise Exception("index of vlabels and position are not the same")
        for i in xrange(self.shape[0]):
            vlabel_old = self[i, :]
            nlargest_index = heapq.nlargest(pos[i], vlabel_old.nonzero()[1], key=lambda x: vlabel_old[0, x])
            vlabel_new = lil_matrix(vlabel_old.shape)
            for i in nlargest_index:
                vlabel_new[0,i] = vlabel_old[0,i]
            self[i, :] = vlabel_new
        return self

    def main(self):
        for i in xrange(self.shape[0]):
            old = self[i,:]
            if len((old>0.0).nonzero()) == 0:
                raise("all elements are zero")
            else:
                max_index = max(old.nonzero()[1], key=lambda x: old[0, x])
                new = lil_matrix(old.shape, dtype='float32')
                new[0, max_index] = 1.0
                self[i, :] = new
        return self

    def normalize(self, n=2):
        result = nvlabels(self.shape, dtype='float32')
        for i in xrange(self.shape[0]):
            vlabel_old = self[i, :]
            norm = 0.0
            index_nonzero = vlabel_old.nonzero()[1]
            for j in index_nonzero:
                norm += math.pow(vlabel_old[0, j], n)
            norm = math.pow(norm, 0.5)
            result[i, :] = vlabel_old/norm
        return result

def nbasic_vlpa(g):
    # initiazaiton
    def n2i(node):
        if type(node) == type([]):
            return [g.nodes().index(node) for node in node]
        else:
            return g.nodes().index(node)

    def i2n(index):
        if type(index) == type([]):
            return [g.nodes()[i] for i in index]
        else:
            return g.nodes()[index]

    def to_labels():
        labels = dict()
        for node in g.nodes():
            old = vecs[n2i(node), :]
            max_index = max(old.nonzero()[1], key = lambda x: old[0, x])
            labels[node] = old[0, max_index]

        symbols = list(set(labels.values()))

        for key in labels:
            labels[key] = symbols.index(labels[key])
        return labels
    N = len(g.nodes())

    vecs = nvlabels((N,N),dtype='float32')
    vecs.initialization(g)
    # propagation step
    n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = [g.degree(node) for node in g.nodes()]
    for step in xrange(10):
        # if step > 50:
        # pos = {}.fromkeys(g.nodes(), 1)

        degree_matrix = sp.matrix(g.degree().values())
        vec_all = vecs.multiply(degree_matrix.T).sum(axis=0)/(2*m)

        vecs_grad = nvlabels((N,N),dtype='float32')
        for i in xrange(vecs.shape[0]):
            val = vecs[n2i(g.neighbors(i2n(i))), :].sum(axis = 0) #first term
            new = val - g.degree(i2n(i)) * vec_all # second term
            vecs_grad[i, :] = new

        vecs_grad = vecs_grad.nlarg(pos).normalize(n=2)
        vecs = nvlabels(vecs * 0.4 + vecs_grad * 0.6).nlarg(pos).normalize(n=2)
        print(step)
    return to_labels()