import scipy as sp
from scipy.sparse import *
import inputdata
import heapq
import math

g = inputdata.read_network('/Users/fangwenyi/Documents/Data_set/network/network_12/12_network.dat')


class nvlabels(dok_matrix):
    def __init__(self, N):
        dok_matrix.__init__(self, (N, N), dtype='float32')

    def initialization(self, g):
        def n2i(n):
            return g.nodes().index(n)

        for node in g.nodes():
            for neigh in g.neighbors(node):
                self[n2i(node), n2i(neigh)] = 1.0 / g.degree(node)

    def print_all(self):
        print(self)

    def nlarg(self, pos):
        if self.shape[0] != len(pos):
            raise Exception("index of vlabels and position are not the same")
        for i in xrange(self.shape[0]):
            vlabel_old = self[i, :]
            nlargest_index = heapq.nlargest(pos[i], vlabel_old.nonzero()[1], key=lambda x: vlabel_old[0, x])
            vlabel_new = dok_matrix(vlabel_old.shape)
            for i in nlargest_index:
                vlabel_new[0,i] = vlabel_old[0,i]
            self[i, :] = vlabel_new

    def main(self):
        pos = [1 for i in xrange(self.shape[0])]
        self.nlarg(pos)

    def normalize(self, n=2):
        result = dok_matrix(self.shape, dtype='float32')
        for i in xrange(self.shape[0]):
            vlabel_old = self[i, :]
            norm = 0.0
            index_nonzero = vlabel_old.nonzero()[1]
            print(index_nonzero)
            for j in index_nonzero:
                norm += math.pow(vlabel_old[i,j], n)
            result[i, :] = vlabel_old/norm













v = nvlabels(30)
v.initialization(g)
v.normalize()
v.print_all()