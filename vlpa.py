import networkx as nx
import numpy as np
import heapq
from random import choice
import sys
import time
import community
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

    def dot(self, other):
        result = 0.0
        for node in self:
            if node in other:
                result += self[node] * other[node]
        return result

    def norm(self, n=2):
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
        # if there are some key has the same maximum value then randomly
        # choose one
        max_value = max(self.values())
        mained = vlabel()
        keys = [k for k in self if self[k] == max_value]
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
        if len(self) == 0:
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
            self[node] = vlabel({neigh: 1.0 / g.degree(node) for neigh in g.neighbors(node)})

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

    def error(self, other):
        error = 0.0
        for node in self:
            error += self[node].error(other[node])
        return error / len(self)

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

    def normalize(self, n=2):
        return vlabels({node: self[node].normalize(n) for node in self})

    def norm2(self, n=2):
        g_all = [(self[k].norm())**n for k in self]
        g2 = sum(g_all)
        return g2

    def to_labels(self):
        labels = dict()
        for node in self:
            labels[node] = self[node].main().keys()[0]

        symbols = list(set(labels.values()))

        for key in labels:
            labels[key] = symbols.index(labels[key])
        return labels

    def main(self):
        for node in self:
            self[node] = self[node].main()
        return 0


def basic_vlpa(g):
    # initiazaiton

    vecs = vlabels()
    vecs.initialization(g)
    # propagation step
    # n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = g.degree()

    def estimate_change_condition():
        cond_one = abs(vecs.error(vecs_new)) < 0.01
        cond_two = step > 10
        return cond_one & cond_two

    def estimate_stop_condition():
        m_new = community.modularity(vecs_new.to_labels(), g)
        m_old = community.modularity(vecs.to_labels(), g)
        cond_three = abs((m_new - m_old) / m_new) < 0.01
        cond_four = step > 5
        return cond_three & cond_four

    for step in xrange(100):
        vec_all = vlabel()
        for node in g.nodes():
            vec_all = vec_all + vecs[node] * g.degree(node)
        vec_all = vec_all * (- 1.0 / (2 * m))

        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        vecs_all = vlabels()
        for node in g.nodes():
            vecs_all[node] = vec_all * g.degree(node)

        vecs_grad = (vecs_grad + vecs_all).nlarg(pos).normalize(n=2)
        vecs_new = (vecs * 0.4 + vecs_grad * 0.6).nlarg(pos).normalize(n=2)

        if estimate_change_condition():
            break
        vecs = vecs_new

    pos = {}.fromkeys(g.nodes(), 1)
    for step in xrange(10):
        vec_all = vlabel()
        for node in g.nodes():
            vec_all = vec_all + vecs[node] * g.degree(node)
        vec_all = vec_all * (- 1.0 / (2 * m))

        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        vecs_all = vlabels()
        for node in g.nodes():
            vecs_all[node] = vec_all * g.degree(node)

        vecs_grad = (vecs_grad + vecs_all).nlarg(pos).normalize(n=2)
        vecs_new = (vecs * 0.4 + vecs_grad * 0.6).nlarg(pos).normalize(n=2)

        if estimate_stop_condition():
            break
        vecs = vecs_new

    return vecs.to_labels()


def convergence_vlpa(g, gamma=0.5, mod='nothing'):
    # initiazaiton
    modularity = []

    vecs = vlabels()
    vecs.initialization(g)
    # propagation step
    # n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = g.degree()
    if mod == 'nothing':
        def grad(vecs):
            return vecs
    elif mod == 'normalize':
        def grad(vecs):
            return vecs.normalize(n=2)
    elif mod == 'both':
        def grad(vecs):
            return vecs.nlarg(pos).normalize(n=2)
    else:
        raise("Unexist module, the mod can only be in [nothing, normalize, both]")

    def estimate_change_condition():
        cond_one = abs(vecs.error(vecs_new)) < 0.01
        cond_two = step > 10
        return cond_one & cond_two

    def estimate_stop_condition():
        m_new = community.modularity(vecs_new.to_labels(), g)
        m_old = community.modularity(vecs.to_labels(), g)
        cond_three = abs((m_new - m_old) / m_new) < 0.01
        cond_four = step > 5
        return cond_three & cond_four

    t1 = time.time()
    for step in xrange(100):
        vec_all = vlabel()
        for node in g.nodes():
            vec_all = vec_all + vecs[node] * g.degree(node)
        vec_all = vec_all * (- 1.0 / (2 * m))

        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        vecs_all = vlabels()
        for node in g.nodes():
            vecs_all[node] = vec_all * g.degree(node)

        vecs_grad = grad(vecs_grad + vecs_all)
        vecs_new = (vecs * (1 - gamma) + vecs_grad * gamma).nlarg(pos).normalize(n=2)

        # if estimate_change_condition():
        # break
        vecs = vecs_new
        modularity.append(community.modularity(vecs.to_labels(), g))

    pos = {}.fromkeys(g.nodes(), 1)
    for step in xrange(10):
        vec_all = vlabel()
        for node in g.nodes():
            vec_all = vec_all + vecs[node] * g.degree(node)
        vec_all = vec_all * (- 1.0 / (2 * m))

        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        vecs_all = vlabels()
        for node in g.nodes():
            vecs_all[node] = vec_all * g.degree(node)

        vecs_grad = grad(vecs_grad + vecs_all)
        vecs_new = (vecs * (1 - gamma) + vecs_grad * gamma).nlarg(pos).normalize(n=2)

        if estimate_stop_condition():
            break
        vecs = vecs_new
    t2 = time.time()
    print('Time of module of ' + mod + ' is ', t2 - t1)
    return modularity


def fista_vlpa(g):
    def estimate_change_condition():
        cond_one = abs(vecs.error(vecs_new)) < 0.01
        cond_two = step > 10
        return cond_one & cond_two

    def estimate_stop_condition():
        m_new = community.modularity(vecs_new.to_labels(), g)
        m_old = community.modularity(vecs.to_labels(), g)
        cond_three = abs((m_new - m_old) / m_new) < 0.01
        cond_four = step > 5
        return cond_three & cond_four

    # initiazaiton
    modularity = []
    x_vecs_old = vlabels()
    x_vecs_old.initialization(g)
    vecs_new = x_vecs_old.copy()

    # propagation step
    # n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = g.degree()

    t1 = 1.0
    t2 = 1.0
    time1 = time.time()

    for step in xrange(100):
        vecs = x_vecs_old
        vec_all = vlabel()
        for node in g.nodes():
            vec_all = vec_all + vecs[node] * g.degree(node)
        vec_all = vec_all * (- 1.0 / (2 * m))

        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        vecs_all = vlabels()
        for node in g.nodes():
            vecs_all[node] = vec_all * g.degree(node)

        vecs_grad = (vecs_grad + vecs_all).nlarg(pos).normalize(n=2)
        x_vecs = (vecs + vecs_grad * 0.4)

        t2 = 0.5 * (1 + np.sqrt(1 + 4 * t1 * t1))

        vecs_new = (x_vecs + (x_vecs - x_vecs_old) * ((t1 - 1) / t2)).nlarg(pos).normalize(n=2)

        if estimate_change_condition():
            break
        t1 = t2
        x_vecs_old = x_vecs
        modularity.append(community.modularity(vecs.to_labels(), g))

    pos = {}.fromkeys(g.nodes(), 1)
    for step in xrange(10):
        vec_all = vlabel()
        for node in g.nodes():
            vec_all = vec_all + vecs[node] * g.degree(node)
        vec_all = vec_all * (- 1.0 / (2 * m))

        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        vecs_all = vlabels()
        for node in g.nodes():
            vecs_all[node] = vec_all * g.degree(node)

        vecs_grad = (vecs_grad + vecs_all).nlarg(pos).normalize(n=2)
        vecs_new = (vecs * 0.4 + vecs_grad * 0.6).nlarg(pos).normalize(n=2)

        if estimate_stop_condition():
            break
        vecs = vecs_new

    time2 = time.time()
    print('Time of module of fista is ', time2 - time1)
    return modularity


def ada_vlpa(g):
    def estimate_change_condition():
        cond_one = abs(vecs.error(vecs_new)) < 0.01
        cond_two = step > 10
        return cond_one & cond_two

    def estimate_stop_condition():
        m_new = community.modularity(vecs_new.to_labels(), g)
        m_old = community.modularity(vecs.to_labels(), g)
        cond_three = abs((m_new - m_old) / m_new) < 0.01
        cond_four = step > 5
        return cond_three & cond_four

    # initiazaiton
    gamma = 0.5
    eta = 0.
    epsilon = 0.1
    Eg = 1.0
    Edelta = 1.0
    modularity = []
    vecs = vlabels()
    vecs.initialization(g)
    modularity = []

    # propagation step
    # n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = g.degree()

    time1 = time.time()

    for step in xrange(50):
        vec_all = vlabel()
        for node in g.nodes():
            vec_all = vec_all + vecs[node] * g.degree(node)
        vec_all = vec_all * (- 1.0 / (2 * m))

        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        vecs_all = vlabels()
        for node in g.nodes():
            vecs_all[node] = vec_all * g.degree(node)

        vecs_grad = vecs_grad + vecs_all  # calculate the gradient = vecs_grad

        Eg = gamma * Eg + (1 - gamma) * vecs_grad.norm2()
        coef = np.sqrt(Edelta + epsilon) / np.sqrt(Eg + epsilon)
        vecs_delta = vecs_grad * coef
        vecs_new = (vecs + vecs_delta).nlarg(pos).normalize(n=2)
        Edelta = gamma * Edelta + (1 - gamma) * vecs_delta.norm2()

        ##### iteration end #####
        vecs = vecs_new
        modularity.append(community.modularity(vecs.to_labels(), g))

    pos = {}.fromkeys(g.nodes(), 1)
    for step in xrange(10):
        vec_all = vlabel()
        for node in g.nodes():
            vec_all = vec_all + vecs[node] * g.degree(node)
        vec_all = vec_all * (- 1.0 / (2 * m))

        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        vecs_all = vlabels()
        for node in g.nodes():
            vecs_all[node] = vec_all * g.degree(node)

        vecs_grad = (vecs_grad + vecs_all).nlarg(pos).normalize(n=2)
        vecs_new = (vecs * 0.4 + vecs_grad * 0.6).nlarg(pos).normalize(n=2)

        if estimate_stop_condition():
            break
        vecs = vecs_new

    time2 = time.time()
    print('Time of module of ata is ', time2 - time1)
    return modularity


def method(posshrink=False, withshrink=False, gamma=0.5):
    def vlpa_pos_shrink(g):
        # initiazaiton

        vecs = vlabels()
        vecs.initialization(g)
        # propagation step
        n = float(len(g.nodes()))
        m = float(len(g.edges()))
        pos = g.degree()
        k_ave = float(sum(g.degree().values())) / n
        for step in xrange(60):
            if step > 50:
                pos = {}.fromkeys(g.nodes(), 1)

            vec_all = vlabel()
            for node in g.nodes():
                vec_all = vec_all + vecs[node] * g.degree(node)
            vec_all = vec_all * (- 1.0 / (2 * m))

            vecs_grad = vlabels()
            for node in g.nodes():
                vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

            vecs_all = vlabels()
            for node in g.nodes():
                vecs_all[node] = vec_all * g.degree(node)

            vecs_grad = (vecs_grad + vecs_all).nlarg(pos).normalize(n=2)
            vecs = (vecs * 0.4 + vecs_grad * 0.6).nlarg(pos).normalize(n=2)

        return vecs.to_labels()

    def vlpa_no_pos_shrink(g):
        # initiazaiton

        vecs = vlabels()
        vecs.initialization(g)
        # propagation step
        n = float(len(g.nodes()))
        m = float(len(g.edges()))
        pos = g.degree()
        k_ave = float(sum(g.degree().values())) / n
        for step in xrange(60):
            vec_all = vlabel()
            for node in g.nodes():
                vec_all = vec_all + vecs[node] * g.degree(node)
            vec_all = vec_all * (- 1.0 / (2 * m))

            vecs_grad = vlabels()
            for node in g.nodes():
                vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

            vecs_all = vlabels()
            for node in g.nodes():
                vecs_all[node] = vec_all * g.degree(node)

            vecs_grad = (vecs_grad + vecs_all).nlarg(pos).normalize(n=2)
            vecs_new = (vecs * 0.4 + vecs_grad * 0.6).nlarg(pos).normalize(n=2)
            print(vecs.error(vecs_new))
            vecs = vecs_new

        return vecs.to_labels()

    def vlpa_with_shrink(g):
        # initiazaiton

        vecs = vlabels()
        vecs.initialization(g)
        # propagation step
        # n = float(len(g.nodes()))
        m = float(len(g.edges()))
        pos = g.degree()
        for step in xrange(60):

            # if step > 50:
            #     pos = {}.fromkeys(g.nodes(), 1)

            vec_all = vlabel()
            for node in g.nodes():
                vec_all = vec_all + vecs[node] * g.degree(node)
            vec_all = vec_all * (- 1.0 / (2 * m))

            vecs_grad = vlabels()
            for node in g.nodes():
                vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

            vecs_all = vlabels()
            for node in g.nodes():
                vecs_all[node] = vec_all * g.degree(node)

            vecs_grad = (vecs_grad + vecs_all).nlarg(pos).normalize(n=2)
            vecs = (vecs * (1 - gamma) + vecs_grad * gamma).shrink(0.05).nlarg(pos).normalize(n=2)

        return vecs.to_labels()

    def vlpa_without_shrink(g):
        # initiazaiton

        vecs = vlabels()
        vecs.initialization(g)
        # propagation step
        n = float(len(g.nodes()))
        m = float(len(g.edges()))
        pos = g.degree()
        k_ave = float(sum(g.degree().values())) / n
        for step in xrange(60):

            # if step > 50:
            #     pos = {}.fromkeys(g.nodes(), 1)

            vec_all = vlabel()
            for node in g.nodes():
                vec_all = vec_all + vecs[node] * g.degree(node)
            vec_all = vec_all * (- 1.0 / (2 * m))

            vecs_grad = vlabels()
            for node in g.nodes():
                vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

            vecs_all = vlabels()
            for node in g.nodes():
                vecs_all[node] = vec_all * g.degree(node)

            vecs_grad = (vecs_grad + vecs_all).nlarg(pos).normalize(n=2)
            vecs = (vecs * (1 - gamma) + vecs_grad * gamma).nlarg(pos).normalize(n=2)

        return vecs.to_labels()

    if posshrink is True:
        return vlpa_pos_shrink
    elif posshrink is False:
        return vlpa_no_pos_shrink

    # if withshrink==True:
    #     return vlpa_with_shrink
    # elif withshrink==False:
    #     return vlpa_without_shrink
    # else:
    #     raise "must be boolean value"
    # pass


def lpa(g):
    def estimate_stop_cond():
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
        if loop_count >= 15:
            break

    return vecs.to_labels()


def clustering_infomap(G):
    """
    Partition network with the Infomap algorithm.
    Annotates nodes with 'community' id and return number of communities found.
    """
    "transform G to g"
    list = G.nodes()

    def n2i(node):
        return list.index(node)
    g1 = nx.Graph()
    for e in G.edges():
        g1.add_edge(n2i(e[0]), n2i(e[1]))

    infomapWrapper = infomap.Infomap("--two-level")

    print("Building Infomap network from a NetworkX graph...")
    for e in g1.edges_iter():
        infomapWrapper.addLink(*e)

    print("Find communities with Infomap...")
    infomapWrapper.run()

    tree = infomapWrapper.tree

    print("Found %d top modules with codelength: %f" % (tree.numTopModules(), tree.codelength()))

    communities = {}
    for node in tree.leafIter():
        communities[node.originalLeafIndex] = node.moduleIndex()
    real_commuinties = {}
    # transform to original commuinties
    for i in communities:
        real_commuinties[list[i]] = communities[i]

    return real_commuinties


def nonmodel(g):
    return {}.fromkeys(g.nodes(), 1)
