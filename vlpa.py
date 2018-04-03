import networkx as nx
import numpy as np
import heapq
from random import choice
import random
import sys
import time
import matplotlib.pyplot as plt
import community
sys.path.append('../infomap/examples/python/infomap')
import infomap
import copy
import inputdata
from collections import namedtuple


class vlabel(dict):
    # structure of vlabel is like {1:0.2, 2:0.3, 3:0.5}
    def __init__(self, *args, **kwargs):
        # initialization
        dict.__init__(self, *args, **kwargs)
        self.name = 'vlabel'

    def copy(self):
        # make a copy of a vlabel
        return copy.copy(self)

    def __add__(self, other):
        # add other to self by sparsity adding
        result = self.copy()
        for key in other:
            if key in self:
                result[key] = self[key] + other[key]
            else:
                result[key] = other[key]
        return result

    def __iadd__(self, other):
        for key in other:
            if key in self:
                self[key]+=other[key]
            else:
                self[key] = other[key]
        return self

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

    def __imul__(self, num):
        for k in self:
            self[k] = num * self[k]
        return self

    def dot(self, other):
        """
        inner product of two vectors
        :param other: self, other are both vlabels
        :return: inner product
        """
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

    def nonnegative(self):
        new_vector = vlabel()
        for key in self:
            if self[key]>0:
                new_vector[key] = self[key]
        return new_vector

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

    def ifclose(self,other,argument='label'):
        switcher = {'inner':'_inner_product_close','label':'_main_label_close'}
        method_name = switcher.get(argument,'_main_label_close')
        method = getattr(self, method_name)
        return method(other)

    def _inner_product_close(self,other):
        """
        decide if two vectors are close enough by inner product
        :param other:self, other
        :return: True or False
        """
        if self.dot(other)>=0.5:
            return True
        else:
            return False

    def _main_label_close(self,other):
        """
        decide if two vectors are close enough by main label
        :param other:
        :return:
        """
        l1 = max(self, key = lambda x: self[x])
        l2 = max(other,key = lambda x: other[x])
        if l1==l2:
            return True
        else:
            return False

    def saddl(self,other,scal=1.0):
        """
        add a long vector label to self, only care about the keys in self

        default scal =1.0

        return = self + other * scal
        """
        v = vlabel()
        for key in self:
            v[key] = self[key] + other[key] * scal
        return v

    def contract(self,other, scal=1.0):
        v = vlabel()
        for key in self:
            value = self[key] + other[key] * scal
            if value>=0:
                v[key] = value
        return v
        pass

    def close2label(self,gamma):
        keys = self.all_max_keys()
        v = vlabel()
        for key in self:
            if key in keys:
                v[key] = max(self[key] + gamma, 1.0)
            elif self[key]>=gamma:
                v[key] = self[key] - gamma

        return v.normalize(n=2)

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

    def initialization2(self, g):
        for node in g.nodes():
            self[node] = vlabel({node:1.0})

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

    def copy(self):
        vs = vlabels()
        for i in self:
            vs[i] = self[i].copy()
        return vs

    def error(self, other):
        error = 0.0
        for node in self:
            error += self[node].error(other[node])
        return error / len(self)

    def sum(self):
        result = vlabel()
        for node in self:
            result += self[node]
        return result

    def shrink(self, v):
        return vlabels({node: self[node].shrink(v) for node in self})

    def close2labels(self,gamma):
        for node in self:
            self[node].close2label(gamma)
        return self

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


def vmod(vecs, g):
    m = len(g.edges())
    vec_all = vlabel()
    for node in g.nodes():
        vec_all += vecs[node] * (g.degree(node))
    vec_all *= -1.0 / (2 * m)

    vecs_grad = vlabels()
    for node in g.nodes():
        vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

    vm = 0.0
    for node in g.nodes():
        vm+= vecs_grad[node].dot(vecs[node]) + vec_all.dot(vecs[node] * g.degree(node))

    return vm/(2*m)


def rms(x):
    """
    root mean squared function

    """
    return np.sqrt(x + 0.0001)


algorithm_output = namedtuple('algorithm_output',['algorithm','time','labels','vmods','mods','mod'])


def algorithm_result(algorithm, t2, labels, vmods, mods, modularity):
    """
    out put result
    """

    result = algorithm_output(algorithm=algorithm, time=t2, labels=labels,
                              vmods=vmods, mods=mods, mod=modularity)

    """
    print result
    """

    print(algorithm, 'Time %f' % t2, 'modularity is %f' % modularity)

    return result


def vlpa(g):
    # initiazaiton
    t1 = time.time()
    vecs = vlabels()
    vecs.initialization(g)
    mods = []
    vmods = []
    # propagation step
    n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = g.degree()
    # propagation
    for step in xrange(80+10):
        if step%10 in [0,1,2,3,4,5,6]:
            pos = g.degree()
        else:
            pos = {}.fromkeys(g.nodes(), 1)

        vec_all = vlabel()
        for node in g.nodes():
            vec_all += vecs[node] * g.degree(node)
        vec_all *= (- 1.0 / (2 * m))
        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        for node in g.nodes():
            vecs_grad[node] = vecs_grad[node].saddl(vec_all * g.degree(node))


        vecs_grad = vecs_grad.nlarg(pos).normalize()
        vecs = (vecs * 0.5 + vecs_grad * 0.5).nlarg(pos).normalize()
        mods.append(community.modularity(vecs.to_labels(), g))
        vmods.append(vmod(vecs, g))

    ## algorithm output
    algorithm = 'vlpa'
    modularity = mods[len(mods)-1]
    labels = vecs.to_labels()
    t2 = time.time() - t1
    ## get algorithm result
    result = algorithm_result(algorithm, t2, labels, vmods, mods, modularity)

    return result


def vlpa_theoretical_best(g):
    # initiazaiton
    t1 = time.time()
    vecs = vlabels()
    vecs.initialization(g)
    mods = []
    vmods = []
    # propagation step
    n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = {}.fromkeys(g.nodes(),4)
    # propagation
    for step in xrange(80+10):
        if step==80:
            pos={}.fromkeys(g.nodes(),1)
        elif step%20==10:
            pos = {}.fromkeys(g.nodes(),1)
        elif step%20==0:
            pos = g.degree()
        else:
            pass

        vec_all = vlabel()
        for node in g.nodes():
            vec_all += vecs[node] * g.degree(node)
        vec_all *= (- 1.0 / (2 * m))
        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        for node in g.nodes():
            vecs[node] = (vecs_grad[node]+vecs[node]).saddl(vec_all,scal=g.degree(node))

        vecs = vecs.nlarg(pos).normalize()
        mods.append(community.modularity(vecs.to_labels(), g))
        vmods.append(vmod(vecs, g))

    ## algorithm output
    algorithm = 'vlpa'
    modularity = mods[len(mods)-1]
    labels = vecs.to_labels()
    t2 = time.time() - t1
    ## get algorithm result
    result = algorithm_result(algorithm, t2, labels, vmods, mods, modularity)

    return result


def fixed_pos_vlpa(g,k=5):
    # initiazaiton
    t1 = time.time()
    vecs = vlabels()
    vecs.initialization2(g)
    mods = []
    vmods = []
    # propagation step
    n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = {}.fromkeys(g.nodes(),k)
    # propagation
    for step in xrange(80+10):
        if step%10 in [0,1,2,3,4,5,6]:
            pos = {}.fromkeys(g.nodes(), k)
        else:
            pos = {}.fromkeys(g.nodes(), 1)

        vec_all = vlabel()
        for node in g.nodes():
            vec_all += vecs[node] * g.degree(node)
        vec_all *= (- 1.0 / (2 * m))
        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        for node in g.nodes():
            vecs_grad[node] = vecs_grad[node].saddl(vec_all * g.degree(node))

        vecs_grad = vecs_grad.nlarg(pos).normalize()
        vecs = (vecs * 0.4 + vecs_grad * 0.6).nlarg(pos).normalize()
        mods.append(community.modularity(vecs.to_labels(), g))
        vmods.append(vmod(vecs, g))

    ## algorithm output
    algorithm = 'fixed_pos_vlpa ' + str(k)
    modularity = mods[len(mods)-1]
    labels = vecs.to_labels()
    t2 = time.time() - t1
    ## get algorithm result
    result = algorithm_result(algorithm, t2, labels, vmods, mods, modularity)

    return result


def final_vlpa(g, gamma=0.5, mod='nothing', pos_shrink='False'):
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

    if pos_shrink=='True':
        def pos_change():
            def num(label, p):
                list_sorted = sorted(label, key=lambda x: label[x], reverse=True)
                num = 0
                value = 0.0
                for i in list_sorted:
                    if value < np.sqrt(p):
                        value += (label[i]) ** 2
                        num += 1
                return num

            for k in pos:
                pos[k] = num(vecs[k], 0.9)
    else:
        def pos_change():
            pass

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

        #if estimate_change_condition():
            #break
        vecs = vecs_new
        if step<20:
            pos_change()
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

    algorithm = 'vlpa'
    modularity = mods[len(mods) - 1]
    t2 = time.time()

    """
            output result

    """

    result = algorithm_output(algorithm=algorithm, time=t2 - t1, labels=vecs.to_labels(),
                              vmods=vmods, mods=mods, mod=modularity)

    """
    print result
    """

    print(algorithm, 'Time %f' % (t2 - t1), 'modularity is %f' % modularity)
    return vecs.to_labels()


def convergence_vlpa(g, gamma=1.0, mod='nothing', pos_shrink='True'):
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

    if pos_shrink=='True':
        def pos_change():
            def num(label, p):
                list_sorted = sorted(label, key=lambda x: label[x], reverse=True)
                num = 0
                value = 0.0
                for i in list_sorted:
                    if value < np.sqrt(p):
                        value += (label[i]) ** 2
                        num += 1
                return num

            for k in pos:
                pos[k] = num(vecs[k], 0.95)
    else:
        def pos_change():
            pass

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

        vecs_grad = grad(vecs_all + vecs_grad)
        vecs_new = (vecs_grad * gamma + vecs * (1 - gamma)).nlarg(pos).normalize(n=2)

        # if estimate_change_condition():
        # break
        vecs = vecs_new
        pos_change()
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
    print(mod+' '+str(gamma), 'Time %f'%(t2 - t1), 'modularity before lpa is %f'%(modularity[len(modularity)-1]),'modularity after lpa is %f'%(community.modularity(vecs.to_labels(),g)))
    return modularity


def time_vlpa(g, gamma=0.6, mod='nothing', pos_shrink='Fasle', if_renorm ='False'):
    # try to get the best time complixity
    # initiazaiton
    mods = []
    vmods = []
    vecs = vlabels()
    vecs.initialization(g)
    # propagation step
    # n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = g.degree()
    if if_renorm ==False:
        def renormalize():
            pass
    else:
        def renormalize():
            vecs.close2labels(0.02)

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

    if pos_shrink=='True':
        def pos_change():
            def num(label, p):
                list_sorted = sorted(label, key=lambda x: label[x], reverse=True)
                num = 0
                value = 0.0
                for i in list_sorted:
                    if value < np.sqrt(p):
                        value += (label[i]) ** 2
                        num += 1
                return num

            for k in pos:
                pos[k] = num(vecs[k], 0.9)
    else:
        def pos_change():
            pass

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

    def change_vec_all(vec_all, p):
        list_sorted = sorted(vec_all, key=lambda x: vec_all[x], reverse=True)
        num = 0
        value = 0.0
        new_vec_all = vlabel()
        for i in list_sorted:
            if value < vec_all.norm() * np.sqrt(p):
                value += (vec_all[i]) ** 2
                new_vec_all[i] = vec_all[i]
        return new_vec_all

    t1 = time.time()
    for step in xrange(100):
        vec_all = vlabel()
        for node in g.nodes():
            vec_all += vecs[node] * g.degree(node)
        vec_all *= (- 1.0 / (2 * m))
        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        vecs_all = vlabels()
        for node in g.nodes():
            vecs_all[node] = vec_all * g.degree(node)

        vecs_grad = grad(vecs_all + vecs_grad)
        vecs_new = (vecs_grad * gamma + vecs * (1 - gamma)).nlarg(pos).normalize(n=2)

        # if estimate_change_condition():
        # break
        vecs = vecs_new
        renormalize()
        pos_change()
        mods.append(community.modularity(vecs.to_labels(), g))
        vmods.append(vmod(g,vecs))
    before_mod = community.modularity(vecs.to_labels(),g)

    pos = {}.fromkeys(g.nodes(), 1)
    for step in xrange(10):
        vec_all = vlabel()
        for node in g.nodes():
            vec_all = vec_all + vecs[node] * g.degree(node)
        vec_all *= (- 1.0 / (2 * m))

        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        vecs_all = vlabels()
        for node in g.nodes():
            vecs_all[node] = vec_all * g.degree(node)

        vecs_grad = grad(vecs_all + vecs_grad)
        vecs_new = (vecs_grad * gamma + vecs * (1 - gamma)).nlarg(pos).normalize(n=2)

        if estimate_stop_condition():
            break
        vecs = vecs_new
    after_mod = community.modularity(vecs.to_labels(), g)
    t2 = time.time()

    """
        output result

    """

    result = algorithm_output(algorithm="time_vlpa", time=t2 - t1, labels=vecs.to_labels(),
                              vmods=vmods, mods=mods, before_mod=before_mod, after_mod=after_mod)

    """
    print result
    """
    print(mod+str(gamma), 'Time %f' % (t2 - t1), 'modularity before is %f' % (before_mod),
          "modularity after is %f" % (after_mod))

    return result


def louvain_vlpa(g, gamma=0.6):
    # try to get the best time complixity
    # initiazaiton
    t1 = time.time()
    mods = []
    vmods = []
    vecs = vlabels()
    vecs_best = vlabels()
    vmod_best = 0.0
    vecs.initialization(g)
    # propagation step
    # n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = g.degree()
    cgrad = vlabel()
    for node in g.nodes():
        cgrad += vecs[node] * g.degree(node)
    cgrad *= (- 1.0 / (2 * m))
    diff_cgrad = vlabel()
    shuffled_nodes = g.nodes()

    for step in xrange(90):
        random.shuffle(shuffled_nodes) # shuffle the node list
        # update one node vector label
        if step==80:
            pos = {}.fromkeys(g.nodes(),1)

        for node in shuffled_nodes:
            pgrad = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()
            cgrad += diff_cgrad
            grad = cgrad * g.degree(node) + pgrad
            grad = grad.nlarg(pos[node]).normalize()
            update = (grad * gamma + vecs[node]*(1-gamma)).nlarg(pos[node]).normalize(n=2)
            diff_cgrad = (update - vecs[node]) * (- float(g.degree(node))/(2 * m))
            vecs[node] = update

        vmods.append(vmod(vecs, g))
        mods.append(community.modularity(vecs.to_labels(), g))

    ## algorithm output
    algorithm = 'louvain_vlpa'
    modularity = mods[len(mods) - 1]
    labels = vecs.to_labels()
    t2 = time.time() - t1
    ## get algorithm result
    result = algorithm_result(algorithm, t2, labels, vmods, mods, modularity)

    return result


def fixed_pos_louvain_vlpa(g, k=5, gamma=0.6):
    # try to get the best time complixity
    # initiazaiton
    t1 = time.time()
    mods = []
    vmods = []
    vecs = vlabels()
    vecs_best = vlabels()
    vmod_best = 0.0
    vecs.initialization(g)
    # propagation step
    # n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = {}.fromkeys(g.nodes(),k)
    cgrad = vlabel()
    for node in g.nodes():
        cgrad += vecs[node] * g.degree(node)
    cgrad *= (- 1.0 / (2 * m))
    diff_cgrad = vlabel()
    shuffled_nodes = g.nodes()

    for step in xrange(80):
        random.shuffle(shuffled_nodes) # shuffle the node list
        # update one node vector label
        if step%10 in [0,1,2,3,4,5,6]:
            pos = {}.fromkeys(g.nodes(), k)
        else:
            pos = {}.fromkeys(g.nodes(), 1)


        for node in shuffled_nodes:
            pgrad = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()
            cgrad += diff_cgrad
            grad = pgrad.saddl(cgrad, g.degree(node))
            update = (grad * gamma + vecs[node]*(1-gamma)).nlarg(pos[node]).normalize(n=2)

            diff_cgrad = (update - vecs[node]) * (- float(g.degree(node))/(2 * m))
            vecs[node] = update

        vmods.append(vmod(vecs, g))
        mods.append(community.modularity(vecs.to_labels(), g))

    ## algorithm output
    algorithm = 'fixed_pos_louvain_vlpa'
    modularity = mods[len(mods) - 1]
    labels = vecs.to_labels()
    t2 = time.time() - t1
    ## get algorithm result
    result = algorithm_result(algorithm, t2, labels, vmods, mods, modularity)

    return result


def fast_louvain_vlpa(g, k=5, gamma=0.6):
    # try to get the best time complixity
    # initiazaiton

    t1 = time.time()
    mods = []
    vmods = []
    vecs = vlabels()
    vecs_best = vlabels()
    vmod_best = 0.0
    vecs.initialization(g)
    # propagation step
    # n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = {}.fromkeys(g.nodes(),k)
    cgrad = vlabel()
    for node in g.nodes():
        cgrad += vecs[node] * g.degree(node)
    cgrad *= (- 1.0 / (2 * m))
    diff_cgrad = vlabel()
    prune = set(g.nodes())
    shuffled_nodes = g.nodes()
    random.shuffle(shuffled_nodes)

    for step in xrange(90):
        if step%10 in [6,7,8,9]:
            pos = {}.fromkeys(g.nodes(), 1)

            random.shuffle(shuffled_nodes)
            for node in shuffled_nodes:
                prune -= set([node])
                pgrad = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()
                cgrad += diff_cgrad
                grad = pgrad.saddl(cgrad, g.degree(node))
                update = (grad * gamma + vecs[node] * (1 - gamma)).nlarg(pos[node]).normalize(n=2)
                diff_cgrad = (update - vecs[node]) * (- float(g.degree(node)) / (2 * m))
                vecs[node] = update
                for neigh in g.neighbors(node):
                    if vecs[neigh].dot(vecs[node]) < 0.1:
                        prune |= set([neigh])
        else:
            pos = {}.fromkeys(g.nodes(), k)

            for node in shuffled_nodes:
                pgrad = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()
                cgrad += diff_cgrad
                grad = pgrad.saddl(cgrad, g.degree(node))
                update = (grad * gamma + vecs[node] * (1 - gamma)).nlarg(pos[node]).normalize(n=2)
                diff_cgrad = (update - vecs[node]) * (- float(g.degree(node)) / (2 * m))
                vecs[node] = update
        vmods.append(vmod(vecs, g))
        mods.append(community.modularity(vecs.to_labels(), g))

    ## algorithm output
    algorithm = 'fast_louvain_vlpa'
    modularity = mods[len(mods) - 1]
    labels = vecs.to_labels()
    t2 = time.time() - t1
    ## get algorithm result
    result = algorithm_result(algorithm, t2, labels, vmods, mods, modularity)



    return result


def information_vlpa(g):
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

        if estimate_change_condition():
            break
        vecs = vecs_new



    return vecs


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

    t1 = time.time()
    mods = []
    vmods = []

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
        mods.append(community.modularity(vecs.to_labels(), g))
        vmods.append(vmod(vecs, g))
        if loop_count >= 15:
            break

    ## algorithm output
    algorithm = 'lpa'
    modularity = mods[len(mods) - 1]
    labels = vecs.to_labels()
    t2 = time.time() - t1
    ## get algorithm result
    result = algorithm_result(algorithm, t2, labels, vmods, mods, modularity)

    return vecs.to_labels()


def clustering_infomap(G):
    """
    Partition network with the Infomap algorithm.
    Annotates nodes with 'community' id and return number of communities found.
    """
    t1 = time.time()
    "transform G to g"
    list = G.nodes()


    def n2i(node):
        return list.index(node)
    g1 = nx.Graph()
    for e in G.edges():
        g1.add_edge(n2i(e[0]), n2i(e[1]))

    infomapWrapper = infomap.Infomap("--two-level")

    #print("Building Infomap network from a NetworkX graph...")
    for e in g1.edges_iter():
        infomapWrapper.addLink(*e)

    #print("Find communities with Infomap...")
    infomapWrapper.run()

    tree = infomapWrapper.tree

    #print("Found %d top modules with codelength: %f" % (tree.numTopModules(), tree.codelength()))

    communities = {}
    for node in tree.leafIter():
        communities[node.originalLeafIndex] = node.moduleIndex()
    labels = {}
    # transform to original commuinties
    for i in communities:
        labels[list[i]] = communities[i]

    ## algorithm output
    algorithm = 'infomap'
    vmods = None
    mods =None
    modularity = community.modularity(labels, G)
    t2 = time.time() - t1
    ## get algorithm result
    result = algorithm_result(algorithm, t2, labels, vmods, mods, modularity)

    return result


def louvain(g):
    t1 = time.time()
    labels = community.best_partition(g)
    t2 = time.time()
    mod = community.modularity(labels,g)

    ## algorithm output
    mods = None
    vmods = None
    algorithm = 'louvain'
    modularity = mod
    t2 = time.time() - t1
    ## get algorithm result
    result = algorithm_result(algorithm, t2, labels, vmods, mods, modularity)

    return result


def final_vlpa(g):
    # try to get the best time complixity
    # initiazaiton
    t1 = time.time()
    k = 5
    gamma = 0.9
    mods = []
    vmods = []
    vecs = vlabels()
    vecs_best = vlabels()
    vmod_best = 0.0
    vecs.initialization(g)
    # propagation step
    # n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = {}.fromkeys(g.nodes(),k)
    cgrad = vlabel()
    for node in g.nodes():
        cgrad += vecs[node] * g.degree(node)
    cgrad *= (- 1.0 / (2 * m))
    diff_cgrad = vlabel()
    shuffled_nodes = g.nodes()



    for step in xrange(6):
        random.shuffle(shuffled_nodes) # shuffle the node list
        # update one node vector label
        for node in shuffled_nodes:
            pgrad = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()
            cgrad += diff_cgrad
            grad = pgrad.saddl(cgrad, g.degree(node))
            update = (grad * gamma + vecs[node]*(1-gamma)).nlarg(pos[node]).normalize(n=2)
            diff_cgrad = (update - vecs[node]) * (- float(g.degree(node))/(2 * m))
            vecs[node] = update

        vmods.append(vmod(vecs, g))
        mods.append(community.modularity(vecs.to_labels(), g))

    for step in xrange(70+10):
        if step==80:
            pos={}.fromkeys(g.nodes(),1)
        elif step%20==10:
            pos = {}.fromkeys(g.nodes(),1)
        elif step%20==0:
            pos = g.degree()
        else:
            pass

        vec_all = vlabel()
        for node in g.nodes():
            vec_all += vecs[node] * g.degree(node)
        vec_all *= (- 1.0 / (2 * m))
        vecs_grad = vlabels()
        for node in g.nodes():
            vecs_grad[node] = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()

        for node in g.nodes():
            vecs_grad[node] = vecs_grad[node].saddl(vec_all * g.degree(node))


        vecs_grad = vecs_grad.nlarg(pos).normalize()
        vecs = (vecs * 0.4 + vecs_grad * 0.6).nlarg(pos).normalize()
        mods.append(community.modularity(vecs.to_labels(), g))
        vmods.append(vmod(vecs, g))

    ## algorithm output
    algorithm = 'final_vlpa'
    modularity = mods[len(mods) - 1]
    labels = vecs.to_labels()
    t2 = time.time() - t1
    ## get algorithm result
    result = algorithm_result(algorithm, t2, labels, vmods, mods, modularity)

    return result


def graph_agg(labels,g):
    g_agg = nx.Graph()
    for e in g.edges():
        l1 = labels[e[0]]
        l2 = labels[e[1]]
        g_agg.add_edge(l1,l2)
        g_agg[l1][l2]['weight'] = 0
    for e in g.edges():
        l1 = labels[e[0]]
        l2 = labels[e[1]]
        g_agg[l1][l2]['weight'] += 1
    return g_agg


def agg_louvain_vlpa(g, k=5, gamma=0.6):
    # try to get the best time complixity
    # aggregrate only once
    # initiazaiton
    result = fixed_pos_louvain_vlpa(g,k,gamma)
    old_labels = result.labels
    g_new = graph_agg(old_labels,g)

    vecs = vlabels()
    vecs.initialization(g_new)
    # propagation step
    # n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = {}.fromkeys(g_new.nodes(), k)
    cgrad = vlabel()

    for node in g.nodes():
        v = vlabel()
        v[old_labels[node]] = 1.0
        cgrad += v * g.degree(node)
    cgrad *= (- 1.0 / (2 * m))
    diff_cgrad = vlabel()
    shuffled_nodes = g_new.nodes()

    for step in xrange(80):
        random.shuffle(shuffled_nodes)  # shuffle the node list
        # update one node vector label
        if step % 10 in [0, 1, 2, 3, 4, 5, 6]:
            pos = {}.fromkeys(g_new.nodes(), k)
        else:
            pos = {}.fromkeys(g_new.nodes(), 1)

        for node in shuffled_nodes:
            pgrad = vlabels({neigh: vecs[neigh] * g_new[node][neigh]['weight'] for neigh in g_new.neighbors(node)}).sum()
            cgrad += diff_cgrad
            grad = pgrad.saddl(cgrad, g_new.degree(node,weight='weight'))
            update = (grad * gamma + vecs[node] * (1 - gamma)).nlarg(pos[node]).normalize(n=2)
            diff_cgrad = (update - vecs[node]) * (- float(g_new.degree(node,weight='weight')) / (2 * m))
            vecs[node] = update
        new_labels = vecs.to_labels()
        labels = {}

        for node in g.nodes():
            labels[node] = new_labels[old_labels[node]]


    return labels, community.modularity(labels,g)

def final_agg_louvain(g, k=5, gamma=0.6):
    # aggregtate when we doing the calculation
    g_cal = g.copy()
    for e in g_cal.edges():
        g_cal[e[0]][e[1]]['weight'] = 1.0
    for node in g_cal.nodes():
        g_cal.add_edge(node,node)
        g_cal[node][node]['weight'] = 0.0

    t1 = time.time()
    n2n = {}
    for node in g.nodes():
        n2n[node] = node
    mods = []
    vmods = []
    vecs = vlabels()
    vecs.initialization(g_cal)
    # propagation step
    # n = float(len(g.nodes()))
    m = float(len(g_cal.edges()))
    pos = {}.fromkeys(g_cal.nodes(), k)
    cgrad = vlabel()
    for node in g_cal.nodes():
        cgrad += vecs[node] * g_cal.degree(node)
    cgrad *= (- 1.0 / (2 * m))
    diff_cgrad = vlabel()
    shuffled_nodes = g_cal.nodes()

    def aggregrate_graph():
        inter_labels = vecs.to_labels()
        g_agg = nx.Graph()
        for e in g_cal.edges():
            l1 = inter_labels[e[0]]
            l2 = inter_labels[e[1]]
            g_agg.add_edge(l1, l2)
            g_agg[l1][l2]['weight'] = 0
        for e in g_cal.edges():
            l1 = inter_labels[e[0]]
            l2 = inter_labels[e[1]]
            g_agg[l1][l2]['weight'] += g_cal[e[0]][e[1]]['weight']
        vecs_agg = vlabels()

        for node in g_cal.nodes():
            if inter_labels[node] not in vecs_agg:
                vecs_agg[inter_labels[node]] = vecs[node]
        n2n_agg = {}
        for node in g.nodes():
            n2n_agg[node] = inter_labels[n2n[node]]

        return g_agg, vecs_agg, n2n_agg

    for step in xrange(80):
        shuffled_nodes = g_cal.nodes()
        random.shuffle(shuffled_nodes)  # shuffle the node list
        # update one node vector label
        if step % 10 in [0, 1, 2, 3, 4, 5, 6]:
            pos = {}.fromkeys(g_cal.nodes(), k)
        else:
            pos = {}.fromkeys(g_cal.nodes(), 1)

        for node in shuffled_nodes:
            pgrad = vlabels({neigh: vecs[neigh] * g_cal[node][neigh]['weight'] for neigh in g_cal.neighbors(node)}).sum()
            cgrad += diff_cgrad
            grad = pgrad.saddl(cgrad, g_cal.degree(node,weight='weight'))
            update = (grad * gamma + vecs[node] * (1 - gamma)).nlarg(pos[node]).normalize(n=2)
            diff_cgrad = (update - vecs[node]) * (- float(g_cal.degree(node,weight='weight')) / (2 * m))
            vecs[node] = update

        if step == 39:
            g_cal, vecs, n2n = aggregrate_graph()

        # aggregrate_graph


        vmods.append(vmod(vecs, g_cal))
        mods.append(community.modularity(vecs.to_labels(), g_cal))

    ## algorithm output
    algorithm = 'final_agg_louvain_vlpa'
    modularity = mods[len(mods) - 1]
    agg_labels = vecs.to_labels()
    labels = {}
    for node in g.nodes():
        labels[node] = agg_labels[n2n[node]]
    t2 = time.time() - t1
    ## get algorithm result
    result = algorithm_result(algorithm, t2, labels, vmods, mods, modularity)

    return result

def lock_louvain_vlpa(g, k=5, gamma=0.6):
    # try to get the best time complixity
    # initiazaiton
    t1 = time.time()
    mods = []
    vmods = []
    vecs = vlabels()
    vecs_best = vlabels()
    vmod_best = 0.0
    vecs.initialization(g)
    # propagation step
    # n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = {}.fromkeys(g.nodes(),k)
    cgrad = vlabel()
    for node in g.nodes():
        cgrad += vecs[node] * g.degree(node)
    cgrad *= (- 1.0 / (2 * m))
    diff_cgrad = vlabel()
    shuffled_nodes = g.nodes()

    def normal_update():
        vecs_new = vecs.copy()
        cgrad_new = cgrad.copy()
        diff_cgrad_new = diff_cgrad.copy()
        for node in shuffled_nodes:
            pgrad = vlabels({neigh: vecs_new[neigh] for neigh in g.neighbors(node)}).sum()
            cgrad_new += diff_cgrad_new
            grad = pgrad.saddl(cgrad_new, g.degree(node))
            update = (grad * gamma + vecs[node]*(1-gamma)).nlarg(pos[node]).normalize(n=2)
            diff_cgrad_new = (update - vecs_new[node]) * (- float(g.degree(node))/(2 * m))
            vecs_new[node] = update
        return vecs_new, cgrad_new, diff_cgrad_new

    def lock_update(n2l):
        vecs_new = vecs.copy()
        cgrad_new = cgrad.copy()
        diff_cgrad_new = diff_cgrad.copy()

        l2n = {}
        for node in n2l:
            if n2l[node] not in l2n:
                l2n[n2l[node]] = [node]
            else:
                l2n[n2l[node]] += [node]

        for l in l2n:
            random_node = random.choice(l2n[l])
            nodes = l2n[l]
            pgrad = vlabel()
            for node in nodes:
                pgrad += vlabels({neigh: vecs_new[neigh] for neigh in g.neighbors(node)}).sum()
            cgrad_new += diff_cgrad_new
            degree = sum([g.degree(node) for node in nodes])
            grad = pgrad.saddl(cgrad, degree)
            v0 = vlabels({node:vecs[node] for node in nodes}).sum()
            update = (grad * gamma + v0 * (1-gamma)).nlarg(pos[random_node]).normalize(n=2)

            diff_cgrad_new = (update - vecs_new[random_node]) * (- degree / (2 * m))
            for node in nodes:
                vecs_new[node] = update

        return vecs_new, cgrad_new, diff_cgrad_new


    for step in xrange(80):
        random.shuffle(shuffled_nodes) # shuffle the node list
        # update one node vector label
        if step%10 in [0,1,2,3,4,5,6]:
            pos = {}.fromkeys(g.nodes(), k)

        else:
            pos = {}.fromkeys(g.nodes(), 1)

        if step in range(40):
            vecs, cgrad, diff_cgrad = normal_update()
            n2l = vecs.to_labels()
        if step >39:
            vecs, cgrad, diff_cgrad = lock_update(n2l)


        vmods.append(vmod(vecs, g))
        mods.append(community.modularity(vecs.to_labels(), g))

    ## algorithm output
    algorithm = 'lock_louvain_vlpa'
    modularity = mods[len(mods) - 1]
    labels = vecs.to_labels()
    t2 = time.time() - t1
    ## get algorithm result
    result = algorithm_result(algorithm, t2, labels, vmods, mods, modularity)

    return result

def real_final_agg_louvain(g, k=5, gamma=0.6):
    # aggregtate when we doing the calculation
    g_cal = g.copy()
    for e in g_cal.edges():
        g_cal[e[0]][e[1]]['weight'] = 1.0
    for node in g_cal.nodes():
        g_cal.add_edge(node,node)
        g_cal[node][node]['weight'] = 0.0

    t1 = time.time()
    n2n = {}
    for node in g.nodes():
        n2n[node] = node
    mods = []
    vmods = []
    vecs = vlabels()
    vecs.initialization(g_cal)
    # propagation step
    # n = float(len(g.nodes()))
    m = float(len(g_cal.edges()))
    pos = {}.fromkeys(g_cal.nodes(), k)
    cgrad = vlabel()
    for node in g_cal.nodes():
        cgrad += vecs[node] * g_cal.degree(node)
    cgrad *= (- 1.0 / (2 * m))
    diff_cgrad = vlabel()
    shuffled_nodes = g_cal.nodes()

    def aggregrate_graph():
        inter_labels = vecs.to_labels()
        g_agg = nx.Graph()
        for e in g_cal.edges():
            l1 = inter_labels[e[0]]
            l2 = inter_labels[e[1]]
            g_agg.add_edge(l1, l2)
            g_agg[l1][l2]['weight'] = 0
        for e in g_cal.edges():
            l1 = inter_labels[e[0]]
            l2 = inter_labels[e[1]]
            g_agg[l1][l2]['weight'] += g_cal[e[0]][e[1]]['weight']
        vecs_agg = vlabels()

        for node in g_cal.nodes():
            if inter_labels[node] not in vecs_agg:
                vecs_agg[inter_labels[node]] = vecs[node]
        n2n_agg = {}
        for node in g.nodes():
            n2n_agg[node] = inter_labels[n2n[node]]

        return g_agg, vecs_agg, n2n_agg

    def discrete_graph():
        label2node = {}
        for node in n2n:
            if n2n[node] not in label2node:
                label2node[n2n[node]] = [node]
            else:
                label2node[n2n[node]] += [node]
        vecs_new = vlabels()
        for node in n2n:
            vecs_new[node] = vecs[n2n[node]]

        g_new = g.copy()
        for e in g_new.edges():
            g_new[e[0]][e[1]]['weight'] = 1.0
        for node in g_new.nodes():
            g_new.add_edge(node, node)
            g_new[node][node]['weight'] = 0.0
        n2n_new = {}
        for node in g.nodes():
            n2n_new[node] = node
        return g_new, vecs_new, n2n_new

    for step in xrange(200):
        shuffled_nodes = g_cal.nodes()
        random.shuffle(shuffled_nodes)  # shuffle the node list
        # update one node vector label
        if step % 10 in [0, 1, 2, 3, 4, 5, 6]:
           pos = {}.fromkeys(g_cal.nodes(), k)
        else:
           pos = {}.fromkeys(g_cal.nodes(), 1)

        for node in shuffled_nodes:
            pgrad = vlabels({neigh: vecs[neigh] * g_cal[node][neigh]['weight'] for neigh in g_cal.neighbors(node)}).sum()
            cgrad += diff_cgrad
            grad = pgrad.saddl(cgrad, g_cal.degree(node,weight='weight'))
            update = (grad * gamma + vecs[node] * (1 - gamma)).nlarg(pos[node]).normalize(n=2)
            diff_cgrad = (update - vecs[node]) * (- float(g_cal.degree(node,weight='weight')) / (2 * m))
            vecs[node] = update

        if step >37 and step%10 in [7,8,9]:
            g_cal, vecs, n2n = aggregrate_graph()
        elif step >37 and step%10 in [0,1,2,3,4,5,6]:
            g_cal, vecs, n2n = discrete_graph()



        # aggregrate_graph


        vmods.append(vmod(vecs, g_cal))
        mods.append(community.modularity(vecs.to_labels(), g_cal))

    ## algorithm output
    algorithm = 'real_final_agg_louvain_vlpa'
    modularity = mods[len(mods) - 1]
    agg_labels = vecs.to_labels()
    labels = {}
    for node in g.nodes():
        labels[node] = agg_labels[n2n[node]]
    t2 = time.time() - t1
    ## get algorithm result
    result = algorithm_result(algorithm, t2, labels, vmods, mods, modularity)

    return result