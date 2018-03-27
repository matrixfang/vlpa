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


def vlpa(g):
    # initiazaiton
    vecs = vlabels()
    vecs.initialization(g)
    # propagation step
    n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = g.degree()
    k_ave = float(sum(g.degree().values())) / n
    for step in xrange(80+10):
        if step==80:
            pos={}.fromkeys(g.nodes(),1)

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


def vmod(g, vecs):
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


algorithm_output = namedtuple('algorithm_output',['algorithm','time','labels','vmods','mods','before_mod','after_mod'])


def fixed_pos_vlpa(g,k):
    # initiazaiton
    vecs = vlabels()
    vecs.initialization2(g)
    # propagation step
    n = float(len(g.nodes()))
    m = float(len(g.edges()))
    pos = {}.fromkeys(g.nodes(),k)
    k_ave = float(sum(g.degree().values())) / n
    for step in xrange(80+10):
        if step==80:
            pos={}.fromkeys(g.nodes(),1)

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
    print(mod+' '+str(gamma), 'Time %f'%(t2 - t1), 'modularity before lpa is %f'%(modularity[len(modularity)-1]),'modularity after lpa is %f'%(community.modularity(vecs.to_labels(),g)))
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


def louvain_vlpa(g, gamma=0.5):
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

    for step in xrange(100 + 10 + 10):
        random.shuffle(shuffled_nodes) # shuffle the node list
        # update one node vector label
        if step==100:
            before_mod = community.modularity(vecs.to_labels(), g)
            pos = {}.fromkeys(g.nodes(), 1)
        elif step%20==10:
            pos = {}.fromkeys(g.nodes(), 1)
        elif step%20==0:
            pos = g.degree()

        for node in shuffled_nodes:
            pgrad = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()
            cgrad += diff_cgrad
            grad = cgrad * g.degree(node) + pgrad
            update = (grad * gamma + vecs[node]*(1-gamma)).nlarg(pos[node]).normalize(n=2)
            diff_cgrad = (update - vecs[node]) * (- float(g.degree(node))/(2 * m))
            vecs[node] = update

        vmod_vecs = vmod(g,vecs)
        vmods.append(vmod_vecs)
        mods.append(community.modularity(vecs.to_labels(), g))
        if vmod_vecs>vmod_best:
            vmod_best = vmod_vecs
            vecs_best = vecs

    vecs = vecs_best
    t2 = time.time()
    after_mod = community.modularity(vecs.to_labels(), g)

    """
    output result

    """

    result = algorithm_output(algorithm="louvian_vlpa",time=t2-t1,labels=vecs.to_labels(),
                              vmods=vmods, mods=mods, before_mod=before_mod,after_mod=after_mod)

    """
    print result
    """

    print(str(gamma), 'Time %f' % (t2 - t1), 'modularity before is %f' % (before_mod),
          "modularity after is %f" % (after_mod))

    return result


def vlpa_sgd_ada(g, gamma=0.5):
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
    eg = {}.fromkeys(g.nodes(),1.0)
    ev = {}.fromkeys(g.nodes(),1.0)

    for step in xrange(100 + 10 + 10):
        random.shuffle(shuffled_nodes) # shuffle the node list
        # update one node vector label
        if step==100:
            before_mod = community.modularity(vecs.to_labels(), g)
            pos = {}.fromkeys(g.nodes(), 1)
        elif step%10==5:
            pos = {}.fromkeys(g.nodes(), 1)
        elif step%10==0:
            pos = g.degree()


        for node in shuffled_nodes:
            pgrad = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()
            cgrad += diff_cgrad
            grad = cgrad * g.degree(node) + pgrad
            eg[node]=gamma * eg[node] + (1 - gamma) * grad.norm(n=2)**2
            learning_rate = rms(ev[node])/rms(eg[node])
            #print(learning_rate)
            delta_v = grad * learning_rate
            update = (delta_v + vecs[node]).nlarg(pos[node]).normalize(n=2)
            diff_cgrad = (update - vecs[node]) * (- float(g.degree(node))/(2 * m))
            ev[node] = gamma * ev[node] + (1 - gamma) *(delta_v).norm(n=2)**2
            vecs[node] = update

        vmod_vecs = vmod(g,vecs)
        vmods.append(vmod_vecs)
        mods.append(community.modularity(vecs.to_labels(), g))
        if vmod_vecs>vmod_best:
            vmod_best = vmod_vecs
            vecs_best = vecs

    vecs = vecs_best
    t2 = time.time()
    after_mod = community.modularity(vecs.to_labels(), g)

    """
    output result

    """

    result = algorithm_output(algorithm="louvian_vlpa",time=t2-t1,labels=vecs.to_labels(),
                              vmods=vmods, mods=mods, before_mod=before_mod,after_mod=after_mod)

    """
    print result
    """

    print(str(gamma), 'Time %f' % (t2 - t1), 'modularity before is %f' % (before_mod),
          "modularity after is %f" % (after_mod))

    return result


def pos_vlpa(g):
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

    def change_pos():
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

        vecs_grad = vecs_grad + vecs_all  # calculate the gradient = vecs_grad

        Eg = gamma * Eg + (1 - gamma) * vecs_grad.norm2()
        coef = np.sqrt(Edelta + epsilon) / np.sqrt(Eg + epsilon)
        vecs_delta = vecs_grad * coef
        vecs_new = (vecs + vecs_delta).nlarg(pos).normalize(n=2)
        Edelta = gamma * Edelta + (1 - gamma) * vecs_delta.norm2()


        ##### iteration end #####
        vecs = vecs_new
        change_pos()
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
    print('Time of module of pos is ', time2 - time1,modularity[len(modularity)-1])
    return modularity


def no_pos_vlpa(g):
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

    def change_pos():
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
    print('Time of module of no pos is ', time2 - time1)
    return modularity


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
    real_commuinties = {}
    # transform to original commuinties
    for i in communities:
        real_commuinties[list[i]] = communities[i]

    print('modularity by infomap is %f' % (community.modularity(real_commuinties, G)))

    return real_commuinties


def nonmodel(g):
    return {}.fromkeys(g.nodes(), 1)


def louvain(g):
    t1 = time.time()
    labels = community.best_partition(g)
    t2 = time.time()
    before_mod = None
    after_mod = community.modularity(labels,g)

    """
        output result

    """

    result = algorithm_output(algorithm="louvian", time=t2 - t1, labels=labels,
                              vmods=None, mods=None, before_mod=before_mod, after_mod=after_mod)

    """
    print result
    """

    print('louvian', 'Time %f' % (t2 - t1), 'modularity before is',
          "modularity after is %f" % (after_mod))

    return result


def louvain_vlpa_opt(g, gamma=0.5):
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

    for step in xrange(100 + 10 + 10):
        random.shuffle(shuffled_nodes) # shuffle the node list
        # update one node vector label
        if step==100:
            before_mod = community.modularity(vecs.to_labels(), g)
            pos = {}.fromkeys(g.nodes(), 1)
        # elif step%20==10:
        #     pos = {}.fromkeys(g.nodes(), 1)
        # elif step%20==0:
        #     pos = g.degree()

        for node in shuffled_nodes:
            pgrad = vlabels({neigh: vecs[neigh] for neigh in g.neighbors(node)}).sum()
            cgrad += diff_cgrad
            grad = cgrad * g.degree(node) + pgrad
            update = (grad * gamma + vecs[node]*(1-gamma)).nlarg(pos[node]).normalize(n=2)
            diff_cgrad = (update - vecs[node]) * (- float(g.degree(node))/(2 * m))
            vecs[node] = update.close2label(0.02).normalize(n=2)

        vmod_vecs = vmod(g,vecs)
        vmods.append(vmod_vecs)
        mods.append(community.modularity(vecs.to_labels(), g))
        if vmod_vecs>vmod_best:
            vmod_best = vmod_vecs
            vecs_best = vecs

    vecs = vecs_best
    t2 = time.time()
    after_mod = community.modularity(vecs.to_labels(), g)

    """
    output result

    """

    result = algorithm_output(algorithm="louvian_vlpa",time=t2-t1,labels=vecs.to_labels(),
                              vmods=vmods, mods=mods, before_mod=before_mod,after_mod=after_mod)

    """
    print result
    """

    print(str(gamma), 'Time %f' % (t2 - t1), 'modularity before is %f' % (before_mod),
          "modularity after is %f" % (after_mod))

    return result