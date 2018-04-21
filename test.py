import scipy.sparse as spsp
import scipy as sp
import time
import copy
import numpy as np
import inputdata
import networkx as nx
from random import choice
import matplotlib.pyplot as plt
import random
from vlpa import vlabel
from vlpa import vlabels
from vlpa import vmod
from vlpa import algorithm_result
import community
import vlpa
from collections import namedtuple

v1 = vlabel({1:3,2:4,5:6,9:5})
v2 = vlabel({2:3,4:3,101:30,50:60})
v3 = np.array(range(1000))
v4 = np.array(range(1000))
N = 100000
t1 = time.time()
for n in xrange(N):
    s1 = v1 + v2
t2 = time.time()
t3 = time.time()
for n in xrange(N):
    s2 = v3 + v4
t4 = time.time()

