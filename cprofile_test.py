import vlpa
import inputdata
import networkx as nx
import numpy as np
import scipy as sp
import draw
import cProfile

def timeit_profile():
    """
    check running time bottleneck
    """
    g, real_label = inputdata.read_lfr(0.9)
    result1 = vlpa.random_vlpa_best(g, ifrecord=False)
    pass

cProfile.run("timeit_profile()", filename="result.out")