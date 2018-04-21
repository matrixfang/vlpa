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
    g, real_label = inputdata.read_lfr(0.6)
    result1 = vlpa.fixed_pos_louvain_vlpa(g, ifrecord=False)
    pass

cProfile.run("timeit_profile()", filename="result.out")