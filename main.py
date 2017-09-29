import networkx as nx
import numpy as np
import scipy as sp
import inputdata
import draw
import matplotlib.pyplot as plt
import vlpa


G, community_label = inputdata.read_lfr(800)

vlabels = vlpa.vlabels(G)
print(vlabels)





# a = vlpa.vlabel()
# b = vlpa.vlabel()
# c = vlpa.vlabel()
# b.fromdic({1:2,3:4})
# a.fromdic({1:3,5:8})
# print(b + (a * 10))
# print(b + a * 10)
# print(a * 10 + b)