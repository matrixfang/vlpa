import networkx as nx
import numpy as np
import scipy as sp
import inputdata
import draw
import matplotlib.pyplot as plt
import vlpa


G, community_label = inputdata.read_lfr(500)

pro = vlpa.Propragation(G)
labels = pro.run()
print(set(labels.values()))

draw.draw_group(G, community_label, labels)
plt.show()



# a = vlpa.vlabel()
# b = vlpa.vlabel()
# c = vlpa.vlabel()
# b.get_from({1:2,3:4})
# a.get_from({1:3,5:8})
# print(b + a +c)
# print(c + a+b)