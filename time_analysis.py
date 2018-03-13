import copy
import vlpa
import inputdata
import time
g,label = inputdata.read_lfr(0.6)
vecs = vlpa.vlabels()
vecs.initialization(g)
v = vecs[300]
t1= time.time()
for i in xrange(10000):
    a = v.copy()
t2 = time.time()
print(t2-t1)


