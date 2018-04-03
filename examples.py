import networkx as nx
import inputdata
import vlpa
import time


g,label = inputdata.read_lfr(200)
v = vlpa.vlabel({}.fromkeys(range(20),1))
v2 = v.copy()
t1 = time.time()
N = 10000
for i in xrange(N):
    print(v.ifclose(v2,argument='inner'))
t2 = time.time()-t1
t3 = time.time()
for i in xrange(N):
    print(v.ifclose(v2,argument='label'))
t4= time.time()-t3

print(t4,t2)