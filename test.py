import scipy.sparse as spsp
import scipy as sp
import vlpa
import time
l1 = [1,2,3,0,0,0,4,5]
l2 = [1,0,0,0,0,0,0,2]
a = spsp.dok_matrix((1,8))
b = spsp.dok_matrix((1,8))

va = vlpa.vlabel()
vb = vlpa.vlabel()


print(a.shape)
for i in range(8):
    if a[0,i]!=0:
        va[i]=a[0,i]
    if b[0,i]!=0:
        vb[i]=b[0,i]
N = 100000

t1 = time.time()


for x in xrange(N):
    s1 = va + vb
t2= time.time()

t3=time.time()

for x in xrange(N):
    s2 = a + b
t4=time.time()

print(t2-t1,t4-t3)

