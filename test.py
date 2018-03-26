import scipy.sparse as spsp
import scipy as sp
import time
import copy
import inputdata
import random
import vlpa
from collections import namedtuple

g,label = inputdata.read_lfr(0.6)

v = vlpa.vlabel()
v[1]=2.0
v[2]=2.0
v[3] = 1.0

print(v)
print(v.close2label(0.5))