# 20181009

How to realize sgd?

$Q = \sum_i Q_i$

1.sum of two parts

$$
2mQ=\sum_{ij} A_{ij} (v_i,v_j) -\sum_{ij}\frac{k_ik_j}{2m}(v_i,v_j)
$$

2.sum of $n(n-1)/2$ parts

$$
2mQ=\sum_{ij} (A_{ij} (v_i,v_j) -\frac{k_ik_j}{2m}(v_i,v_j))
$$

3. sum of n parts

$$
2mQ = \sum_i\sum_j (A_{ij} (v_i,v_j) -\frac{k_ik_j}{2m}(v_i,v_j))
$$





## experimental  results

results of situation one:

not good, because the two parts are not alike.





## 20181010

try random positive partical derivative by p_i+r*ball

![Screen Shot 2018-10-10 at 10.49.38 AM](/Users/fangwenyi/Documents/GitHub/vlpa/Screen Shot 2018-10-10 at 10.49.38 AM.png)

not good for r=0.7 just as well as VPAs with randonlarg2.



