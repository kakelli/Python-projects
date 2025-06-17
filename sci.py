from scipy.stats import kstest
import numpy as np
import matplotlib.pyplot as plt
#Checking the version
'''print(scipy.__version__)'''

#Scientific constants
'''print(constants.pi)'''

#Units of constants
'''print(dir(constants))'''
#Types
'''print(constants.gram)'''

#Finding a root of non linear equation- optimizers.root:
'''def eqn(x):
    return x+cos(x)

f = root(eqn, 0)
print(f)'''

#Finding Minima of a function - scipy.optimizers.minimize():
'''def eqn(x):
    return x**2+x+2

g = minimize(eqn, 0, method='BFGS')
print(g)'''

#Compressed Sparse Row Matrix():
'''d = np.array([0,0,0,0,1,2,0,0,0,])
print(csr_matrix(d))'''

#Viewing non-zero elements- data:
'''a = np.array([0,0,0,0,3,0,1,2,0,0])
print(csr_matrix(a).data)'''

#Counting the non-zero elemnts- count_nonzero:
'''d = np.array([[0,0,0],[2,0,0],[0,3,0],[1,2,0]])
print(csr_matrix(d). count_nonzero)'''

#Removing zero entries- eliminate_zeros():
'''f = np.array([0,0,0,0,0,3,2,0,0,2,2,0,0])
g = csr_matrix(f)
g.eliminate_zeros()
print(g)'''

#Conversion from  CSR to CSC:
'''d = np.array([[0,0,7], [0,1,2], [0,9,2]])
x = csr_matrix(d).tocsc
print(x)'''

#Working with graphs- chcecking the connection:
'''s = np.array([
    [0,1,2],
    [1,0,0],
    [2,0,0]
])
d= csr_matrix(s)
print(connected_components(d))'''

#Dijkstra
'''d = np.array([
    [0,1,2],
    [1,0,0],
    [2,0,0]
])

k = csr_matrix(d)
print(dijkstra(k, return_predecessors = True, indices = 0))'''

#Floyd warshall method- floyd_warshall():
'''r = np.array([
    [0,1,2],
    [1,0,0],
    [2,0,0]
])
d = csr_matrix(r)
print(floyd_warshall(d, return_predecessors = True))'''

#Bellman Ford method- bellman_ford():
'''r = np.array([
    [0,1,2],
    [1,0,0],
    [2,0,0]
])
d = csr_matrix(r)
print(bellman_ford(d, return_predecessors = True))'''

#Depth first order- depth_first_order():
'''d = np.array([
    [0,1,0,1],
    [1,1,1,1],
    [2,1,1,0],
    [0,1,0,1]
])
e = csr_matrix(d)
print(depth_first_order(e, 1))'''

#Breadth first order- breadth_first_order():
'''d = np.array([
    [0,1,0,1],
    [1,1,1,1],
    [2,1,1,0],
    [0,1,0,1]
])

v = csr_matrix(d)
print(breadth_first_order(v, 1))'''

#Triangulation- Delauney():

'''points = np.array([
    [2,4],
    [3,4],
    [3,0],
    [2,2],
    [4,1]
])

plot = Delaunay(points).simplices

plt.triplot(points[: ,0], points[:,1], plot)
plt.scatter(points[:,0], points[:,1], color = 'r')
plt.show()'''

#Convex hull- ConvexHull():
'''points = np.array([
    [2,4],
    [5,6],
    [2,0],
    [3,0],
    [3,3],
    [2,2]
])

plot = ConvexHull(points)
simplices = plot.simplices

plt.scatter(points[:,0], points[:,1])
for i in simplices:
    plt.plot(points[i,0], points[i,1], 'k-')

plt.show()'''

#KDTree method- KDTree() and queries():

'''points = [(-1,1), (0,2), (-1,2), (2,3)]

f = KDTree(points)
r= f.query((1,1))
print(r)'''

#Euclidean Distance between two points- euclidean():
'''p1 = (2,0)
p2 = (2,3)

res = euclidean(p1,p2)
print(res)'''

#Cityblock(Manhattan) Distance- cityblock():
'''p1 = (2,4)
p2 = (3,6)

f = cityblock(p1,p2)
print(f)'''

#Cosine Distance- cosine():
'''p1 = (1,3)
p2 = (2,3)

r = cosine(p1,p2)
print(r)'''

#Hamming distance- hammming():
'''p1 = (True, True, True)
p2 = (False, True, False)

f = hamming(p1,p2)
print(f)'''

#Exporting Data in matlab- savemat():
'''arr = np.arange(10)
io.savemat('arr.mat', {"vec":arr})'''

#Importing data in matlab- loadmat():
'''arr = np.array([0,1,2,3,4,5,6,7,8,9])
io.savemat('arr.mat', {"vec":arr})#Export
o = io.loadmat('arr.mat')#Import
print(o["vec"])'''

#Using squeeze_me to make in 1D:
'''o = io.loadmat('arr.mat', squeeze_me = True)
print(o['vec'])'''

#1D Interpolation- interp1d():
'''x = np.arange(10)
y = 2*x +1

inter_func = interp1d(x,y)
arr = inter_func(np.arange(2.1,3,0.1))
print(arr)'''

#Spline Interpolation- UnivariateSpline():
'''x = np.arange(10)
y = x**2+ np.sin(x)+1

inter_func = UnivariateSpline(x,y)
new_arr = inter_func(np.arange(2.1,3,0.1))
print(new_arr)'''

#Interpolation with radial basis function- Rbf():
'''x = np.arange(10)
y = x**2+np.sin(x)+1

func = Rbf(x,y)
f  = func(np.arange(2.3,3,0.1))
print(f)'''

#T-test - ttest_ind():
'''v1 = np.random.normal(size=100)
v2 = np.random.normal(size=100)

r = ttest_ind(v1,v2).pvalue
print(r)'''

#KS-Test- kstest():
'''d = np.random.normal(size=100)
r = kstest(d, 'norm')
print(r)'''



