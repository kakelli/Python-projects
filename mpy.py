from numpy import random
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from math import log

'''arr1 = np.array(20) #THis is 0D array
arr2 = np.array([1,2,3,4,5])#This is a 1D array
arr3 = np.array([[1,2,3],[4,5,6]])
arr4 =np.array([[[1,2,3], [4,5,6]], [[11,12,13],[14,15,16]]])#This is 3D array
print(arr1.ndim)
print(arr2.ndim)
print(arr3.ndim)
print(arr4.ndim)'''

'''b=np.array([1,2,3,4,5], ndmin=5)
print(b)
print(b.ndim)'''

'''arr3 = np.array([[1,2,3],[4,5,6]])
print(arr3[0:2,0:2])'''

'''a = np.array([1,2,3,4], dtype='i4')
print(a.dtype)'''

'''c = np.array([1,2,3])
d = c.astype(bool)

print(d.dtype)
print(d)'''

#Making a copy of array
'''c = v.copy()
v[2] = 69

print(v)
print(c)'''

#Making a view of an array & checking
'''d = np.array([1,2,3,4,5])
f = d.view()
c = d.copy()
d[3] = 69

print(c.base)
print(f.base)'''

#Finding the shape of the array
'''s = np.array([1,2,3,4,5], ndmin = 7)
print(s.shape)'''

#Reshaping the array
'''f = np.array([1,2,3,4,5,6,7,8])
d = f.reshape(2,2,-1)

print(d.base)'''

'''f = np.array([[1,2,3,4,5], [6,7,8,9,10]])
d = f.reshape(-1)
print(d)'''

#Iterating through loop
'''e = np.array([1,2,3,4,5])#1-D looping
for i in e:
    print(i)'''

'''r = np.array([[1,2,3,4], [5,6,7,8]])#2D looping
for i in r:
    for j in i:
        print(j)'''

'''t = np.array([[[1,2,3], [4,5,6]], [[7,8,9],[10,11,12]]])#3D looping

for i in t:
    for j in i:
        for k in j:
            print(k)'''

#Iterating through nditer()
'''e = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
for i in np.nditer(e):
    print(i)'''

#Chnaging datatype in iteration
'''j = np.array([1,2,3,4])
for i in np.nditer(j,flags=['buffered'], op_dtypes=['S']):
    print(i)'''

#Step value in Iteration
'''r = np.array([[1,2,3,4], [5,6,7,8]])
for i in np.nditer(r[:, ::2]):
    print(i)'''

#enumeration(finding index of the particular value)
'''y = np.array([[1,2,3],[4,5,6]])
for y,x in np.ndenumerate(y):
    print(y,x)'''

#Concatenation(joining of two array)
'''d = np.array([[1,2],[3,4]])
r = np.array([[6,7], [8,9]])

arr = np.concatenate((d,r), axis = 1)
print(arr)'''

#Row Stacking(horizintal)
'''f = np.array([1,2,3,4])
t = np.array([6,7,8,9])

arr = np.hstack((f,t))
print(arr)'''

#Column stacking(vertical)
'''g = np.array([1,2,3,4])
u = np.array([6,7,8,9])

arr = np.vstack((g,u))
print(arr)'''

#Height stacking(Depth)
'''q = np.array([1,2,3,4])
l = np.array([6,7,8,9])

arr = np.dstack((q,l))
print(arr)'''

#Splitting arrays
'''o = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15], [16,17,18]])    

arr = np.array_split(o, 3, axis=1)
print(arr   )'''

#Horizontal splitting
'''c = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15], [16,17,18]])
arr = np.hsplit(c,3)
print(arr)'''

#Searching array
'''f = np.array([1,2,3,4,5,6,3,4,4,3,4])
arr = np.where(f%2==0)
print(arr)'''

#Sorting out searches
'''n=np.array([1,2,3,4,5])
arr = np.searchsorted(n,3,side="right")#The default is left
print(arr)'''

#Sorting arrays(num)
'''n = np.array([3,4,5,2,0,1,9])
arr = np.sort(n)
print(arr)
print(n)#Original value is not changed'''

#Sorting arrays(string)
'''g = np.array(["cherry", "bananna", "apple"])
arr = np.sort(g)
print(arr)'''

#Sorting arrays(boolean)
'''d = np.array([True, False, True])
print(np.sort(d))'''

#Sorting 2D arrays
'''e = np.array([[1,2,5], [0,4,6]])
print(np.sort(e))'''

#Filtering arrays
'''t = np.array([1,2,3,4])
x = [True,False,True,False]

d=t[x]
print(d)'''

#Application
'''arr = np.array([13,14,15,16])
filterarray = []

for i in arr:
    if i%2==0:
        filterarray.append(True) 
    else:
        filterarray.append(False)
newarray = arr[filterarray]
print(newarray)'''

#Pseudo random number
'''x = random.randint(5)
print(x)'''
#Pseudo random float
'''c = random.rand()
print(c)'''

#Generating random arrays 1D
'''d = random.randint(10, size=(5))
print(d)'''

#Generatng random arrays 2D(row,column)
'''f = random.randint(10, size=(3,5))
print(f)'''

#Generating array sizes
'''s =random.rand(2,4)
print(s)'''
#Generate random numbers from array
'''f = random.choice([1,2,3,4,5,6,7,8,9],size=(3,5))
print(f)'''

#Probability setting
'''p = random.choice([1,2,3,4], p=[0.1, 0.2,0.7, 0.0], size=(2,2))
print(p)'''

#Random Permutations(shuffle)
'''f = np.array([1,2,3,4,5])
random.shuffle(f)
print(f)'''

#Random Permutation(permutation)
'''p = np.array([1,2,3,4,5])
print(random.permutation(p))'''

#Normal Distribution randomly
'''x = random.normal(size = (2,3))
print(x)'''

#Normal distribution based on seaborn
'''sns.displot(random.normal(size = 100), kind='kde')
plt.show()'''

#Seaborn(using displot)
'''sns.displot([0,1,2,3,4,5], kind='kde')
plt.show()'''

#Seaborn(without histogram)
'''sns.displot([0,1,2,3,4,5], kind = 'kde')
plt.show()'''

#Binomial Distribution
'''e = random.binomial(n=10,p=0.5, size = 10)
print(e)'''

#Visualizing binomial distribution
'''r = random.binomial(n=10,p=0.5,size = 1000)
sns.displot(r)
plt.show()'''

#Data distribtuion in seaborn normal and binomial
'''data = {
    'normal':random.normal(loc=50,scale=5,size = 1000),
    'binomial':random.binomial(n = 100, p = 0.5,size = 1000)
}
sns.displot(data,kind='kde')
plt.show()'''

#Poisson Distribution
'''t = random.poisson(lam =2, size = 10)
print(t)'''

#Poisson Distribution using seaborn
'''s = random.poisson(lam=2,size=1000)
sns.displot(s)
plt.show()'''

#Data distribution in seaborn normal v/s poisson
'''data={
    'normal': random.normal(loc = 50, scale = 7, size = 1000),
    'poisson': random.poisson(lam = 50,size = 1000)
}
sns.displot(data, kind='kde')
plt.show()'''

#Data distribution in seaborn binomial v/s poisson
'''data={
    'binomial':random.binomial(n=1000,p=0.01,size = 1000),
    'poisson':random.poisson(lam=10,size=1000)
}
sns.displot(data, kind='kde')
plt.show()'''

#Random uniform distribution
'''o = random.uniform(size=(2,3))
print(o)'''

#Random uniform distribution seaborn
'''i = random.uniform(size=1000)
sns.displot(i,kind='kde')
plt.show()'''

#Random logistical distribution 
'''l = random.logistic(loc = 1,scale =2,size=(2,3))
print(l)'''

#Random logistical distribution seaborn
'''l = random.logistic(size=1000)
sns.displot(l,kind='kde')
plt.show()'''

#Data distribution using Logistical and Normal
'''data={
    'normal': random.normal(scale=2,size=1000),
    'logistics':random.logistic(size=1000)
}
sns.displot(data,kind='kde')
plt.show()'''

#Multinomial Distribution
'''x = random.multinomial(n=6,pvals=[1/6,1/6,1/6,1/6,1/6,1/6])
print(x)'''

#Exponenetial Distribution
'''d = random.exponential(scale=2,size=(2,3))
print(d)'''

#Exponential Distribution using seaborn
'''d = random.exponential(size=1000)
sns.displot(d,kind='kde')
plt.show()'''

#Chi Square Distribution
'''x = random.chisquare(df = 2,size=(2,3))
print(x)'''

#Chi Square Distribution seaborn
'''c = random.chisquare(df=1,size=1000)
sns.displot(c,kind='kde')
plt.show()'''

#Rayleigh Distribution
'''r = random.rayleigh(scale=2,size=(2,3))
print(r)'''

#Rayleigh Distribution using seaborn
'''t = random.rayleigh(size=1000)
sns.displot(t, kind='kde')
plt.show()'''

#Pareto Distribution(80-20)
'''d = random.pareto(a = 2,size=(2,3))
print(d)'''

#Pareto Distribution seaborn
'''p = random.pareto(a=2,size=1000)
sns.displot(p)
plt.show()'''

#Zipf Distribution
'''z = random.zipf(a=2,size=(2,3))
print(z)'''

#Zipf Distribution
'''x = random.zipf(a = 2,size=(2,3))
print(x)'''

#Zipf Distribution seaborn
'''r = random.zipf(a=2,size=1000)
sns.displot(r[r<10])
plt.show()'''

#without ufunc zip():
'''x = [1,2,3,4,5]
y = [6,7,8,9,10]
z=[]

for i,j in zip(x,y):
    z.append(i+j)
print(z)'''

#with ufunc add():
'''x=[1,2,3,4]
y=[5,6,7,8]
z=np.add(x,y)
print(z)'''

#Creating a ufunc:
'''def myadd(x,y):
    return x+y
myadd = np.frompyfunc(myadd,2,1)
print(myadd([1,2,3,4],[5,6,7,8]))'''

#Checking the ufnc:
#print(type(np.selmon))

#Application
'''if type(np.add) == True:
    print("THis is a universal function")
else:
    print("No this aint")'''

#Addition ufunc
'''s = np.array([1,2,3])
f = np.array([1,2,3])
y = np.add(s,f)
print(y)'''

#Subtraction ufunc
'''r = np.array([11,12,13,14,15])
t = np.array([21,34,54,45,56])
p=np.subtract(r,t)
print(p)'''

#Multiplication ufunc
'''e = np.array([12,34,22,34,35,44])
k =  np.array([11,22,33,44,55,66])
d=np.multiply(e,k)
print(d)'''

#Division ufunc
'''d = np.array([1,2,3,4,5,6])
f = np.array([11,22,33,44,55,66])
c = np.divide(d,f)
print(c)'''

#Power ufunc
'''p = np.array([22,33,44,55])
o = np.array([1,2,3,4])
f = np.power(p,o)
print(f)'''

#Remainder ufunc with mod():
'''r = np.array([30,40,50,60])
a = np.array([3,4,5,15])
d = np.mod(r,a)
print(d)'''

#Remainder ufunc with remainder():
'''r = np.array([12,23,34,45,56])
t = np.array([12,34,24,25,56])
print(np.remainder(r,t))'''

#Divmod ufunc
'''a = np.array([10,20,30,40])
b = np.array([1,2,3,4])
f = np.divmod(a,b)
print(f) '''

#Absoltue values with adsolute() or abs():
'''p = np.array([1,-2,3,-4,5])
f = np.abs(p)
print(f)'''

#Removal of decimals is truncation:
#Truncating decimal - trunc(): 
'''r = np.array([-3.122,-9.344])
d = np.trunc(r)
print(d)'''

#Truncating decimal - fix():
'''d = np.array([-2.122, -0.991])
f = np.fix(d)
print(f)'''

#Rounding decimal - around():
'''s = np.array([3.4442,5.9888,6]) 
print(np.around(s))'''

#Rounding off decimals to nearest lower integer is floor
#Flooring of decimals
'''d = np.array([-3.14444,3.66698])
print(np.floor(d))'''

#Rounding off decimals to nearest upper integer is ceil
#Ceiling of decimals
'''f = np.array([9.0009,-9.86755])
print(np.ceil(f))'''

#Log with base 2: log2()
'''s = np.arange(1,10)#10 not included
print(np.log2(s))'''

#Log with base 10: log10()
'''g = np.arange(1,11)
print(np.log10(g))'''

#Natural log: loge()
'''f = np.arange(1,11)
print(np.log(f))'''

#log at any base: 
'''l = np.frompyfunc(log,2,1)
print(l(4,2)) #head,base'''

#Summation in Numpy python
'''a1 = np.array([1,2,3])
a2 = np.array([4,5,6])

s = np.sum([a1,a2])
print(s)'''

#Summation as a matrix
'''s = np.array([1,2,3])
f = np.array([4,5,6])

d = np.sum([s,f], axis = 1)
print(d)'''

#Cummulative summation
'''a1 = np.array([1,2,3,4])
print(np.cumsum(a1))'''

#Product in Numpy function: prod()
'''d = np.array([1,2,3,4])
f = np.prod(d)
print(f)'''

#Product as a matrix in input:
'''f = np.array([1,2,3])
d = np.array([1,2,3])
g = np.prod([f,d])
print(g)'''

#Product as a matrix in output:
'''d = np.array([1,2,3])
t = np.array([1,2,3])
k = np.prod([d,t], axis=1)
print(k)'''

#Cummulative product
'''f = np.array([1,2,3])
print(np.cumprod(f))'''

#Difference in numpy matrix
'''d = np.array([1,2,3,4]) #2-1,3-2,4-3
arr = np.diff(d)
print(arr)'''

#Difference in mumpy matrix with n:
'''s = np.array([1,2,3,4])#Will subtract twicw
h = np.diff(s,n=2)
print(h)'''

#Finding LCM in python
'''n1 = 4
n2 = 5
print(np.lcm(n1,n2))'''

#LCM in arrays
'''n1 = np.array([3,4,6])
print(np.lcm.reduce(n1))'''

#LCM in ranges
'''ar = np.arange(1,10)
print(np.lcm.reduce(ar))'''

#GCD in python
'''n = 4
f = 6
print(np.gcd(n,f))'''

#GCD(HCF) in arrays:
'''b = np.array([1,2,3,4])
print(np.gcd.reduce(b))'''

#Numpy trigonometric function- sin():
'''x = np.sin(np.pi/2)
print(x)'''

'''d = np.array([np.pi/2,np.pi/3,np.pi/4])
f = np.sin(d)
print(f)'''

#Degree to radian
'''d = np.array([90,80,270,360])
u = np.deg2rad(d)
print(u)'''

#Radian to degree
'''r = [1.57079633 ,1.3962634,  4.71238898 ,6.28318531]
f = np.rad2deg(r)
print(f)'''

#Inverse triginometric function- arcsin():
'''a = np.arcsin(1)
print(np.rad2deg(a))'''

#Inverse ytigonometric functions as an array- arcsin():
'''a = np.array([-1,1,-0.1])
l = np.arcsin(a)
print(np.rad2deg(l))'''

#Finding hypotenuse through numpy
'''b = 3
h = 6
print(np.hypot(b,h))'''

#Numpy hyperbolic functions- sinh():
'''b = np.sinh(np.pi/2)
print(b)'''

#Numpy functions hyperbolic - cosh():
'''f = np.array([np.pi, np.pi/2,np.pi/3])
print(np.cosh(f))'''

#Creating sets in numpy- unique():
'''d = np.array([1,1,1,1,2,3,4,5,5,6])
c = np.unique(d)
print(c)'''

#Finding union in numpy- union1d():
'''f = np.array([1,2,3,2,1])
h = np.array([3,4,5,2,4])

arr = np.union1d(f,h)
print(arr)'''

#Finding intersection in numpy- intersection1d():
'''t = np.array([1,2,3,4])
u = np.array([1,2,3,5])
print(np.intersect1d(t,u, assume_unique=True))'''

#Difference in two sets in numpy- difference()
'''r = np.array([1,2,3,4])
f = np.array([2,3,5,6])
print(np.setdiff1d(r,f,assume_unique=True))'''

#Symmetric difference in numpy -setxor1d():
'''f = np.array([1,2,3,4])
g = np.array([2,3,4,5])
print(np.setxor1d(f,g, assume_unique=True))'''












 


