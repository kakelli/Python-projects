import numpy as np 
from numpy.linalg import inv, det, eig

#Creating arrays
'''a = np.array([1,2,3])
print(a) #1D array

b = np.array([[1,2,3], [4,5,6]]) #2D Array
print(b)'''

'''print(np.zeros((2,3))) #Zero matrix'''
'''print(np.ones((2,3))) #One matrix'''
'''print(np.eye(3)) #Identity matrix'''
'''print(np.arange(10)) #Array from 0 to 9'''
'''print(np.linspace(0,1,5)) #5 points between 0 and 1'''

#Basic array Operations
'''a = np.array([1,2,3])
b = np.array([4,5,6])'''

'''print(a+b) #Addition of two arrays'''
'''print(a*b) #Multiplication of two arrays'''
'''print(b/a) #Division of two arrays'''
'''print(a@b) #Dot product'''
'''print(a.mean()) #Mean of array'''
'''print(a.sum()) #Sum of elements'''

#Indexing and slicing
'''a = np.array([[1,2,3],[3,4,5]])'''

'''print(a[0,1]) #Slicing element at row 0, column 1'''
'''print(a[:,1]) #All rows, column 1'''
'''print(a[1, :]) #Row 1, all column'''

#Array Reshaping and transposing
'''a = np.arange(6)
a = a.reshape((2,3))
print(a.T) #Transposing '''

#Boolean Indexing
'''a = np.array([1,2,3,4])
d = a>1
print(a[d]) #[2,3,4]'''

#Random Numbers generating
'''print(np.random.seed(0)) #Repeatable sequences
print(np.random.rand(2,3)) #From [0,1) and shape of array determined'''
'''print(np.random.randn(3,3)) #Normal distribution and shape of array determined'''
'''print(np.random.randint(1,10,(2,3))) #Sequence and shape determined'''

#Axis operation
'''a = np.array([[1,2],[3,4]])'''

'''print(a.sum(axis=0)) #Column wise sum: [4,6]'''
'''print(a.sum(axis=1)) #Row wise sum: [3,7]'''

#Broadcasting
'''a = np.array([[1],[2], [3]]) #3x1
b = np.array([10,20,30]) #1x3

print(a+b) '''

#Vectorization(Avoid loops)
'''a = np.arange(1000)
b = np.sqrt(a) #This is vectorized
print(b)'''

#Instead of b=[math.sqrt(x) for x in a]

#Linear Algerba
'''a = np.array([[1,2],[3,4]])
ia = inv(a) #Inverse of a
da = det(a) #Determinant of a

eigenvalues, eigenvectors = eig(a)
print(a, ia, da, eigenvalues,eigenvectors)'''

#Advanced Array Manipulation
'''a = np.array([1,2,3,4])

print(np.tile(a,2)) #Repeat a twice
print(np.repeat(a,2)) #Repeat a's element twice'''

#Making & Conditional Replacement
'''a = np.array([1,2,3,4])
a[a%2==0] = 0 #Replacing the even numbers of a
print(a)'''

#Other important functions
#1.Limiting  values in array
'''a = np.array([1,2,3,4])
n = np.clip(a , a_min = 3, a_max = 5)
print(a)'''

#2. Limiting to only unique values in array
'''s = np.array([1,2,3,4,2])
d = np.unique(s)
print(d) #Return unique values'''

#3. Joining two arrays
'''d = np.array([1,2,3,4])
f = np.array([5,6,7,8])

f = np.concatenate((d,f))
print(f)'''

#4. Stacking the arrays
s = np.array([1,2,3,4])
f = np.array([2,3,4,5])

'''r = np.hstack((s,f))#Horizintal stacking'''
w = np.vstack((s,f)) #Vertical stacking

print(w)


