'''f = open(r'/home/johnbright/Desktop/Sample.txt', 'r')
s= f.read(69)
print(s)
print(f.seek(3,0))
f.close()'''

'''n = int(input("ENter how many time do you need to enter? "))
for i in range (n):
    x = input("Write anything in this file: ")
    s = f.write(x)
    f.write('\n')
f.close()

with open(r'/home/johnbright/Desktop/Sample.txt', 'r') as f:
    s = f.read()
    print(s)'''

#import pickle
#f = open(r'/home/johnbright/Desktop/binary.dat', 'rb')
'''n = int(input('ENter how many times to enetr? '))
for i in range (n):
    x = input("Entre the content: ")
    s = list(x)
    pickle.dump(s,f)
f.close()'''
'''try:
    s = pickle.load(f)
    print(s)
    k = pickle.load(f)
    print(k)
except EOFError:
    f.close()'''
'''with open(r'/home/johnbright/Desktop/binary.dat', 'rb') as f:
    s = pickle.load(f)
    print(s)
    k = pickle.load(f)
    print(k)
    f.tell()'''

import csv
try:
    f = open(r'/home/johnbright/Desktop/ComSepVal.csv', 'r', newline = '\r\n')
    s = csv.reader(f, delimiter='|')
    for i in s:
        print(i)
except:
    f.close()

'''s = csv.writer(f, delimiter='|')
n = int(input("ENter the number of times? "))
for i in range(n):
    x = input("ENter the content: ")
    l = tuple(x)
    k = s.writerow(l)
f.close()'''
