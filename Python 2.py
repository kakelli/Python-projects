#x = "My favortie programming language and My favortie subject is CS  "
#print(x[3])
#print(len(x))
#for i in x:
#    print(i , end = " ")
#print(chr(80))
#print(x[1:9:3])
#print(x.capitalize())
#print(x.find('are'))
#print(x.isalnum())
#print(x.isalpha())
#print(x.isdigit())
#print(x.isspace())
#print(x.islower())
#print(x.isupper())
#print(x.istitle())
#print(x.lower())
#print(x.upper())
#print(x.title())
#print(x.startswith('My'))
#print(x.endswith('language'))
#print(x.swapcase())
#print(x.partition('favortie'))
#print(x.count('My'))
#print(x.lstrip())
#print(x.rstrip())

#l  = [10,20,30,40,50,40]
'''for i in l:
    print(i)'''
#print(l*2)
#print(l[::2])
'''l[1:2] = [30,20]
print(l)'''
'''l.append(50)
print(l)'''

'''l[2] = 15
print(l)'''

'''del l
print(l)'''  

'''l1 = list(l)
print(l1)'''

#print(l.index(30))
'''l.append(50)
print(l)'''

'''l2 = [0,5,10]
l.extend(l2)
print(l)'''

'''l.insert(2,15)
print(l)'''

'''l.pop(2)
print(l)'''
'''l.remove(40)
print(l)'''

'''l.clear()
print(l)'''

#print(l.count(40))

'''l.reverse()
print(l)'''

'''l.sort(reverse=True)
print(l)'''

#t = (1,2,3,4,5)
'''for i in t:
    print(i)'''

'''t1 = (4,5,6,7)
print(t+t1)'''

#print(t*3)
#print(t[1:4:2])

'''a,b,c,d,e = t
print(a,b,c,d,e, sep = ' ~ ')'''

#print(len(t))

'''print(max(t))
print(min(t))'''

#print(t.index(4))

#d = {1:'a', 2:'b', 3:'c',4:'d'}

#print(d[3])
#print(d.keys())
#print(d.values())

'''for i in d:
    print(i, ':', d[i])'''
'''d[2] = "j"
print(d)'''

'''d[5] = 'e'
print(d)'''

'''del d[2]
print(d)'''

'''d.pop(4)
print(d)'''

##print( 2 in d)

#print(len(d))
'''d.clear()
print(d)'''
'''print(d.get(3))
print(d.items())'''

l = [1,3,4,5,31,3,4,2]
n = len(l)
for i in range (n):
    for j in range (0,n-1-i):
        if l[j] > l[j+1]:
            l[j],l[j+1] = l[j+1], l[j]
print(l)
