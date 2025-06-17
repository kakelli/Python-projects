'''print("Enter your name: ")
name = input()
print(f"Hello there {name}, nice to meet you!!")'''# This method is called String formatting

#Main method
'''it = int(input("Enter the name: "))
if it is int:
    print("This is integer")
else:
    print("Enter the integer")'''

#Python File Handling
#Using file object, the default way
'''f = open("/home/johnbright/Desktop/Sample.txt", 'r')
k = f.readline()
s = k.split()
for i in s:
    print(k)
f.close()'''

#Using with stmt
'''with open("/home/johnbright/Desktop/Sample.txt", "r") as f:
    k = f.read()
    s = k.split()
    for i in s:
        print(i)'''


#Deleting existing content
'''with open("/home/johnbright/Desktop/Sample.txt", 'w') as f:
    f.write("The content has been overwritten!!")

#Reading the content
with open("/home/johnbright/Desktop/Sample.txt", 'r') as f:
    k = f.read()
    print(k)'''

#Creating a file
#f = open("/home/johnbright/Desktop/newprogram.txt", 'x')

'''import os
if os.path.exists("/home/johnbright/Desktop/newprogram.txt"):
    print("This file exists")
    yn = input("Do you want to remove this file(yes/no)? ")
    if yn.lower() == "yes":
        os.remove("/home/johnbright/Desktop/newprogram.txt")
    else:
        print("Alright!!")
    
else:
    print("File not found")'''

#Delete a directory(only empty ones)
'''import os
os.rmdir("/home/johnbright/Desktop/to_do")'''