import random
z = random.randint(1,4)
n = input("Enter your prediction: ")
if z == 1:
    print("PSG wins")
    if n.lower == "PSG":
        print("Your prediction works")
elif z == 2:
    print("Barcelona wins")
    if n.lower == "Barcelona":
        print("Your prediction works")
elif z == 3:
    print("Arsenal wins")
    if n.lower == "Arsenal":
        print("Your prediction works")
elif z == 4:
    print("Inter Milan wins")
    if n.lower == "Inter Milan":
        print("Your prediction works")

