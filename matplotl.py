import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#Basic plot
'''x = [1,2,3,4,5]
y = [10,20,30,40,50]

plt.plot(x,y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Basic plot")
plt.show()'''

#Line, marker, and color
'''x = [1,2,3,4,5]
y = [10,20,30,40,50]

plt.plot(x,y, marker='o', color = 'r', linestyle = '--')
plt.show()'''

#Multiple lines
'''x = [1,2,3,4,5]
y = [10,20,30,40,50]

a = [12,13,14,15]
b = [20,30,40,50]

plt.plot(x,y, label = 'Line 1')
plt.plot(a,b, label  = 'Line 2')

plt.show()'''

#Bar chart
'''x =  ['A', 'B', 'C','D','E']#categories
y = [12,13,14,15,16]#VALUES

plt.bar(x,y)
plt.title("Simple Bar Chart")
plt.show()'''

#Histogram
'''data = [1,2,3,4,5,6,3,2,4,2,6,7,8,9,10,5]
plt.hist(data, bins=5)
plt.title("Histogram")
plt.show()'''

#Scatter plot
'''x = [1,2,3,4,5]
y = [5,4,3,2,1]

plt.scatter(x,y)
plt.title("Scatter Plot")
plt.show()'''

#Pie Chart
'''sizes = [40,30,20,10]
labels = ['A', 'B','C','D']

plt.pie(sizes, labels=labels, autopct='%1.1f%%') #autopct to show percentage
plt.title("Pie Chart")
plt.show()'''

#Subplots
'''x = [1,2,3,4,5]
y = [10,20,30,40,50]
fig, axs = plt.subplots(2,2)

axs[1,1].plot(x,y)
axs[0,0].bar(x, y)
axs[0,1].scatter(x, y)
axs[1,0].hist(y, bins=5)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()'''

#Customizing plot appearance
'''x = [1,2,3,4,5]
y = [10,20,30,40,50]

plt.plot(x, y, marker='s', color='purple',  linewidth=2, markersize=8)
plt.grid(True)  # Add grid

plt.xlim   (0, 6)  # Set x-axis limits
plt.ylim(0, 60)  # Set y-axis limits

plt.show()'''

#Adding annotations
'''x = [1, 2, 3, 4, 5]
y = [10, 20, 30, 40, 50]

plt.plot(x,y)
plt.annotate('Peak', xy=(5, 50), xytext=(4, 40),arrowprops=dict(facecolor = 'black'))
plt.show()'''

#Saving plots
'''x = [1, 2, 3, 4, 5]
y = [10, 20, 30, 40, 50]

plt.plot(x,y)
plt.savefig("plot1.png", dpi = 300)  # Save with high resolution'''

#Using object oriented approach
'''fig, ax = plt.subplots()  # Create a figure and axis object
x = [1, 2, 3, 4, 5]
y = [10, 20, 30, 40, 50]

ax.set_title("Object Oriented Approach")    
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis") 

ax.plot(x,y)
plt.show()'''

#3D plotting

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-5, 10, 100)
y = np.linspace(-5, 10, 100)
X,Y = np.meshgrid(x,y)
Z =np.cos  (np.sqrt(X**2 + Y**2))

ax.plot_surface(X, Y, Z, cmap='viridis')
plt.show()