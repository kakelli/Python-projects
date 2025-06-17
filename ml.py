import matplotlib.pyplot as plt
import numpy as np

x = ["German Shepheard", "Labrador", "Golden Retriever", "Pomerian"]
y = np.array([50,15,25,10])
a = [0,0.1,0,0]

plt.pie(y, labels=x, startangle=90,explode=a,shadow=True)
plt.legend()
plt.show()