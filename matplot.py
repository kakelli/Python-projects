'''import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A","B","C","D"])
y = np.array([2,4,6,7])

plt.barh(x,y, height=0.1)
plt.plot()
plt.show()'''

'''import numpy as np
import matplotlib.pyplot as plt

rn = np.random.normal(120,10,100)
plt.hist(rn)
plt.show()'''

import matplotlib.pyplot as plt
import numpy as np

x = np.array([35,15,15,45])
thelabels = ["Bugatti", "Pagani", "Rolls Royce", "Lamborgini"]

plt.pie(x,labels=thelabels)
plt.legend(title = "Four cars: ")
plt.show()