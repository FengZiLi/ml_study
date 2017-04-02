import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-5,5,200)
y=x*x+3
figure=plt.figure()
ax=figure.add_subplot(1,1,1)
ax.scatter(x,y)
# plt.ion()
plt.show()
# input()
