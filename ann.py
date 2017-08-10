import numpy as np;
import sklearn.datasets as  dt;
import matplotlib.pyplot as plt;
np.random.seed(0)
X, y = dt.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show();