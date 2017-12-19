# coding: utf-8

# # Testing ploting grid slice

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


def plot_iam_grid_slices(x, y, z, grid):
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    for ii, y_val in enumerate(y):
        ax = plt.subplot(111)

        cmap = ax.contour(X[:, ii, :], Z[:, ii, :], grid[:, ii, :])
        plt.colorbar(cmap)
        ax.plot
        ax.set_title("grid slice for y={}".format(y_val))
        plt.show()
        # pltname = os.path.join(simulators.paths[], "grid_plots",
        #                      "grid_slice")
        # plt.savefig(".png")

    for jj, z_val in enumerate(z):
        ax = plt.subplot(111)

        cmap = ax.contourf(X[:, :, jj], Y[:, :, jj], grid[:, :, jj])
        plt.colorbar(cmap)
        ax.plot
        ax.set_title("grid slice for z={}".format(z_val))
        plt.show()
        # pltname = os.path.join(simulators.paths[], "grid_plots",
        #                       "grid_slice")
        # print(pltname)
        # plt.savefig(".png")


# In[ ]:


x = np.arange(100)
y = np.arange(15)
z = np.arange(7)

grid = np.random.randn(len(x), len(y), len(z))
print(grid.shape)

# In[ ]:


XX, YY, ZZ = np.meshgrid(x, y, z, indexing="ij")
XX.shape

# In[ ]:


plot_iam_grid_slices(x, y, z, grid)
