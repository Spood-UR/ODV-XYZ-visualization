import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from find_files_with_extension import find_file_with_extension
from import_xyz import read_xyz_file_r2


# It is my trial to create a class that reads Ocean Data View Surface
# griding data (clipboard copy) and allows re-create that grid and convert it to NetCDF file.

# August 2021
# Author: Ivan Dudkov
# Affiliations:
# P.P Shirshov's Institute of Oceanography, Atlantic Branch, Kaliningrad, Russia. (2018 - ????)
# Center for Coastal and Ocean Mapping, University of New Hampshire. Durham, USA. (2020-2021)

# A more than half of that code is from RegGrid3D class from my "dynamic-surface-class"
# GitHub repository, which is actually a code converted from Matlab script. That script
# originally was written by Semme Dijkstra from UNH, Center for Coastal and Ocean Mapping, USA
# in 2017. Thank you a lot, Semme!

class odv_vis:
    """
    Handling the ODV XYZ data
    """

    def __init__(self, x, y, z, rinfl=None, odv_nan_value=-10000000000.00000, path=os.getcwd()):
        nan_value = odv_nan_value
        x = x - 360
        z[z == nan_value] = np.nan

        x_diff = []
        y_diff = []
        for i in range(len(x)):
            if i > 0:
                x_diff.append(x[i - 1] - x[i])
                y_diff.append(y[i - 1] - y[i])
        self.x_res = np.abs(np.amin(x_diff))
        self.y_res = np.abs(np.amin(y_diff))

        if rinfl is None:
            self.x_rinfl = self.x_res
            self.y_rinfl = self.y_res
        self.path = path
        self.x_rpix = round(self.x_rinfl / self.x_res)
        self.y_rpix = round(self.y_rinfl / self.y_res)

        self.x_rpix2coord = np.sum(np.arange(0, self.x_rpix + 1) * self.x_res)
        self.y_rpix2coord = np.sum(np.arange(0, self.y_rpix + 1) * self.y_res)

        self.rangeX = np.zeros(2)
        self.rangeY = np.zeros(2)

        self.X = None  # X-coord
        self.Y = None  # Y-coord
        self.weighGrid = None  # Weights Grid
        self.sumWeight = None  # Grid of the sum Weights
        self.Z = None  # Z-values

        # Create a distance weighting kernel that weighs by 1/R**2,
        # except for R=0, for which the weight = 1
        wx, wy = np.meshgrid(np.arange(-self.x_rpix, self.x_rpix + 1),
                             np.arange(-self.y_rpix, self.y_rpix + 1))

        # Create a kernel weighs grid
        np.seterr(divide='ignore')  # ignore zero-division warn
        self.kWeight = 1 / (wx ** 2 + wy ** 2)
        np.seterr(divide='warn')  # set back to default
        # Deal with the weight at the center i.e., R = 0
        self.kWeight[self.x_rpix, self.y_rpix] = 1
        self.kWeight = \
            np.where(self.kWeight < self.x_rpix ** -2, 0, self.kWeight)
        self.kWeight = \
            np.where(self.kWeight < self.y_rpix ** -2, 0, self.kWeight)

        self.rangeX[0] = np.amin(x) - self.x_rpix2coord
        self.rangeX[1] = np.amax(x) + self.x_rpix2coord
        self.rangeY[0] = np.amin(y) - self.y_rpix2coord
        self.rangeY[1] = np.amax(y) + self.y_rpix2coord

        self.X, self.Y = \
            np.meshgrid(
                np.arange(0, (np.ceil(np.diff(self.rangeX)[0] / self.x_res)) + 1)
                * self.x_res + self.rangeX[0],
                np.arange(0, (np.ceil(np.diff(self.rangeY)[0] / self.y_res)) + 1)
                * self.y_res + self.rangeY[0])

        self.sumWeight = np.zeros(np.shape(self.X))
        self.weighGrid = np.zeros(np.shape(self.X))

        for i in range(len(z)):
            # Get the location of the data in the grid
            x_grid = np.argwhere(x[i] >= self.X[0, :])[-1][0]
            y_grid = np.argwhere(y[i] >= self.Y[:, 0])[-1][0]

            # Set the location of associated kernel in the grid
            k = np.array([[x_grid - self.x_rpix, x_grid + (self.x_rpix + 1)],
                          [y_grid - self.y_rpix, y_grid + (self.y_rpix + 1)]])

            # Add the contribution to both the weighed grid as well as
            # the grid of summed weights
            self.sumWeight[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1]] = \
                self.sumWeight[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1]] + \
                self.kWeight

            self.weighGrid[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1]] = \
                self.weighGrid[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1]] + \
                self.kWeight * z[i]

        self.Z = self.weighGrid / self.sumWeight

    def save_to_nc(self, filename="filename.nc"):
        # Create new nc-file
        ds = Dataset(os.path.join(self.path, filename), "w", format="NETCDF3_64BIT_OFFSET")
        ds.set_fill_off()
        # Creating dimensions
        lat = ds.createDimension("lat", np.shape(self.X)[0])
        lon = ds.createDimension("lon", np.shape(self.Y)[1])

        # Creating variables
        longitudes = ds.createVariable("lon", "f8", ("lon",))
        longitudes.long_name = 'Longitude'
        longitudes.standard_name = 'longitude'
        longitudes.units = 'degrees_east'

        latitudes = ds.createVariable("lat", "f8", ("lat",))
        latitudes.long_name = 'Latitude'
        latitudes.standard_name = 'latitude'
        latitudes.units = 'degrees_north'

        z_values = ds.createVariable("val", "f4", ("lat", "lon",))
        z_values.long_name = 'indf'
        z_values.standard_name = 'indf'
        z_values.units = 'indf'

        # Passing data into variables
        longitudes[:] = self.X[0, :]
        latitudes[:] = self.Y[:, 0]
        z_values[:] = self.Z[:]

        print(ds)
        print(ds.variables)

        # Close .nc file creation
        ds.close

    def plot(self):
        # Original description from Matlab script
        #         % This is certainly not the fastest way to plot the data (that
        #         % would be using the surface functions) but it does allow for
        #         % easier manipulation of the data - note that adding lighting
        #         % to this would be nice and slick - this will not be hard to do
        #         dtm = (self.weighGrid / self.sumWeight)*exag

        # Set up some common objects for plotting
        x_label = "Easting [m]"
        y_label = "Northing [m]"
        title1 = "ODV Griding"

        # 2d plots
        fig, ax = plt.subplots(1, 1)
        fig.set_figheight(5)
        fig.set_figwidth(10)

        # Digital Terrain Model 2D surface
        plot1 = ax.pcolormesh(self.X, self.Y, self.Z,
                              cmap='gist_rainbow_r', vmin=np.nanmin(self.Z), vmax=np.nanmax(self.Z))

        ax.ticklabel_format(useOffset=False, style='plain')
        ax.set_title(title1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Add colorbars for each plot
        fig.colorbar(plot1, ax=ax)

        # Tight figure's layout
        fig.tight_layout()
        plt.show()
