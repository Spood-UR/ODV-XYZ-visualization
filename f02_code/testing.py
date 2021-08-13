import os
import numpy as np
from find_files_with_extension import find_file_with_extension
from import_xyz import read_xyz_file_r2
from odv_vis_class import odv_vis

paths = find_file_with_extension(".txt")

xyz_array = read_xyz_file_r2(paths[0], header_length=1, separator="	")

# Resolution in geographical degrees
odv_grid = odv_vis(xyz_array[0], xyz_array[1], xyz_array[2])

#odv_grid.plot()

odv_grid.save_to_nc()
