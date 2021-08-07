import os
import numpy as np

from find_files_with_extension import find_file_with_extension
from import_xyz import read_xyz_file_r2



class odv_vis:
    """
    Handling the ODV XYZ data
    """

    def __init__(self, path=os.getcwd()):

        self.path = path
        self.y = np.empty()
        self.x = np.empty()
        self.z = np.empty()








