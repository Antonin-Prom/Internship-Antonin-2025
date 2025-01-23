import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple, List, Optional, Union

class Order_param:
    def __init__(self,angle_trajs):
        self.angles = angle_trajs

    def compute_order(self):
        t_max = len(self.angles[0])
        for i in self.angles:
            continue #finish later
