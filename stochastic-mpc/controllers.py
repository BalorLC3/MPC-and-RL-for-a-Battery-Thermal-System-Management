import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import scipy
from abc import ABC, abstractmethod

class BaseController(ABC):
    @abstractmethod
    def compute_control(self, state, disturbance):
        pass

class Thermostat(BaseController):
    '''Rule based controller; threshold based'''
    def __init__(self):
        self.cooling_active = False

    def compute_control(self, state, disturbance):
        T_batt, _, _ = state
        T_upper, T_lower = [34.0, 32.5]
        
        self.cooling_active = np.where(
            T_batt > T_upper, 
            True, 
            np.where(T_batt < T_lower, False, self.cooling_active)
        )
        
        w_pump = np.where(self.cooling_active, 2000.0, 0.0)
        w_comp = np.where(self.cooling_active, 3000.0, 0.0)
        
        return w_comp, w_pump
    
        
class SMPC(BaseController):
    def __init__(self):
        self.n_constraints = 5
        T_mins = np.array([30, 28]) # T_batt, T_clnt
        T_maxs = np.array([35, 34])
        w_mins = np.array([0, 0])   # w_comp, w_pump
        w_maxs = np.array([5000, 6000])
        P_batt_min  = np.array([0])
        P_batt_max  = np.array([200])
    
    def optimize(self, state, disturbance, transition_matrix):
        N = len(disturbance)
        T = ca.MX.sym('T', 2)
        W = ca.MX.sym('W', 2)