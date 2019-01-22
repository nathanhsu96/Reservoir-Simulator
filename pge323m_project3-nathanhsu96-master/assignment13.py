
# coding: utf-8

# # Homework Assignment 13
# 
# ## Instructions
# 
# Consider the reservoir shown below with the given properties that has been discretized into unequal grid blocks.
# 
# ![image](images/grid.png)
# 
# Extending the code you wrote for [Assignment 12](https://github.com/PGE323M-Fall2018/assignment12), add functionality to the functions `compute_transmissibility()` and `compute_accumulation()` such that they will compute interblock transmissiblities for grid blocks that could potentially differ in permeability, grid spacing, and area, as well as accumulation terms that can vary from grid block to grid block in terms of volumes and porosities.  This means that now `compute_transimilibility(i, j)` should be a function of two arguments where `i` and `j` refer to  the grid block numbers which you want to compute the transmissibility between and `compute_accumulation(i)` should also take an argument indicating the grid block number.
# 
# You should include logic in your code that checks whether the inputs given for permeability, porosity, etc. are lists (corresponding to an independent value for each grid block) or they are single valued (corresponding to constant for the entire reservoir) and handle those situations accordingly.
# 
# Use these modified functions to construct the matrices and vectors needed in the `solve_one_step()` method.  Be sure to handle boundary conditions correctly and in a way that is general enough that the boundary conditions can be change (there are tests for boundary conditions that are different from the ones shown this time).
# 
# You may continue this implementation from your `assignment12.ipynb` submission, but you must rename the file to
# `assignment13.ipynb` and add it to this git repository before submission. 
# 

# In[1]:


import numpy as np
import yaml
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt


# In[2]:


class OneDimReservoir():
    
    def __init__(self, inputs):
        '''
            Class for solving one-dimensional reservoir problems with
            finite differences.
        '''
        
        #stores input dictionary as class attribute
        if type(inputs) == str:
            with open(inputs) as f:
                self.inputs = yaml.load(f)
        else:
            self.inputs = inputs
        
        #assigns class attributes from input data
        self.parse_inputs()
        
        #calls fill matrix method (must be completely implemented to work)
        self.fill_matrices()
        
        #applies the initial reservoir pressues to self.p
        self.apply_initial_conditions()
        
        #create an empty list for storing data if plots are requested
        if 'plots' in self.inputs:
            self.p_plot = []
            
        return
    
    def parse_inputs(self):
        
        self.viscosity = self.inputs['fluid']['water']['viscosity']
        self.formation_volume_factor = self.inputs['fluid']['water']['formation volume factor']
        self.compressibility = self.inputs['fluid']['water']['compressibility'] 
        self.ngrids = self.inputs['numerical']['number of grids']
        self.delta_t = self.inputs['numerical']['time step']
        
        #Read in 'unit conversion factor' if it exists in the input deck, 
        #otherwise set it to 1.0
        if 'conversion factor' in self.inputs:
            self.conversion_factor = self.inputs['conversion factor']
        else:
            self.conversion_factor = 1.0
            
        
        phi = self.inputs['reservoir']['porosity']
        k = self.inputs['reservoir']['permeability']
        A = self.inputs['reservoir']['cross sectional area']
        
        self.permeability = self.check_input_and_return_data(k)
        self.area = self.check_input_and_return_data(A)
        self.porosity = self.check_input_and_return_data(phi)
        
        #computes delta_x
        self.delta_x = self.assign_delta_x_array()
        
    
    def assign_delta_x_array(self):
        ngrids = self.ngrids

        if 'delta x' not in self.inputs['numerical']:
            length = self.inputs['reservoir']['length']
            delta_x = np.float(length) / ngrids
            delta_x_arr = np.ones(ngrids) * delta_x
        else:
            delta_x_arr = np.array(self.inputs['numerical']['delta x'], 
                              dtype=np.double)

            length_delta_x_arr = delta_x_arr.shape[0]
            
            #makes sure dx array matches number of grids
            assert length_delta_x_arr == ngrids, ("User defined 'delta x' array doesn't match 'number of grids'")

        return delta_x_arr
    
    
    def check_input_and_return_data(self, input_name):

        if isinstance(input_name, str):
            filename = input_name
            data = np.loadtxt(filename, dtype=np.double)
            
        elif isinstance(input_name, (list, tuple)):
            data = np.array(input_name, dtype=np.double)

        else:
            ngrids = self.inputs['numerical']['number of grids']
            data = (input_name *  np.ones(ngrids))
            
        return data
    
    
    def compute_transmissibility(self, i, j):
        '''
            Computes the transmissibility.
        '''
        
        mu = self.viscosity
        k = self.permeability
        A = self.area
        B_alpha = self.formation_volume_factor
        dx = self.delta_x
        
        kA_half = 2 * k[i] * A[i] * k[j] * A[j] / (k[i] * A[i] * dx[j] + k[j] * A[j] * dx[i]) 
        transmissibility = kA_half / mu / B_alpha
        
        return transmissibility
    
    
    
    def compute_accumulation(self, i):
        '''
            Computes the accumulation.
        '''
        
        c_t = self.compressibility
        phi = self.porosity
        B_alpha = self.formation_volume_factor
        
        A = self.area
        dx = self.delta_x
        
        accumulation = A[i] * dx[i] * phi[i] * c_t / B_alpha
        
        return accumulation 
    
    
    def fill_matrices(self):
        '''
            Fills the matrices A, I, and \vec{p}_B and applies boundary
            conditions.
        '''    
        
        N = self.ngrids
        
        #Complete implementation here
        
        factor = self.conversion_factor
        T = scipy.sparse.lil_matrix((N, N), dtype=np.double)
        B = np.zeros(N, dtype=np.double)
        Q = np.zeros(N, dtype=np.double)

        bcs = self.inputs['boundary conditions']
        bc_type_1 = bcs['left']['type'].lower()
        bc_type_2 = bcs['right']['type'].lower()
        bc_value_1 = bcs['left']['value']
        bc_value_2 = bcs['right']['value']
      
        for i in range(N):

            # Left BC
            if i == 0:
                T[i, i+1] = -self.compute_transmissibility(i, i + 1)

                if bc_type_1 == 'prescribed flux':
                    T[i, i] = T[i,i] - T[i, i+1]
                elif bc_type_1 == 'prescribed pressure':
                    #Computes the transmissibility of the ith block
                    T0 = self.compute_transmissibility(i, i)
                    T[i, i] = T[i,i] - T[i, i+1] + 2.0 * T0
                    Q[i] = 2.0 * T0 * bc_value_1 * factor
                else:
                    raise ValueError('No left boundary condition prescribed!')

            # Right BC
            elif i == (N - 1):
                T[i, i-1] = -self.compute_transmissibility(i, i - 1)

                if bc_type_2 == 'prescribed flux':
                    T[i, i] = T[i,i] - T[i, i-1]
                elif bc_type_2 == 'prescribed pressure':
                    #Computes the transmissibility of the ith block
                    T0 = self.compute_transmissibility(i, i)
                    T[i, i] = T[i, i] - T[i, i-1] + 2.0 * T0
                    Q[i] = 2.0 * T0 * bc_value_2 * factor
                else:
                    raise ValueError('No right boundary condition prescribed!')

            else:
                T[i, i-1] = -self.compute_transmissibility(i, i-1)
                T[i, i+1] = -self.compute_transmissibility(i, i+1)
                T[i, i] = (self.compute_transmissibility(i, i-1) +
                           self.compute_transmissibility(i, i+1))

            B[i] = self.compute_accumulation(i)


        self.T = T.tocsr() * factor
        self.B = scipy.sparse.csr_matrix((B, (np.arange(N), np.arange(N))), shape=(N,N))
        self.Q = Q
        
        return
        
            
                
    def apply_initial_conditions(self):
        '''
            Applies initial pressures to self.p
        '''
        
        N = self.inputs['numerical']['number of grids']
        
        self.p = np.ones(N) * self.inputs['initial conditions']['pressure']
        
        return
                
                
    def solve_one_step(self):
        '''
            Solve one time step using either the implicit or explicit method
        '''
        
        B = self.B
        T = self.T
        Q = self.Q
        
        dt = self.delta_t
        
        if self.inputs['numerical']['solver'] == 'explicit':
            self.p = self.p + dt * 1. / B.diagonal() * (Q - T.dot(self.p)) 
        elif self.inputs['numerical']['solver'] == 'implicit':
            self.p = scipy.sparse.linalg.cg(T + B / dt, B.dot(self.p) / dt + Q)[0]
        elif 'mixed method' in self.inputs['numerical']['solver']:
            theta = self.inputs['numerical']['solver']['mixed method']['theta']
            if theta > 1 or theta < 0:
                raise ValueError('Theta not between 0 and 1!')
            else:
                self.p = scipy.sparse.linalg.cg((1 - theta) * T + B / dt, (B / dt - theta * T).dot(self.p) + Q)[0]
        else:
            raise ValueError('No numerical solver specified.')
            
        return
            
            
    def solve(self):
        '''
            Solves until "number of time steps"
        '''
        
        for i in range(self.inputs['numerical']['number of time steps']):
            self.solve_one_step()
            
            if i % self.inputs['plots']['frequency'] == 0:
                self.p_plot += [self.get_solution()]
                
        return
                
    def plot(self):
        '''
           Crude plotting function.  Plots pressure as a function of grid block #
        '''
        
        if self.p_plot is not None:
            for i in range(len(self.p_plot)):
                plt.plot(self.p_plot[i])
        
        return
            
    def get_solution(self):
        '''
            Returns solution vector
        '''
        return self.p

