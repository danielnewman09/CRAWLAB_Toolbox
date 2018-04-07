#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
#------------------------------------------------------------------------------
# Specified Insensitivity Input Shaping - si_shaping.py
#
# 
#------------------------------------------------------------------------------
"""
import os

import numpy as np
import warnings
import pdb
import matplotlib.pyplot as plt
from scipy import optimize

# Let's also improve the printing of NumPy arrays.
np.set_printoptions(suppress=True, formatter={'float': '{: 0.4f}'.format})

# Define a few constants
HZ_TO_RADS = 2.0 * np.pi

#------ Optimized Shapers ---------------------------------------------------------
class Optimized_Shaper(Shaper):
    '''
    This abstract class contains all of the methods for 
    generating an input shaper which must be optimized

    freq_args is a mandatory list of inputs to create any
    optimized shaper

    ic_args is an optional list of inputs which can be used
    to create an initial condition input shaper

    Inputs:
        freq_args:  

            X0 - Initial guess
            f_min - minimum suppressed frequency
            f_max - maximum suppressed frequency
            Vtol - tolerable level of vibration
            zeta - damping ratio
        ic_args:
            x0 - initial position
            v0 - initial velocity
            tacc - acceleration time for the system
            Vmax - maximum velocity or "step size"
            sign - direction in which the system is traveling
            design_freq - frequency for which the shaper was designed
            shifted - boolean indicating whether to account for phase and amp shifts


    Methods:
        __init__:
            establish all of the baseline parameters for the input shaper
            and call the "solve_for_shaper" method
        objective:
            return the time of the final impulse of the current input shaper
        objective_deriv:
            return the derivative of the objective funciton
        amp_sum:
            return the sum of the amplitudes. Used to form constraint

    Abstract Methods:
        vib: 
            Create the cosine and sine terms of the vibration equation 
        form_constraints:
            Create the basic input shaping constraints
        solve_for_shaper:


        
    '''
    def __init__(self,freq_args,
                 tol=1e-6,eps=1e-9):

        X0,f_min,f_max,Vtol,zeta = freq_args

        # Variables for the SLSQP optimization routine
        self.tol = tol
        self.eps = eps

        # Determine median frequency of range to be suppressed
        # and Insensitivity.
        self.wmin = 2 * np.pi * f_min                   # convert to rad/s
        self.wmax = 2 * np.pi * f_max                   # convert to rad/s
        self.wn = (self.wmin + self.wmax) / 2.0         # median frequency (rad/s)
        self.f = self.wn / (2.0 * np.pi)                # median frequency (Hz)
        self.Ins = (self.wmax - self.wmin) / self.wn    # Insensitivity

        # number of frequencies which will be evaluated
        self.num_points = 50
        
        # These are good parameters to keep track of regardless of shaper type
        self.num_impulses = 0
        self.bnds = ()
        self.consts = ()
        self.seeking_solution = True

        # Calculate period use for initial time guess 
        self.tau = 2 * np.pi/self.wn

        self.res, self.shaper = self.solve_for_shaper(freq_args)

    def objective(self,x,*args):
        """ Optimization objective function to minimize - here is min(tn) """
        return np.amax(x[0:self.num_impulses])**2

    def objective_deriv(self, x, *args):
        """ Objective Function derivative """

        deriv = np.zeros_like(x)
        deriv[self.num_impulses-1] = 2.0*np.amax(x[0:self.num_impulses])
        return deriv

    def amp_sum(self, x, *args):
        """ Function to set up the sum of the impulses constraint """

        return np.sum(x[self.num_impulses:])

    def vib(self, x, freq_args):
        """ Function to calculate the vibration from a series of impulses over a 
        range of frequencies.
        
        Inputs: 
            x = [Ai ti] array, where Ai and ti are ith impulse amplitude and time
            f_min = the minimum frequency to limit vibration at (Hz)
            f_max = the maximum frequency to limit vibration at (Hz)
            zeta = damping ratio
            Vtol = the tolerable level of vibration (0.05 = 5%)
        
        Returns:
            vib = prop. vibration at num_points number of points in the range of freq.
        """
        X0,f_min,f_max,Vtol,zeta = freq_args

        x = np.asarray(x)

        num_impulses = int(np.round(len(x)/2))

        vib = np.zeros(self.num_points,)

        for ii, freq in enumerate(np.linspace(f_min * HZ_TO_RADS, f_max * HZ_TO_RADS, self.num_points)):
            wd = freq * np.sqrt(1 - zeta**2)

            cos_term = np.sum(x[num_impulses:] * np.exp(zeta*freq*x[0:num_impulses])\
                              * np.cos(wd*x[0:num_impulses]))
            sin_term = np.sum(x[num_impulses:] * np.exp(zeta*freq*x[0:num_impulses])\
                              * np.sin(wd*x[0:num_impulses]))

            # Return the evaluated vibration
            vib[ii] = np.exp(-zeta * freq * x[num_impulses-1]) * np.sqrt((cos_term)**2 + (sin_term)**2)

        return vib

    def form_constraints(self,freq_args):
        """ Function to define the problem constraints """
        X0,f_min,f_max,Vtol,zeta = freq_args

        # Define the contraints for vibration, amplitude-sum, and t_1=0
        self.consts += ({'type': 'ineq',
                'fun': lambda x: Vtol - self.vib(x, freq_args)},
                {'type':'eq',
                'fun': lambda x: self.amp_sum(x) - 1.0, # sum(Ai) = 1
                'jac': lambda x: np.hstack((np.zeros((1,self.num_impulses)), np.ones((1,self.num_impulses))))},
                {'type':'eq',
                'fun': lambda x: x[0]}, # t1 = 0}
                )

    def solve_for_shaper(self,freq_args):
        """ Perform all necessary functions to solve for the optimized shaper"""

        X0,f_min,f_max,Vtol,zeta = freq_args

        # Form the generic constraints
        self.form_constraints(freq_args)

        # Generate an initial guess
        X0 = self.initial_guess(freq_args)

        # Perform the optimization
        res = self.optimize(X0)

        return res, self.shaper

    def optimize(self,X0):
        """ Perform the optimization"""

        # Call the optimization routine
        res = optimize.minimize(self.objective, X0, jac = self.objective_deriv, bounds = self.bnds,
                            constraints = self.consts, method='SLSQP', tol = self.tol, 
                            options={'maxiter':4e2, 'eps':self.eps, 'disp':True})

        # Check for repeated times before returning the shaper
        if res.success:
            
            shaper = np.reshape(res.x,(2,self.num_impulses)).T
            shaper = shaper[shaper[:,0].argsort()]
   
            times = []
            amps = []

            tms_X0 = shaper[:,0]
            amps_X0 = shaper[:,1]

            # check for impulses occuring at the same time 
            # if so, remove one, sum amplitudes, and resolve
            times = np.append(times, tms_X0[0])
            amps = np.append(amps, amps_X0[0])

            for ii in range(1, self.num_impulses):

                if np.abs(tms_X0[ii] - tms_X0[ii-1]) < 1e-4:
                    print('\nRepeated Times. Shortening and resolving...')
                    amps[-1] = amps[-1] + amps_X0[ii]
                else:
                    times = np.append(times, tms_X0[ii])
                    amps = np.append(amps, amps_X0[ii])
            
            # Put the result in standard shaper form
            self.num_impulses = len(amps)
            self.amps = amps.reshape(self.num_impulses,1)
            self.times = times.reshape(self.num_impulses,1)
            self.shaper = np.hstack((self.times,self.amps))
    
        else:
            shaper = []
            print('\nOptimization failed.\n')
            print('Possible Solutions:')
            print('  * Improve your initial guess. Options include:')
            print('    - Use the "closest" closed-form shaper for the initial guess.')
            print('    - Solve for a "nearby" point and use it as the initial guess.')
            print('  * Try a slightly different Insensitivity range.')
            print('  * Try a slightly different damping ratio.')
            print('  * Normalize the range of frequencies by the midpoint.')
            print('\nAs of 12/22/14, solutions nearest to EI-form shapers work best ')
            print('  when no initial guess is given.')
        
        return res

class Positive_SI(Optimized_Shaper):
    isPositive = True

    #def __init__(self,freq_args,**kwargs):
    #    super().__init__(freq_args,**kwargs)

    def form_bounds(self):
        for ii in range(self.num_impulses):
            self.bnds += ((0.0, 3*self.tau),)

        # Create the bounds
        for ii in range(self.num_impulses):
            # create bounds on impulse ampliutdes of 0 < A_i < 1
            self.bnds += ((0.0, 1.0),)

    def initial_guess(self,freq_args):
        """ Generate an initial guess to begin the optimization"""

        X0,f_min,f_max,Vtol,zeta = freq_args

        # Use the nearest closed-form shaper if no initial guess is supplied
        if X0 is None:
            if self.Ins <= 0.06:         # ZV     
                zv = ZV(self.f, zeta)
                X0 = np.hstack((zv.times, zv.amps))
            elif self.Ins <= 0.3992:     # EI  
                ei = EI(self.f, zeta, Vtol)
                X0 = np.hstack((ei.times, ei.amps))
            elif self.Ins <= 0.7262:     # Two-hump EI
                ei2hump = EI2HUMP(self.f, zeta, Vtol)
                X0 = np.hstack((ei2hump.times, ei2hump.amps))
            elif self.Ins <= 0.9654:     # Three-hump EI
                ei3hump = EI3HUMP(self.f, zeta, Vtol)
                X0 = np.hstack((ei3hump.times, ei3hump.amps))
            else:
                raise ValueError('Code only works (as of 12/26/14) for positive shapers up to I(5%) = 0.9654.')

        self.num_impulses = int(np.round(len(X0) / 2))

        return X0

class UM_SI(Optimized_Shaper):
    isPositive = False

    def __init__(self,freq_args,**kwargs):
        X0,f_min,f_max,Vtol,zeta = freq_args
        self.amp_tol = 1e-5
        super().__init__(freq_args,**kwargs)

    def form_constraints(freq_args):
        super().form_constraints(freq_args)

    def cum_sum(self,x):
        """Function to determine the running sum of the shaper at every point"""

        shaper = np.reshape(x,(2,self.num_impulses)).T
        shaper = shaper[shaper[:,0].argsort()]
        
        return np.abs(np.cumsum(shaper[0:self.num_impulses,1]))  
        
    def solve_for_shaper(self,freq_args):
        X0,f_min,f_max,Vtol,zeta = freq_args

        for ii in range(self.num_impulses):
            self.bnds += ((0.0, 3*self.tau),)

        # Unity Magnitude Shaper
        for ii in range(self.num_impulses):
            # create bounds on impulse ampliutdes of +/-1
            self.bnds += (((-1)**ii - self.amp_tol, (-1)**ii + self.amp_tol),)

        self.consts += (
            # Enforce the cumulative sum constraint
            {'type': 'ineq',
            'fun': lambda x: 1 - self.cum_sum(x)},
            )

        form_constraints(freq_args)

        X0 = self.optimization_setup(freq_args)
        res = super().optimize(X0)

        return res, self.shaper

    def initial_guess(self,freq_args):
        X0,f_min,f_max,Vtol,zeta = freq_args

        if X0 is None:
            print('As of 12/22/14, UM shaper solution is extremely sensitive to initial guess.')
            print('So, you may want to supply one, rather than use the default.\n') 
            if self.Ins <= 0.0333 - 0.0672 + 0.3956: # UM-EI
                umei = UMEI(self.f, zeta, Vtol)
                X0 = np.hstack((umei.times, umei.amps))
            elif self.Ins <= 0.0604 - 0.1061 + 0.7186: # UM-Two-Hump EI
                um2ei = UM2EI(self.f, zeta, Vtol)
                X0 = np.hstack((um2ei.times, um2ei.amps))
            elif Ins <= .2895 - 0.6258 + 0.5211 - 0.2382 + 0.9654: # UM-Three-Hump EI
                um3ei = UM3EI(self.f, zeta, Vtol)
                X0 = np.hstack((um3ei.times, um3ei.amps))
            else:
                raise ValueError('Code only works (as of 02/16/15) for negative shapers up to I(5%) = 0.912')
                self.seeking_solution = False

        self.num_impulses = int(np.round(len(X0) / 2))

        return X0