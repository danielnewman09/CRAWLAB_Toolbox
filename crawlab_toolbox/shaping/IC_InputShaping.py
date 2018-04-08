#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
#------------------------------------------------------------------------------
# Initial Condition Input Shaping Module - IC_InputShaping.py
#
# Python module for initial condition input shaping
#
# Created: 4/4/18 - Daniel Newman - danielnewman09@gmail.com
#
# Modified:
#   * 4/4/18 - DMN - danielnewman09@gmail.com
#       - Added Documentation for this module
# 
#------------------------------------------------------------------------------
"""
import os
import pdb

import numpy as np
from scipy import optimize
from scipy.interpolate import griddata

# Let's also improve the printing of NumPy arrays.
np.set_printoptions(suppress=True, formatter={'float': '{: 0.4f}'.format})

# Define a few constants
HZ_TO_RADS = 2.0 * np.pi

this_path = os.path.split(os.path.realpath(__file__))[0]

data = np.genfromtxt(this_path + '/damped_imp_amp.txt',skip_header=1,delimiter=',')
data_Amps = data[:,0]
data_Phi = data[:,1]
data_tacc = data[:,2]
data_zeta = data[:,3]

data_tacc = data_tacc.reshape(len(data_tacc),1)
data_zeta = data_zeta.reshape(len(data_zeta),1)

#------ IC Utility Functions --------------------------------------------------
def ic_shift(shifted,design_freq,zeta,tacc):
    '''
    return the shifted properties of a given system based on actuator constraints

    Created by: Daniel Newman - danielnewman09@gmail.com

    Inputs:
        shifted - boolean specifying whether the shifts will be applied
        design_freq - designed natural frequency for the system
        zeta - designed damping ratio
        tacc - acceleration time for the pulse-limited system

    Returns:
        amplitude_shift - modification to the amplitude of the impulses
        phase_shift - modification to the phase of the impulses

    '''

    if shifted == True:
        wd = design_freq * np.sqrt(1 - zeta**2)

        # Undamped period of oscillation
        tau = 2 * np.pi / wd

        # Acceleration time normalized by the oscillation period
        norm_tacc = tacc / tau

        amplitude_shift = griddata(
                                     np.hstack((data_tacc, data_zeta)),
                                     data_Amps, (norm_tacc, zeta))
        phase_shift = griddata(
                                np.hstack((data_tacc, data_zeta)), 
                                data_Phi, (norm_tacc, zeta))
    else:
        phase_shift = np.array([0.])
        amplitude_shift = np.array([1.0])

    amplitude_shift[np.isnan(amplitude_shift)]=1.0
    phase_shift[np.isnan(phase_shift)]=0.

    return phase_shift,amplitude_shift

def input_phase(ic_pos,ic_vel,freq,zeta,is_impulse):
    '''
    Return the input phase of the initial condition, depending on whether
    the desired command is an impulse or a step

    Created by:  Daniel Newman -- danielnewman09@gmail.com
    
    Inputs:
        ic_pos - initial position of the flexible mode
        ic_vel - initial velocity of the flexible mode
        freq - current design frequency
        zeta - damping ratio
        is_impulse - boolean used to determine whether the reference command
                     is an impulse or step

    Returns:
        phase - initial condition phase

    '''
    if is_impulse:
        phase = np.arctan2(ic_pos * np.sqrt(1 - zeta**2),(ic_vel / freq + zeta * ic_pos))
    else:
        phase = -np.arctan2((ic_vel / freq + zeta * ic_pos), ic_pos * np.sqrt(1 - zeta**2))
    return phase

def input_amp(ic_pos,ic_vel,freq,zeta,is_impulse):
    '''
    Return the input amplitude of the initial condition, depending on whether
    the desired command is an impulse or a step

    Created by:  Daniel Newman -- danielnewman09@gmail.com
    
    Inputs:
        ic_pos - initial position of the flexible mode
        ic_vel - initial velocity of the flexible mode
        freq - current design frequency
        zeta - damping ratio
        is_impulse - boolean used to determine whether the reference command
                     is an impulse or step

    Returns:
        amp - normalized initial condition amplitude

    '''
    amp = np.sqrt((ic_pos)**2 + ((ic_vel / freq + zeta * ic_pos)/np.sqrt(1 - zeta**2))**2)

    if is_impulse:
        amp /= (freq / np.sqrt(1 - zeta**2))
    return amp

def modify_amp_terms(cos_term,sin_term,norm_amp,phase,amplitude_shift,sign,is_impulse):
    '''
    Return the modified cosine and sine terms of the vibration equation based on the 
    given initial conditions

    Created by:  Daniel Newman -- danielnewman09@gmail.com
    
    Inputs:
        cos_term - current cosine term
        sin_term - current sine term
        norm_amp - normalized amplitude of the initial condition
        phase - phase angle of the initial condition
        amplitude_shift - amplitude shift due to actuator limitations
        sign - direction of motion
        is_impulse - boolean used to determine whether the reference command
                     is an impulse or step

    Returns:
        cos_term - modified cosine term
        sin_term - modified sine term

    ''' 
    if is_impulse:
        cos_term += sign * norm_amp / amplitude_shift * np.cos(-phase) 
        sin_term += sign * norm_amp / amplitude_shift * np.sin(-phase)
    else:
        cos_term -= sign * norm_amp / amplitude_shift * np.cos(phase)
        sin_term += sign * norm_amp / amplitude_shift * np.sin(phase)

    return cos_term,sin_term

def normalize_vib(phase_shift,amplitude_shift,norm_amp,input_phase,is_impulse):
    '''
    Determine the denominator of the vibration equation for an 
    initial condition input shaper

    Created by: Daniel Newman -- danielnewman09@gmail.com

    Inputs:
        phase_shift - phase shift due to actuator limitations
        amplitude_shift - amplitude shift due to actuator limitations
        norm_amp - normalized amplitude of the initial conditions
        input_phase - unshifted phase of the initial conditions
        is_impulse - boolean used to determine whether the reference command
             is an impulse or step

    Returns:
        vib - denominator of the vibration equation for initial conditions
    '''

    vib = np.sqrt((amplitude_shift * np.cos(phase_shift) + norm_amp * np.cos(-input_phase))**2\
           +(amplitude_shift * np.sin(phase_shift) + norm_amp * np.sin(-input_phase))**2)

    return vib
            
#------ Optimized Shapers ---------------------------------------------------------
class IC_Shaper(object):
    '''
    This abstract class contains all of the methods for 
    generating an input shaper which must be optimized

    freq_args is a list of inputs to create any
    optimized shaper

    ic_args is a list of inputs which specify
    the necessary values to create the initial condition shaper

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
   
    '''

    def __init__(self,freq_args,ic_args,
                 tol=1e-6,eps=1e-9,num_points=50,
                 trial_amp=1.,nominal_amp=0.4,is_impulse=True):
 
        X0,f_min,f_max,Vtol,zeta = freq_args
        x0,v0,tacc,Vmax,sign,design_freq,shifted = ic_args

        # Variables specific to IC shapers
        self.trial_amp = trial_amp
        self.num_points = num_points
        self.is_impulse = is_impulse
        self.best_vib = 100.
        self.nominal_amp = nominal_amp

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
        
        # These are good parameters to keep track of regardless of shaper type
        self.num_impulses = 0
        self.bnds = ()
        self.consts = ()
        self.seeking_solution = True

        # Calculate period use for initial time guess 
        self.tau = 2 * np.pi/self.wn

        phase_shift, amplitude_shift = ic_shift(shifted,design_freq,zeta,tacc)

        ic_args = [x0,v0,tacc,Vmax,sign,design_freq,shifted,amplitude_shift,phase_shift]

        # Define the contraints for vibration, amplitude-sum, and t_1=0
        self.consts += ({'type': 'ineq',
                'fun': lambda x: Vtol - self.vib(x, freq_args,ic_args)},
                {'type':'eq',
                'fun': lambda x: self.amp_sum(x) - 1.0, # sum(Ai) = 1
                'jac': lambda x: np.hstack((np.zeros((1,self.num_impulses)), 
                                            np.ones((1,self.num_impulses))))},
                {'type':'eq',
                'fun': lambda x: x[0]}, # t1 = 0}
                )

        return ic_args

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

    def vib(self, x, freq_args, ic_args):
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
        x0,v0,tacc,Vmax,sign,wn,shifted,amplitude_shift,phase_shift =  ic_args

        ic_pos = x0 / Vmax
        ic_vel = v0 / Vmax

        x = np.asarray(x)

        num_impulses = int(np.round(len(x)/2))

        vib = np.zeros(self.num_points,)

        for ii, freq in enumerate(np.linspace(f_min * HZ_TO_RADS, f_max * HZ_TO_RADS, self.num_points)):
            wd = freq * np.sqrt(1 - zeta**2)

            # The magnitudes of the shaper, excluding the initial conditions
            cos_term = np.sum(x[num_impulses:] * np.exp(zeta*freq*x[0:num_impulses])\
                              * np.cos(wd*x[0:num_impulses]))
            sin_term = np.sum(x[num_impulses:] * np.exp(zeta*freq*x[0:num_impulses])\
                              * np.sin(wd*x[0:num_impulses]))
            
            # This is the amplitude of the initial condition vector
            norm_amp = input_amp(ic_pos,ic_vel,freq,zeta,self.is_impulse)
            
            # Calculate the phase angle of the initial conditions
            original_phase = input_phase(ic_pos,ic_vel,freq,zeta,self.is_impulse)

            # add the numerically evaluated phase shift to the calculated phase.
            phase = original_phase + phase_shift

            # Modify the cos and sin terms based on the initial conditions
            cos_term,sin_term = modify_amp_terms(
                                    cos_term,
                                    sin_term,
                                    norm_amp,
                                    phase,
                                    amplitude_shift,
                                    sign,
                                    self.is_impulse)

            # Vibration resulting from the shaped input
            vib[ii] = np.sqrt((cos_term)**2 + (sin_term)**2)
            
            # Vibration resulting from a unity magnitude step at t=0
            vib[ii] /= normalize_vib(
                                phase_shift,
                                amplitude_shift,
                                norm_amp,
                                original_phase,
                                self.is_impulse)

                              
        return vib  

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
        
        return res

    def test_vib(self,Vtol,res_success,current_vib):
        '''
        The initial condition shapers can have difficulty reaching the desired 
        conditions. Therefore, test each solution and make the appropriate adjustments.
        '''

        if res_success:

            # This value should be zero, indicating that 
            # the vibration constraint is met
            vib_diff = current_vib - Vtol
            vib_diff = np.clip(vib_diff,0,5)

            # If the vibration constraint is not met
            if np.any(vib_diff > 1e-3):

                self.trial_amp -= 0.05

                # We're going to iterate until we find the best solution
                if np.mean(vib_diff) < self.best_vib:
                    self.best_vib = np.mean(vib_diff)
                    self.best_shaper = np.hstack((self.times,self.amps))
            else:
                # We have found the correct shaper
                self.seeking_solution = False

        else:
            # If the optimization failed, 
            # try slightly different initial conditions
            if self.trial_amp >= 0.0:
                self.trial_amp -= 0.05
                print('Retrying... Trial Amp = {}'.format(self.trial_amp))
            else:
                # We have reached the end of the allowable iterations
                self.shaper = self.best_shaper
                self.seeking_solution = False

class ZVIC_Shaper(IC_Shaper):

    def __init__(self,freq_args,ic_args,**args):
        ic_args = IC_Shaper.__init__(self,freq_args,ic_args,**args)
        self.num_points = 1
        self.res = self.solve_for_shaper(freq_args,ic_args)

    def solve_for_shaper(self,freq_args,ic_args):
        X0,f_min,f_max,Vtol,zeta = freq_args

        X0 = self.optimization_setup(freq_args,ic_args)

        for ii in range(self.num_impulses):
            self.bnds += ((0.0, 3*self.tau),)

        # Positive amplitude shaper
        for ii in range(self.num_impulses):
            # create bounds on impulse ampliutdes of +/-1
            self.bnds += ((0.0,1.0),)
                    
        while self.seeking_solution:
            
            X0 = self.optimization_setup(freq_args,ic_args)
            
            res = IC_Shaper.optimize(self,X0)

            thisvib = np.asarray(IC_Shaper.vib(self,res.x,freq_args,ic_args))

            IC_Shaper.test_vib(self,Vtol,res.success,thisvib)

            X0 = None

        return res 

    def optimization_setup(self,freq_args,ic_args):
        x0,v0,tacc,Vmax,sign,wn,shifted,amplitude_shift,phase_shift =  ic_args
        X0,f_min,f_max,Vtol,zeta = freq_args

        self.num_impulses = 2

        ic_pos = x0 / Vmax
        ic_vel = v0 / Vmax
    
        if X0 is None:                # if I.C.'s unknown, create X0
            phase = np.arctan2(
                                ic_pos * np.sqrt(1 - zeta**2),
                                (ic_vel / wn + zeta * ic_pos))

            norm_amp = np.sqrt(
                        (ic_pos)**2 + 
                        ((ic_vel / wn+ zeta * ic_pos)/np.sqrt(1 - zeta**2))**2)\
                        / (wn / np.sqrt(1 - zeta**2))

            shifted_amp = norm_amp / amplitude_shift

            X0 = np.zeros(2 * self.num_impulses)

            if self.is_impulse:
                def objective(T):
                    '''
                    We want to minimize the time of the second impulse
                    '''
                    return T**2

                # The constraint equation for the damped ZV-IC shaper is \
                # derived by setting the imaginary component of the impulse
                # amplitude to equal zero
                if x0 < 1e-6:
                    x0 = 1e-6

                consts = ({'type': 'eq',
                        'fun': lambda t: (1 - np.exp(zeta * wn * t) * np.cos(wn * np.sqrt(1 - zeta**2) * t))\
                                         - (wn**2 + zeta * wn * x0 + v0)/(x0 * wn * np.sqrt(1 - zeta**2)) \
                                         * (np.exp(zeta * wn * t) * np.sin(wn * np.sqrt( 1 - zeta**2) * t))})
                # Find the optimal time for the second impulse
                bounds = (((0.00,self.tau)),)
                t_opt = optimize.minimize(objective, self.tau/2, bounds = bounds, 
                                    constraints = consts, method='SLSQP',
                                    options={'maxiter':1e3, 'disp':False})
                        
                t_opt = t_opt.x[0]

                # Bookkeeping functions for the amplitude solutions
                C = (1 - np.exp(zeta * wn * t_opt) * np.cos(wn * np.sqrt(1 - zeta**2) * t_opt))
                S = np.exp(zeta * wn * t_opt) * np.sin(wn * np.sqrt(1 - zeta**2) * t_opt)

                # Solve for the shaper amplitudes based on the optimal impulse time
                A_1 = ((C - 1 - zeta * x0 / wn - v0 / wn**2) * C + (S - x0 / wn * np.sqrt(1 - zeta**2)) * S) / (C**2 + S**2)
                A_2 = ((1 + zeta * x0 / wn + v0 / wn**2) * C + x0 / wn * np.sqrt(1 - zeta**2) * S) / (C**2 + S**2)

                if np.isnan(A_1):
                    A_1 = 1.
                    A_2 = 0.
                    t_opt = tau / 2

                X0[1] = t_opt * self.trial_amp
                X0[2] = A_1 * self.trial_amp
                X0[3] = 1 - A_1

            else:
        
                # Calculate the phase angle of the initial conditions
                phase = -np.arctan2(x0,v0/(wn))

                # Create a normalized error
                phase = phase + np.pi * tacc * wn / (2 * np.pi)

                norm_amp = np.sqrt(x0**2 + (v0 / wn)**2) 

                if phase > -np.pi:
                    X0[0] = (phase + 2 * np.pi) / wn
                    X0[1] = (phase + np.pi) / wn
                else:
                    X0[0] = (phase + 4 * np.pi) / wn
                    X0[1] = (phase + 3 * np.pi) / wn

                X0[2] = (norm_amp + 1 * self.trial_amp) / 2
                X0[3] = 1 - X0[2]

        self.num_impulses = int(np.round(len(X0) / 2))

        return X0

class SIIC_Shaper(IC_Shaper):

    def __init__(self,freq_args,ic_args,**args):
        
        ic_args = IC_Shaper.__init__(self,freq_args,ic_args,**args)

        self.res, self.shaper = self.solve_for_shaper(freq_args,ic_args)

    def cum_sum(self,x):
        """Function to determine the running sum of the shaper at every point"""

        shaper = np.reshape(x,(2,self.num_impulses)).T
        shaper = shaper[shaper[:,0].argsort()]
        
        return np.abs(np.cumsum(shaper[0:self.num_impulses,1]))  

    def imp_space(self,x):
        """Function to determine the spacing between impulses"""
        
        shaper = np.reshape(x,(2,self.num_impulses)).T
        shaper = shaper[shaper[:,0].argsort()]

        imp_space = np.diff(shaper[:,0])

        return imp_space

    def solve_for_shaper(self,freq_args,ic_args):

        x0,v0,tacc,Vmax,sign,wn,shifted,amplitude_shift,phase_shift =  ic_args
        X0,f_min,f_max,Vtol,zeta = freq_args

        X0 = self.initial_guess(freq_args,ic_args)

        for ii in range(self.num_impulses):
            self.bnds += ((0.0, 3*self.tau),)

        # Unity Magnitude Shaper
        for ii in range(self.num_impulses):
            # create bounds on impulse ampliutdes of +/-1
            self.bnds += (((-1),(1)),)      

        self.consts += (
            # Enforce the cumulative sum constraint
            {'type': 'ineq',
            'fun': lambda x: 1 - self.cum_sum(x)},
            {'type': 'ineq',
            'fun': lambda x: self.imp_space(x) - tacc},
            )
            
        while self.seeking_solution:

            X0 = self.initial_guess(freq_args,ic_args)
            res = IC_Shaper.optimize(self,X0)

            args = [X0,f_min,f_max,Vtol,zeta,ic_args]

            thisvib = np.asarray(IC_Shaper.vib(self,res.x,freq_args,ic_args))

            IC_Shaper.test_vib(self,Vtol,res.success,thisvib)

            X0 = None

        return res, self.shaper

    def initial_guess(self,freq_args,ic_args):
        """Form the initial guess for the SI-IC Shaper """

        x0,v0,tacc,Vmax,sign,wn,shifted,amplitude_shift,phase_shift =  ic_args
        X0,f_min,f_max,Vtol,zeta = freq_args

        ic_pos = x0 / Vmax
        ic_vel = v0 / Vmax
    
        phase = np.arctan2(
                            ic_pos * np.sqrt(1 - zeta**2),
                            (ic_vel / wn + zeta * ic_pos))

        norm_amp = np.sqrt(
                    (ic_pos)**2 
                    + ((ic_vel / wn+ zeta * ic_pos)/np.sqrt(1 - zeta**2))**2)\
                        / (wn / np.sqrt(1 - zeta**2)
                           )

        shifted_amp = norm_amp / amplitude_shift

        # If no initial guess is given, then create one based on curve fits of known solutions
        # As of 12/22/14, these initial guesses assume no damping, so damped solutions mail fail
        if X0 is None:                # if I.C.'s unknown, create X0

            if self.Ins <= 0.39 and np.average(shifted_amp) < self.nominal_amp:
                ShaperLength = 5
                shaper = np.zeros([ShaperLength,2])
                shaper[0,0] = 0
                shaper[1,0] = -0.0091*self.trial_amp**2 - 0.041*self.trial_amp + 0.1466
                shaper[2,0] = 0.0923*self.trial_amp**2 - 0.2134*self.trial_amp + 0.4881
                shaper[3,0] = 0.1933*self.trial_amp**2 - 0.3856*self.trial_amp + 0.8297
                shaper[4,0] = 0.1843*self.trial_amp**2 - 0.4267*self.trial_amp + 0.9763
                
                shaper[0,1] = 0.4067*self.trial_amp**3 - 0.5703*self.trial_amp**2 + 0.9112*self.trial_amp + 0.25
                shaper[1,1] = -self.trial_amp
                shaper[2,1] = -0.8135*self.trial_amp**3 + 1.1401*self.trial_amp**2 + 0.1782*self.trial_amp + 0.4998
                shaper[3,1] = -self.trial_amp
                shaper[4,1] = shaper[0,1]

                X0 = np.hstack((shaper[:,0], shaper[:,1]))          
         
            elif self.Ins <= .7262 and np.average(shifted_amp) < self.nominal_amp:
                ShaperLength = 7
                shaper = np.zeros([ShaperLength,2])
                shaper[0,0] = 0
                shaper[1,0] = -0.0698*self.trial_amp**2 + 0.0185*self.trial_amp + 0.1171
                shaper[2,0] = 0.1952*self.trial_amp**2 - 0.3032*self.trial_amp + 0.4949
                shaper[3,0] = 0.1571*self.trial_amp**2 - 0.3053*self.trial_amp + 0.7294
                shaper[4,0] = 0.1190**self.trial_amp**2 - 0.3075*self.trial_amp + 0.9639
                shaper[5,0] = 0.3839*self.trial_amp**2 - 0.6290*self.trial_amp + 1.3416
                shaper[6,0] = 0.3138*self.trial_amp**2 - 0.6102*self.trial_amp + 1.4587

                shaper[0,1] = 1.6343*self.trial_amp**4 - 2.4423*self.trial_amp**3 + 1.0978*self.trial_amp**2 + 0.5355*self.trial_amp + 0.1772
                shaper[1,1] = -self.trial_amp
                shaper[2,1] = -1.6343*self.trial_amp**4 + 2.4423*self.trial_amp**3 - 1.0978*self.trial_amp**2 + 0.9645*self.trial_amp +0.3228
                shaper[3,1] = -self.trial_amp
                shaper[4,1] = shaper[2,1]
                shaper[5,1] = -self.trial_amp
                shaper[6,1] = shaper[0,1]
                X0 = np.hstack((shaper[:,0], shaper[:,1]))
         
            elif self.Ins <= .9654:
                ShaperLength = 9
                shaper = np.zeros([ShaperLength,2])
                shaper[0,0] = 0
                shaper[1,0] = (-0.0856*self.trial_amp**2 + 0.0235*self.trial_amp + 0.1148) * self.tau
                shaper[2,0] = (0.3095*self.trial_amp**2 - 0.4085*self.trial_amp + 0.4947) * self.tau
                shaper[3,0] = (0.1438*self.trial_amp**2 - 0.3033*self.trial_amp + 0.7011) * self.tau
                shaper[4,0] = (0.2201*self.trial_amp**2 - 0.3867*self.trial_amp + 0.9686) * self.tau
                shaper[5,0] = (0.2973*self.trial_amp**2 - 0.4709*self.trial_amp + 1.237) * self.tau
                shaper[6,0] = (0.1308*self.trial_amp**2 - 0.3648*self.trial_amp + 1.4424) * self.tau
                shaper[7,0] = (0.5266*self.trial_amp**2 - 0.7978*self.trial_amp + 1.8226) * self.tau
                shaper[8,0] = (0.4407*self.trial_amp**2 - 0.7738*self.trial_amp + 1.9372) * self.tau

                shaper[0,1] = 0.3615*self.trial_amp**5 + 2.2773*self.trial_amp**4 - 4.501*self.trial_amp**3 + 2.5652*self.trial_amp**2 + 0.1458*self.trial_amp + 0.1537
                shaper[1,1] = -self.trial_amp
                shaper[2,1] = -4.7821*self.trial_amp**5 + 10.014*self.trial_amp**4 - 7.4091*self.trial_amp**3 + 2.4361*self.trial_amp**2 + 0.492*self.trial_amp + 0.2475
                shaper[3,1] = -self.trial_amp
                shaper[4,1] = 8.7667*self.trial_amp**5 - 24.359*self.trial_amp**4 + 23.578*self.trial_amp**3 - 9.8884*self.trial_amp**2 + 2.7027*self.trial_amp + 0.1989
                shaper[5,1] = -self.trial_amp
                shaper[6,1] = shaper[2,1]
                shaper[7,1] = -self.trial_amp
                shaper[8,1] = shaper[0,1]
                X0 = np.hstack((shaper[:,0], shaper[:,1]))

            else:
                raise ValueError('Code only works (as of 12/26/14) for negative shapers up to I(5%) = 1.2')
                self.seeking_solution = False
                
        
        self.num_impulses = int(np.round(len(X0) / 2))

        return X0

def ic_sensplot(shaper, fmin, fmax, p,zeta, is_impulse=True, numpoints = 500):
    '''
    Create the sensitivity plot for an initial condition input shaper

    Created by: Daniel Newman -- danielnewman09@gmail.com

    Inputs:
        shaper - input shaper
        fmin - minimum frequency to plot
        fmax - maximum frequency to plot
        p - initial condition args
        zeta - damping ratio
        is_impulse - boolean indicating whether the reference command is an impulse
                     or a step
        numpoints - number of points to evaluate

    '''
    x0,v0,tacc,Vmax,sign,design_freq,shifted = p

    frequency = np.zeros((numpoints,1))
    amplitude = np.zeros((numpoints,1))

    ic_pos = x0 / (Vmax)
    ic_vel = v0 / (Vmax)

    phase_shift, amplitude_shift = ic_shift(shifted,design_freq,zeta,tacc)

    for ii, freq in enumerate(np.linspace(fmin * (2*np.pi), fmax * (2*np.pi), numpoints)):

        wd = freq * np.sqrt(1 - zeta**2)
            
        cos_term = np.sum(shaper[:,1] * np.exp(zeta*freq*shaper[:,0]) * np.cos(wd*shaper[:,0]))
        sin_term = np.sum(shaper[:,1] * np.exp(zeta*freq*shaper[:,0]) * np.sin(wd*shaper[:,0]))
        
        # This is the amplitude of the initial condition vector
        norm_amp = input_amp(ic_pos,ic_vel,freq,zeta,is_impulse)

        original_phase = input_phase(ic_pos,ic_vel,freq,zeta,is_impulse)
    
        # add the numerically evaluated phase shift to the calculated phase.
        phase = original_phase + phase_shift

        cos_term, sin_term = modify_amp_terms(
                                    cos_term,sin_term,
                                    norm_amp,
                                    phase,
                                    amplitude_shift,
                                    sign,
                                    is_impulse)

        frequency[ii,0] = freq / HZ_TO_RADS

        amplitude[ii,0] = 100 * np.sqrt((cos_term)**2 + (sin_term)**2) \
                                / np.exp(-zeta * freq * (shaper[-1,0] - phase_shift \
                                / (freq * np.sqrt(1 - zeta**2))))

        amplitude[ii,0] /= normalize_vib(
                                phase_shift,
                                amplitude_shift,
                                norm_amp,
                                original_phase,
                                is_impulse)

    return frequency, amplitude


def ic_phase_sensplot(shaper, phasemin, phasemax, p, zeta, is_impulse=True, numpoints = 500):
    '''
    Create the sensitivity plot for an initial condition input shaper

    Created by: Daniel Newman -- danielnewman09@gmail.com

    Inputs:
        shaper - input shaper
        phasemin - minimum phase to plot
        phasemax - maximum phase to plot
        p - initial condition args
        zeta - damping ratio
        is_impulse - boolean indicating whether the reference command is an impulse
                     or a step
        numpoints - number of points to evaluate

    '''

    x0,v0,tacc,Vmax,sign,design_freq,shifted = p

    norm_phase = np.zeros((numpoints,1))
    amplitude = np.zeros((numpoints,1))
    
    wd = design_freq * np.sqrt(1 - zeta**2)

    phase_shift, amplitude_shift = ic_shift(shifted,design_freq,zeta,tacc)

    # Calculate the phase angle of the initial conditions
    original_phase = input_phase(ic_pos,ic_vel,design_freq,zeta,is_impulse)
            
    act_phase = original_phase

    for ii, curr_phase in enumerate(np.linspace(phasemin, phasemax, numpoints)):

        unshifted_phase = curr_phase + act_phase
        
        cos_term = np.sum(shaper[:,1] * np.exp(zeta*design_freq*shaper[:,0]) * np.cos(wd*shaper[:,0]))
        sin_term = np.sum(shaper[:,1] * np.exp(zeta*design_freq*shaper[:,0]) * np.sin(wd*shaper[:,0]))

        # Create a normalized error
        phase = unshifted_phase + phase_shift

        # This is the amplitude of the initial condition vector
        norm_amp = input_amp(ic_pos,ic_vel,design_freq,zeta,is_impulse)
        
        cos_term, sin_term = modify_amp_terms(
                            cos_term,sin_term,
                            norm_amp,
                            phase,
                            amplitude_shift,
                            sign,
                            is_impulse)  

        norm_phase[ii,0] = np.rad2deg((phase - phase_shift - act_phase))
        amplitude[ii,0] = np.sqrt((cos_term)**2 + (sin_term)**2)
        amplitude[ii,0] /= normalize_vib(
                                phase_shift + unshifted_phase,
                                amplitude_shift,
                                norm_amp,
                                original_phase,
                                is_impulse)
                                    
    return norm_phase, amplitude * 100

def ic_phase_freq_sens(shaper, fmin, fmax, phasemin, phasemax, p, zeta, 
                       numpoints = 500,folder='3D Sens/',filename='phase_freq_data'):
    '''
    Create the sensitivity plot for an initial condition input shaper.

    This function creates a 3D representation showing the phase and frequency

    Created by: Daniel Newman -- danielnewman09@gmail.com

    Inputs:
        shaper - input shaper
        fmin - minimum frequency to plot
        fmax - maximum frequency to plot
        phasemin - minimum phase to plot
        phasemax - maximum phase to plot
        p - initial condition args
        zeta - damping ratio
        is_impulse - boolean indicating whether the reference command is an impulse
                     or a step
        numpoints - number of points to evaluate

    '''    
    x0,v0,tacc,Vmax,sign,design_freq,shifted = p
    
    phase_shift, amplitude_shift = ic_shift(shifted,design_freq,zeta,tacc)

    if not os.path.exists(folder):
        os.makedirs(folder) 

    data = open(folder + filename + '.txt','w')
    data.write('Frequency, Phase, Amplitude \n')

    for ii, freq in enumerate(np.linspace(fmin * (2*np.pi), fmax * (2*np.pi), numpoints)):

        wd = freq * np.sqrt(1 - zeta**2)

        # Calculate the phase angle of the initial conditions
        original_phase = input_phase(ic_pos,ic_vel,freq,zeta,is_impulse)

        for nn, curr_phase in enumerate(np.linspace(phasemin, phasemax, numpoints)):
                           
            unshifted_phase = curr_phase + original_phase
            
            cos_term = np.sum(shaper[:,1] * np.exp(zeta*freq*shaper[:,0]) * np.cos(wd*shaper[:,0]))
            sin_term = np.sum(shaper[:,1] * np.exp(zeta*freq*shaper[:,0]) * np.sin(wd*shaper[:,0]))

            # Create a normalized error
            phase = unshifted_phase + phase_shift

            norm_amp = input_amp(ic_pos,ic_vel,freq,zeta,is_impulse)

            cos_term, sin_term = modify_amp_terms(
                                cos_term,sin_term,
                                norm_amp,
                                phase,
                                amplitude_shift,
                                sign,
                                is_impulse)  

            norm_phase = np.rad2deg(phase - phase_shift - original_phase)
            amplitude = np.sqrt((cos_term)**2 + (sin_term)**2) * 100

            amplitude /= normalize_vib(
                                    phase_shift + unshifted_phase,
                                    amplitude_shift,
                                    norm_amp,
                                    original_phase,
                                    is_impulse)

            data.write('{}, {}, {} \n'.format(np.round(freq / design_freq,3),np.round(norm_phase,3),np.round(amplitude,3)))
    data.close()

    return np.genfromtxt(folder + filename + '.txt',skip_header=1,delimiter=',')

def ic_amp_sensplot(shaper, ampmin, ampmax, p, zeta, numpoints = 500):
    '''
    Create the sensitivity plot for an initial condition input shaper

    Created by: Daniel Newman -- danielnewman09@gmail.com

    Inputs:
        shaper - input shaper
        phasemin - minimum amplitude to plot
        phasemax - maximum amplitude to plot
        p - initial condition args
        zeta - damping ratio
        is_impulse - boolean indicating whether the reference command is an impulse
                     or a step
        numpoints - number of points to evaluate
    '''
    
    x0,v0,tacc,Vmax,sign,design_freq,shifted = p

    wd = design_freq * np.sqrt(1 - zeta**2)

    norm_amp = np.zeros((numpoints,1))
    amplitude = np.zeros((numpoints,1))

    phase_shift, amplitude_shift = ic_shift(shifted,design_freq,zeta,tacc)

    # Calculate the phase angle of the initial conditions
    original_phase = input_phase(ic_pos,ic_vel,design_freq,zeta,is_impulse)
    phase = original_phase + phase_shift

    act_amp = input_amp(ic_pos,ic_vel,design_freq,zeta,is_impulse)

    for ii, curr_amp in enumerate(np.linspace(ampmin, ampmax, numpoints)):
        amp = curr_amp

        cos_term = np.sum(shaper[:,1] * np.exp(zeta*design_freq*shaper[:,0]) * np.cos(wd*shaper[:,0]))
        sin_term = np.sum(shaper[:,1] * np.exp(zeta*design_freq*shaper[:,0]) * np.sin(wd*shaper[:,0]))

        cos_term, sin_term = modify_amp_terms(
                            cos_term,sin_term,
                            (act_amp + amp),
                            phase,
                            amplitude_shift,
                            sign,
                            is_impulse)  

        norm_amp[ii,0] = (amp) * amplitude_shift
        amplitude[ii,0] = np.sqrt((cos_term)**2 + (sin_term)**2)

        amplitude[ii,0] /= normalize_vib(
                                phase_shift,
                                amplitude_shift,
                                (act_amp + amp),
                                original_phase,
                                is_impulse)

    return norm_amp, amplitude * 100


def ic_amp_freq_sens(shaper, fmin, fmax, ampmin, ampmax, p, zeta, 
                     numpoints = 500,folder='3D Sens/',filename='amp_freq_data'):
    '''
    Create the sensitivity plot for an initial condition input shaper

    Created by: Daniel Newman -- danielnewman09@gmail.com

    Inputs:
        shaper - input shaper
        ampmin - minimum amplitude to plot
        ampmax - maximum amplitude to plot
        fmin - minimum frequency to plot
        fmax - maximum frequency to plot
        p - initial condition args
        zeta - damping ratio
        is_impulse - boolean indicating whether the reference command is an impulse
                     or a step
        numpoints - number of points to evaluate
    '''
    x0,v0,tacc,Vmax,sign,design_freq,shifted = p

    if not os.path.exists(folder):
        os.makedirs(folder) 

    data = open(folder + filename + '.txt','w')
    data.write('Frequency, Norm_Amp, Amplitude \n')

    print('\n \n Writing to {}'.format(folder + filename))

    phase_shift, amplitude_shift = ic_shift(shifted,design_freq,zeta,tacc)

    for ii, freq in enumerate(np.linspace(fmin * (2*np.pi), fmax * (2*np.pi), numpoints)):

        wd = freq * np.sqrt(1 - zeta**2)

        # Calculate the phase angle of the initial conditions
        original_phase = input_phase(ic_pos,ic_vel,freq,zeta,is_impulse)
        phase = original_phase + phase_shift

        # This is the amplitude of the initial condition vector
        act_amp = input_amp(ic_pos,ic_vel,freq,zeta,is_impulse)

        for nn, curr_amp in enumerate(np.linspace(ampmin, ampmax, numpoints)):

            amp = curr_amp
            
            cos_term = np.sum(shaper[:,1] * np.exp(zeta*freq*shaper[:,0]) * np.cos(wd*shaper[:,0]))
            sin_term = np.sum(shaper[:,1] * np.exp(zeta*freq*shaper[:,0]) * np.sin(wd*shaper[:,0]))

            cos_term, sin_term = modify_amp_terms(
                                cos_term,sin_term,
                                (act_amp + amp),
                                phase,
                                amplitude_shift,
                                sign,
                                is_impulse)

            norm_amp = amp * amplitude_shift
            amplitude = 100 * np.sqrt((cos_term)**2 + (sin_term)**2)
            
            amplitude[ii,0] /= normalize_vib(
                                    phase_shift,
                                    amplitude_shift,
                                    (act_amp + amp),
                                    original_phase,
                                    is_impulse)

            data.write('{}, {}, {} \n'.format(
                                            np.round(freq / design_freq,3),
                                            np.round(norm_amp,3),
                                            np.round(amplitude,3)))

    data.close()

    return np.genfromtxt(folder + filename + '.txt',skip_header=1,delimiter=',')

def ic_phase_amp_sens(shaper, ampmin, ampmax, phasemin, phasemax, p, zeta, 
                    numpoints = 500,folder='3D Sens/',filename='phase_amp_data'):
    '''
    Create the sensitivity plot for an initial condition input shaper

    Created by: Daniel Newman -- danielnewman09@gmail.com

    Inputs:
        shaper - input shaper
        ampmin - minimum amplitude to plot
        ampmax - maximum amplitude to plot
        phasemin - minimum phase to plot
        phasemax - maximum phase to plot
        p - initial condition args
        zeta - damping ratio
        is_impulse - boolean indicating whether the reference command is an impulse
                     or a step
        numpoints - number of points to evaluate
    '''
    x0,v0,tacc,Vmax,sign,design_freq,shifted = p

    if not os.path.exists(folder):
        os.makedirs(folder) 

    phase_shift, amplitude_shift = ic_shift(shifted,design_freq,zeta,tacc)

    data = open(folder + filename + '.txt','w')
    data.write('Amp, Phase, Amplitude \n')

    wd = design_freq * np.sqrt(1 - zeta**2)

    # Calculate the phase angle of the initial conditions
    act_phase = input_phase(ic_pos,ic_vel,design_freq,zeta,is_impulse)
    act_phase += phase_shift

    # This is the amplitude of the initial condition vector
    act_amp = input_amp(ic_pos,ic_vel,design_freq,zeta,is_impulse)

    for ii, curr_amp in enumerate(np.linspace(ampmin, ampmax, numpoints)):
        amp = curr_amp

        for nn, curr_phase in enumerate(np.linspace(phasemin, phasemax, numpoints)):

            phase = curr_phase + act_phase
            
            cos_term = np.sum(shaper[:,1] * np.exp(zeta*design_freq*shaper[:,0]) * np.cos(wd*shaper[:,0]))
            sin_term = np.sum(shaper[:,1] * np.exp(zeta*design_freq*shaper[:,0]) * np.sin(wd*shaper[:,0]))

            # Create a normalized error
            phase = phase + phase_shift

            cos_term, sin_term = modify_amp_terms(
                    cos_term,sin_term,
                    (act_amp + amp),
                    (phase + phase_shift),
                    amplitude_shift,
                    sign,
                    is_impulse)

            norm_amp = amp * amplitude_shift
            norm_phase = np.rad2deg(phase - phase_shift - act_phase)
            amplitude = 100 * np.sqrt((cos_term)**2 + (sin_term)**2)
            amplitude /= normalize_vib(
                                    (phase_shift - phase),
                                    amplitude_shift,
                                    (act_amp + amp),
                                    original_phase,
                                    is_impulse)

            data.write('{}, {}, {} \n'.format(
                                        np.round(amp,3),
                                        np.round(norm_phase,3),
                                        np.round(amplitude,3)))
    data.close()
    return np.genfromtxt(folder + filename + '.txt',skip_header=1,delimiter=',')