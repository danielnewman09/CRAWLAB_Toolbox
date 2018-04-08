#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
#------------------------------------------------------------------------------
# Input Shaping Module - InputShaping.py
#
# Python module for the input shaping toolbox
#   - Adapted from MATLAB input shaping toolbox
#
# Created: 2/18/13 - Joshua Vaughan - joshua.vaughan@louisiana.edu
#
# Modified:
#   * 2/19/13 - JEV - joshua.vaughan@louisiana.edu
#       - Added positive ZV-type input shapers
#       - Added positive EI-type input shapers
#   * 2/20/13 - JEV
#       - Added UM-ZV-type input shapers
#       - Added UM-EI-type input shapers
#   * 2/26/13 - JEV
#       - Added sensplot
#   * 3/26/13 - JEV
#       - began adding proper docstrings for use with help(___) or ___?
#   * 09/19/14 - JEV
#       - fixed numpy namespace
#   * 12/27/14 - JEV
#       - some improved formatting to be more idiomatic, still more to do
#   * 01/07/15 - JEV
#       - began move to class based structure
#   * 02/01/15 - JEV
#       - began work on two mode shapers
#       - added EI shaper and parent tolerable level shaper class
#   * 02/16/15 - JEV
#       - Finished class conversion for all "common" shapers
#       - Added common input types as functions
#       - Added functional shaped-command formulation
#------------------------------------------------------------------------------
"""
import os

import numpy as np
import warnings
import pdb
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.interpolate import griddata

from abc import ABCMeta, abstractmethod

# Let's also improve the printing of NumPy arrays.
np.set_printoptions(suppress=True, formatter={'float': '{: 0.4f}'.format})

# Define a few constants
HZ_TO_RADS = 2.0 * np.pi

#from crawlab_toolbox.utilities import digseq, seqconv, sensplot


class Shaper(object):
    ''' Parent class for all shapers

    Attributes:
    shaper : exact representation of the shaper
    digitized_shaper : digitized version of the shaper
    duration : duration of the shaper (s)
    type : type of shaper
    amp_type : type of amplitude constraints (Positive, Negative, or SNA)
    design_freq : the frequency used to design the shaper (Hz)
    design_damping : the damping ratio used to design the Shaper
    '''

    __metaclass__ = ABCMeta

    def __init__(self, frequency, zeta, deltaT = 0.01):
        """ Shaper Initialization function

        Parses the user inputs and calls solve_for_shaper method
        to get amplitudes and times

        Arguments:
          frequency : the design frequency for the shaper (Hz)
          zeta : the design damping ratio for the shaper
          deltaT : the samping time (s), used for the digitized version, default = 0.01
        """
        self.design_freq = frequency
        self.design_damping = zeta
        self.design_deltaT = deltaT
        self.shaper, self.digitized_shaper = self.solve_for_shaper(frequency, zeta, deltaT)
        self.times = self.shaper[:,0]
        self.amps = self.shaper[:,1]
        self.duration = self.times[-1]

    def __str__(self):
        """ Set up pretty printing of the shaper """
        type = 'Shaper Type \t \t \t {}\n'.format(self.type)
        designfreq = 'Design Frequency \t \t {:0.4f}\t Hz\n'.format(self.design_freq)
        designdamp = 'Design Damping Ratio \t \t {:0.4f}\n'.format(self.design_damping)
        duration = 'Duration \t \t \t {:0.4f}\t s \n'.format(self.duration)
        shaper = '\n' + '     ti      Ai \n{}\n'.format(self.shaper)

        return '\n' + type + designfreq + designdamp + duration + shaper

    def plot_sensitivity(self):
        """ Method to plot the sensitivity curve for the shaper between 0 and
        2x the design frequency

        For other, using the Input Shaping module sensplot function
        """
        sensplot(self.shaper, 0.0, 2.0*self.design_freq,
                 self.design_damping, numpoints = 2000, plotflag = 1)

    @abstractmethod
    def solve_for_shaper(self, *args):
        """ Return the shaper impulse amplitudes and times"""
        pass



#----- Positive ZV-Form Shapers (ZV, ZVD, ZVDD, ...) --------------------------
class ZV(Shaper):
    """ Class describing a ZV shaper """
    type = 'ZV'
    amp_type = 'Positive'
    isPositive = True

    def solve_for_shaper(self, frequency, zeta, deltaT):
        """ Return the shaper impulse amplitudes and times

        Arguments:
          The created instance, uses instance variables for calculation

        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """

        wn = frequency * HZ_TO_RADS
        K = np.exp(-zeta*np.pi / (np.sqrt(1-zeta**2)))

        # Set up the impulse time spacing
        shaperdeltaT = np.pi / (wn*np.sqrt(1-(zeta)**2))

        # Define the impulse times
        times = np.array([[0.0], [shaperdeltaT]])

        # Define the shaper impulse amplitudes
        amps = np.array([[1.0 / (1 + K)], [K / (1 + K)]])

        shaper = np.hstack((times, amps))
        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper

class ZVD(Shaper):
    """ Class describing a ZVD shaper """
    type = 'ZVD'
    amp_type = 'Positive'
    isPositive = True

    def solve_for_shaper(self, frequency, zeta, deltaT):
        """ Return the shaper impulse amplitudes and times

        Arguments:
          frequency : the design frequency for the shaper (Hz)
          zeta : the design damping ratio for the shaper
          deltaT : the samping time (s), used for the digitized version

        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """
        wn = frequency * HZ_TO_RADS
        K = np.exp(-zeta*np.pi / (np.sqrt(1-zeta**2)))

        # Set up the impulse time spacing
        shaperdeltaT = np.pi / (wn*np.sqrt(1-(zeta)**2))

        # Define the impulse times
        times = np.array([[0.0], [shaperdeltaT], [2.0*shaperdeltaT]])

        # Define the shaper impulse amplitudes
        amps = np.array([[1.0 / (1 + 2*K + K**2)],
                         [2.0*K / (1 + 2*K + K**2)],
                         [K**2 / (1 + 2*K + K**2)]])

        shaper = np.hstack((times, amps))
        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper



class ZVDD(Shaper):
    """ Class describing a ZVDD shaper """
    type = 'ZVDD'
    amp_type = 'Positive'
    isPositive = True

    def solve_for_shaper(self, frequency, zeta, deltaT):
        """ Return the shaper impulse amplitudes and times

        Arguments:
          frequency : the design frequency for the shaper (Hz)
          zeta : the design damping ratio for the shaper
          deltaT : the samping time (s), used for the digitized version

        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """
        wn = frequency * HZ_TO_RADS
        K = np.exp(-zeta*np.pi / (np.sqrt(1-zeta**2)))

        # Set up the impulse time spacing
        shaperdeltaT = np.pi / (wn*np.sqrt(1-(zeta)**2))

        # Define the impulse times
        times = np.array([[0.0], [shaperdeltaT], [2.0*shaperdeltaT], [3.0*shaperdeltaT]])

        # Define the shaper impulse amplitudes
        amps = np.array([[1.0 / (1 + 3*K + 3*K**2 + K**3)],
                         [3.0*K / (1 + 3*K + 3*K**2 + K**3)],
                         [3*K**2 / (1 + 3*K + 3*K**2 + K**3)],
                         [K**3 / (1 + 3*K + 3*K**2 + K**3)]])

        shaper = np.hstack((times, amps))
        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper



class ZVDDD(Shaper):
    """ Class describing a ZVDDD shaper """
    type = 'ZVDDD'
    amp_type = 'Positive'
    isPositive = True

    def solve_for_shaper(self, frequency, zeta, deltaT):
        """ Return the shaper impulse amplitudes and times

        Arguments:
          frequency : the design frequency for the shaper (Hz)
          zeta : the design damping ratio for the shaper
          deltaT : the samping time (s), used for the digitized version

        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """
        wn = frequency * HZ_TO_RADS
        K = np.exp(-zeta*np.pi / (np.sqrt(1-zeta**2)))

        # Set up the impulse time spacing
        shaperdeltaT = np.pi / (wn*np.sqrt(1-(zeta)**2))

        # Define the impulse times
        times = np.array([[0.0], [shaperdeltaT], [2.0*shaperdeltaT],
                          [3.0*shaperdeltaT], [4.0*shaperdeltaT]])

        # Define the shaper impulse amplitudes
        amps = np.array([[1.0 / (1 + 4*K + 6*K**2 + 4*K**3 + K**4)],
                         [4.0*K / (1 + 4*K + 6*K**2 + 4*K**3 + K**4)],
                         [6.0*K**2 / (1 + 4*K + 6*K**2 + 4*K**3 + K**4)],
                         [4.0*K**3 / (1 + 4*K + 6*K**2 + 4*K**3 + K**4)],
                         [K**4 / (1 + 4*K + 6*K**2 + 4*K**3 + K**4)]])

        shaper = np.hstack((times, amps))
        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper



#----- UM-ZV-Form Shapers (ZV, ZVD, ZVDD, ...) --------------------------------

class UMZV(Shaper):
    """ Class describing a UM-ZV shaper """
    type = 'UM-ZV'
    amp_type = 'Negative'
    isPositive = False

    def solve_for_shaper(self, frequency, zeta, deltaT):
        """ Return the shaper impulse amplitudes and times

        Arguments:
          frequency : the design frequency for the shaper (Hz)
          zeta : the design damping ratio for the shaper
          deltaT : the samping time (s), used for the digitized version

        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """
        wn = frequency * HZ_TO_RADS

        tau = 1.0 / frequency

        if zeta > 0.4:
            warnings.warn('\n \nWARNING: Damping Ratio is probably too large.\n')

        # Define the impulse times
        times = np.array([[0.0],
                          [(0.16658 + 0.29277 * zeta + 0.075438 * zeta**2 + 0.21335 * zeta**3) * tau],
                          [(0.33323 + 0.0053322 * zeta + 0.17914 * zeta**2 + 0.20125 * zeta**3) * tau]])

        # Define the shaper impulse amplitudes
        amps = np.array([[1.0],[-1.0],[1.0]])

        shaper = np.hstack((times, amps))
        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper


class UMZVD(Shaper):
    """ Class describing a UM-ZVD shaper """
    type = 'UM-ZVD'
    amp_type = 'Negative'
    isPositive = False

    def solve_for_shaper(self, frequency, zeta, deltaT):
        """ Return the shaper impulse amplitudes and times

        Arguments:
          frequency : the design frequency for the shaper (Hz)
          zeta : the design damping ratio for the shaper
          deltaT : the samping time (s), used for the digitized version

        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """
        wn = frequency * HZ_TO_RADS

        tau = 1.0 / frequency

        if zeta > 0.4:
            warnings.warn('\n \nWARNING: Damping Ratio is probably too large.\n')

        # Define the impulse times
        times = np.array([[0.0],
                          [(0.08945 + 0.28411 * zeta + 0.23013*zeta**2 + 0.16401*zeta**3) * tau],
                          [(0.36613 - 0.08833 * zeta + 0.24048*zeta**2 + 0.17001*zeta**3) * tau],
                          [(0.64277 + 0.29103 * zeta + 0.23262*zeta**2 + 0.43784*zeta**3) * tau],
                          [(0.73228 + 0.00992 * zeta + 0.49385*zeta**2 + 0.38633*zeta**3) * tau]])

        # Define the shaper impulse amplitudes
        amps = np.array([[1.0],[-1.0],[1.0],[-1.0],[1.0]])

        shaper = np.hstack((times, amps))
        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper



#----- Positive EI-Form Shapers (EI, 2-Hump EI, ...) --------------------------
class Tolerable_Level_Shapers(Shaper):
    """ Parent class for all tolerable vibation shapers (EI, SI, etc) """
    def __init__(self, frequency, zeta, Vtol=0.05, deltaT = 0.01):
        """ Shaper Initialization function

        Overrides the Shaper class __init__ to add Vtol

        Parses the user inputs and calls solve_for_shaper method
        to get amplitudes and times

        Arguments:
          frequency : the design frequency for the shaper (Hz)
          zeta : the design damping ratio for the shaper
          Vtol : the tolerable level of vibration, default is 5% = 0.05
          deltaT : the samping time (s), used for the digitized version, default = 0.01
        """
        self.design_freq = frequency
        self.design_damping = zeta
        self.design_deltaT = deltaT
        self.Vtol = Vtol
        self.shaper, self.digitized_shaper = self.solve_for_shaper(frequency, zeta, Vtol, deltaT)
        self.times = self.shaper[:,0]
        self.amps = self.shaper[:,1]
        self.duration = self.times[-1]

    def __str__(self):
        """ Set up pretty printing of the shaper """
        type = 'Shaper Type \t \t \t {}\n'.format(self.type)
        designfreq = 'Design Frequency \t \t {:0.4f}\t Hz\n'.format(self.design_freq)
        designdamp = 'Design Damping Ratio \t \t {:0.4f}\n'.format(self.design_damping)
        designVtol = 'Design Vtol \t \t \t {:0.4f}\t % \n'.format(self.Vtol*100)
        duration = 'Duration \t \t \t {:0.4f}\t s \n'.format(self.duration)
        shaper = '\n' + '     ti      Ai \n{}\n'.format(self.shaper)

        return '\n' + type + designfreq + designdamp + designVtol + duration + shaper


class EI(Tolerable_Level_Shapers):
    """ Class describing a EI shaper """
    type = 'EI'
    amp_type = 'Positive'
    isPositive = True


    def solve_for_shaper(self, frequency, zeta, Vtol, deltaT):
        """ Return the shaper impulse amplitudes and times

        Arguments:
          frequency : the design frequency for the shaper (Hz)
          zeta : the design damping ratio for the shaper
          deltaT : the samping time (s), used for the digitized version

        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """
        wn = frequency * HZ_TO_RADS
        wd =  wn * np.sqrt(1-zeta**2)

        # Set up the impulse time spacing
        shaperdeltaT = np.pi / (wn*np.sqrt(1-(zeta)**2))

        # Define the impulse times
        times = np.array([[0.0],
                          [2.0*np.pi*(0.499899+0.461586*Vtol*zeta + 4.26169*Vtol*zeta**2 + 1.75601*Vtol*zeta**3 + 8.57843*Vtol**2*zeta - 108.644*Vtol**2*zeta**2 + 336.989*Vtol**2*zeta**3) / wd],
                          [2.0 * np.pi/wd]])

        # Define the shaper impulse amplitudes
        amps = np.array([[0.249684 + 0.249623*Vtol + 0.800081*zeta + 1.23328*Vtol*zeta + 0.495987*zeta**2 + 3.17316*Vtol*zeta**2],
                         [0.0],
                         [0.251489 + 0.21474*Vtol - 0.832493*zeta + 1.41498*Vtol*zeta + 0.851806*zeta**2 - 4.90094*Vtol*zeta**2]])

        # Now add the 2nd impulse
        amps[1] = 1.0 - (amps[0] + amps[2])

        shaper = np.hstack((times, amps))
        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper



class EI2HUMP(Tolerable_Level_Shapers):
    """ Class describing a Two-hump EI shaper """
    type = 'Two-Hump EI'
    amp_type = 'Positive'
    isPositive = True


    def solve_for_shaper(self, frequency, zeta, Vtol, deltaT):
        """ Return the shaper impulse amplitudes and times

        Arguments:
          frequency : the design frequency for the shaper (Hz)
          zeta : the design damping ratio for the shaper
          deltaT : the samping time (s), used for the digitized version

        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """
        wn = frequency * HZ_TO_RADS
        wd =  wn * np.sqrt(1-zeta**2)

        # Set up the impulse time spacing
        tau = 1.0 / frequency

        if zeta == 0.0:
            X = (Vtol**2 * (np.sqrt(1-Vtol**2)+1))**(1.0/3)

            # Define the impulse times
            times = np.array([[0.0],
                              [0.5 * tau],
                              [tau],
                              [1.5 * tau]])

            # Define the shaper impulse amplitudes
            amps = np.array([[(3*X**2 + 2*X + 3*Vtol**2) / (16*X)],
                             [0.5 - (3*X**2 + 2*X + 3*Vtol**2) / (16*X)],
                             [0.5 - (3*X**2 + 2*X + 3*Vtol**2) / (16*X)],
                             [(3*X**2 + 2*X + 3*Vtol**2) / (16*X)]])
        else:
             # Define the impulse times
            times = np.array([[0.0],
                              [(0.4989+0.1627*zeta-0.54262*zeta**2+6.1618*zeta**3) * tau],
                              [(0.99748+0.18382*zeta-1.5827*zeta**2+8.1712*zeta**3) * tau],
                              [(1.4992-0.09297*zeta-0.28338*zeta**2+1.8571*zeta**3) * tau]])

            # Define the shaper impulse amplitudes
            amps = np.array([[0.16054+0.76699*zeta+2.2656*zeta**2-1.2275*zeta**3],
                             [0.33911+0.45081*zeta-2.5808*zeta**2+1.7365*zeta**3],
                             [0.34089-0.61533*zeta-0.68765*zeta**2+0.42261*zeta**3],
                             [0.0]])

            amps[3] = 1.0 - amps[0] - amps[1] - amps[2]


        shaper = np.hstack((times, amps))
        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper


class EI3HUMP(Tolerable_Level_Shapers):
    """ Class describing a Three-hump EI shaper """
    type = 'Three-Hump EI'
    amp_type = 'Positive'
    isPositive = True


    def solve_for_shaper(self, frequency, zeta, Vtol, deltaT):
        """ Return the shaper impulse amplitudes and times

        Arguments:
          frequency : the design frequency for the shaper (Hz)
          zeta : the design damping ratio for the shaper
          deltaT : the samping time (s), used for the digitized version

        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """
        wn = frequency * HZ_TO_RADS
        wd =  wn * np.sqrt(1-zeta**2)

        # Set up the impulse time spacing
        tau = 1.0 / frequency

        # Define the impulse times
        times = np.array([[0.0],
                          [0.5 * tau],
                          [1.0 * tau],
                          [1.5 * tau],
                          [2.0 * tau]])

        # Define the shaper impulse amplitudes
        amps = np.array([[(1+3*Vtol+2*np.sqrt(2*(Vtol**2+Vtol))) / 16],
                         [(1-Vtol) / 4],
                         [0.0],
                         [(1-Vtol) / 4],
                         [(1+3*Vtol+2*np.sqrt(2*(Vtol**2+Vtol))) / 16]])

        amps[2] = 1.0 - 2 * (amps[0] + amps[1])


        shaper = np.hstack((times, amps))
        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper



#----- Negative EI-Form Shapers (EI, 2-Hump EI, ...) --------------------------

class UMEI(Tolerable_Level_Shapers):
    """ Class describing a UM-EI shaper """
    type = 'UM-EI'
    amp_type = 'Negative'
    isPositive = False

    def solve_for_shaper(self, frequency, zeta, Vtol, deltaT):
        """ Return the shaper impulse amplitudes and times

        Arguments:
          frequency : the design frequency for the shaper (Hz)
          zeta : the design damping ratio for the shaper
          Vtol : The tolerable level of vibration 0.05 = 5%
          deltaT : the samping time (s), used for the digitized version

        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """
        wn = frequency * HZ_TO_RADS

        tau = 1.0 / frequency

        if zeta > 0.4:
            warnings.warn('\n \nWARNING: Damping Ratio is probably too large.\n')

        # Define the impulse times
        times = np.zeros((5,1))
        if Vtol == 0.05:
          times[0] = 0
          times[1] = (0.09374 + 0.31903 * zeta + 0.13582 * zeta**2 + 0.65274 * zeta**3) * tau
          times[2] = (0.36798 - 0.05894 * zeta + 0.13641 * zeta**2 + 0.63266 * zeta**3) * tau
          times[3] = (0.64256 + 0.28595 * zeta + 0.26334 * zeta**2 + 0.24999 * zeta**3) * tau
          times[4] = (0.73664 + 0.00162 * zeta + 0.52749 * zeta**2 + 0.19208 * zeta**3) * tau

        elif Vtol == 0.0125:
          times[0] = 0
          times[1] = (0.09051 + 0.29315 * zeta + 0.20436 * zeta**2 + 0.29053 * zeta**3) * tau
          times[2] = (0.36658 - 0.081044 * zeta + 0.21524 * zeta**2 + 0.27994 * zeta**3) * tau
          times[3] = (0.64274 + 0.28822 * zeta + 0.25424 * zeta**2 + 0.34977 * zeta**3) * tau
          times[4] = (0.73339 + 0.006322 * zeta + 0.51595 * zeta**2 + 0.29764 * zeta**3) * tau
        else:
           warnings.warn('Only V = 0.05 or V = 0.0125 can be used at this time.\n')

        # Define the shaper impulse amplitudes
        amps = np.array([[1.0],[-1.0],[1.0],[-1.0],[1.0]])

        shaper = np.hstack((times, amps))
        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper


class UM2EI(Tolerable_Level_Shapers):
    """ Class describing a  UM-Two-Hump EI shaper """
    type = 'UM-Two-Hump EI'
    amp_type = 'Negative'
    isPositive = False

    def solve_for_shaper(self, frequency, zeta, Vtol, deltaT):
        """ Return the shaper impulse amplitudes and times

        Arguments:
          frequency : the design frequency for the shaper (Hz)
          zeta : the design damping ratio for the shaper
          Vtol : The tolerable level of vibration 0.05 = 5%
          deltaT : the samping time (s), used for the digitized version

        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """
        wn = frequency * HZ_TO_RADS

        tau = 1.0 / frequency

        if zeta > 0.4:
            warnings.warn('\n \nWARNING: Damping Ratio is probably too large.\n')

        # Define the impulse times
        times = np.zeros((7,1))

        if Vtol == 0.05:
          times[0] = 0.0
          times[1] = (0.059696 + 0.3136 * zeta + 0.31759 * zeta**2 + 1.5872 * zeta**3) * tau
          times[2] = (0.40067 - 0.085698 * zeta + 0.14685 * zeta**2 + 1.6059 * zeta**3) * tau
          times[3] = (0.59292 + 0.38625 * zeta + 0.34296 * zeta**2 + 1.2889 * zeta**3) * tau
          times[4] = (0.78516 - 0.088283 * zeta + 0.54174 * zeta**2 + 1.3883 * zeta**3) * tau
          times[5] = (1.1264 + 0.20919 * zeta + 0.44217 * zeta**2 + 0.30771 * zeta**3) * tau
          times[6] = (1.1864 - 0.029931 * zeta + 0.79859 * zeta**2 + 0.10478 * zeta**3) * tau

        elif Vtol == 0.0125:
          times[0] = 0
          times[1] = (0.052025 + 0.25516 * zeta + 0.33418 * zeta**2 + 0.70993 * zeta**3) * tau
          times[2] = (0.39946 - 0.13396 * zeta + 0.23553 * zeta**2 + 0.59066 * zeta**3) * tau
          times[3] = (0.58814 + 0.33393 * zeta + 0.4242 * zeta**2 + 0.4844 * zeta**3) * tau
          times[4] = (0.77682 - 0.13392 * zeta + 0.61271 * zeta**2 + 0.63186 * zeta**3) * tau
          times[5] = (1.1244 + 0.21132 * zeta + 0.55855 * zeta**2 + 0.12884 * zeta**3) * tau
          times[6] = (1.1765 - 0.016188 * zeta + 0.9134 * zeta**2 - 0.068185 * zeta**3) * tau
        else:
           warnings.warn('\n \nOnly V = 0.05 or V = 0.0125 can be used at this time.\n')

        # Define the shaper impulse amplitudes
        amps = np.array([[1.0],[-1.0],[1.0],[-1.0],[1.0],[-1.0],[1.0]])

        shaper = np.hstack((times, amps))
        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper


class UM3EI(Tolerable_Level_Shapers):
    """ Class describing a  UM-Three-Hump EI shaper """
    type = 'UM-Three-Hump EI'
    amp_type = 'Negative'
    isPositive = False

    def solve_for_shaper(self, frequency, zeta, Vtol, deltaT):
        """ Return the shaper impulse amplitudes and times

        Arguments:
          frequency : the design frequency for the shaper (Hz)
          zeta : the design damping ratio for the shaper
          Vtol : The tolerable level of vibration 0.05 = 5%
          deltaT : the samping time (s), used for the digitized version

        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """
        wn = frequency * HZ_TO_RADS

        tau = 1.0 / frequency

        if zeta > 0.4:
            warnings.warn('\n \nWARNING: Damping Ratio is probably too large.\n')

        # Define the impulse times
        times = np.zeros((9,1))

        if Vtol == 0.05:
            times[0] = 0
            times[1] = (0.042745 + 0.31845 * zeta + 0.46272 * zeta**2 + 3.3763 * zeta**3) * tau
            times[2] = (0.42418 - 0.05725 * zeta + 0.049893 * zeta**2 + 3.9768 * zeta**3) * tau
            times[3] = (0.56353 + 0.48068 * zeta + 0.38047 * zeta**2 + 4.2431 * zeta**3) * tau
            times[4] = (0.83047 - 0.097848 * zeta + 0.34048 * zeta**2 + 4.4245 * zeta**3) * tau
            times[5] = (1.0976 + 0.38825 * zeta + 0.3529 * zeta**2 + 2.9484 * zeta**3) * tau
            times[6] = (1.2371 - 0.08706 * zeta + 0.81706 * zeta**2 + 2.8367 * zeta**3) * tau
            times[7] = (1.6189 + 0.099638 * zeta + 0.4278 * zeta**2 + 1.3151 * zeta**3) * tau
            times[8] = (1.6619 - 0.097105 * zeta + 0.80045 * zeta**2 + 1.0057 * zeta**3) * tau

        elif Vtol == 0.0125:
            times[0] = 0
            times[1] = (0.032665 + 0.23238 * zeta + 0.33164 * zeta**2 + 1.8423 * zeta**3) * tau
            times[2] = (0.42553 - 0.12863 * zeta + 0.052687 * zeta**2 + 1.7964 * zeta**3) * tau
            times[3] = (0.55502 + 0.36614 * zeta + 0.50008 * zeta**2 + 1.7925 * zeta**3) * tau
            times[4] = (0.82296 - 0.19383 * zeta + 0.45316 * zeta**2 + 2.0989 * zeta**3) * tau
            times[5] = (1.091 + 0.31654 * zeta + 0.46985 * zeta**2 + 1.2683 * zeta**3) * tau
            times[6] = (1.2206 - 0.14831 * zeta + 0.93082 * zeta**2 + 1.2408 * zeta**3) * tau
            times[7] = (1.6137 + 0.1101 * zeta + 0.68318 * zeta**2 + 0.18725 * zeta**3) * tau
            times[8] = (1.6466 - 0.063739 * zeta + 1.0423 * zeta**2 - .10591 * zeta**3) * tau
        else:
           warnings.warn('\n \nOnly V = 0.05 or V = 0.0125 can be used at this time.\n')

        # Define the shaper impulse amplitudes
        amps = np.array([[1.0],[-1.0],[1.0],[-1.0],[1.0],[-1.0],[1.0],[-1.0],[1.0]])

        shaper = np.hstack((times, amps))
        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper


#------ 2-Mode Shapers ---------------------------------------------------------
class Two_Mode_Shaper(Shaper):
    """ Parent class for all Two-Mode Shapers """

    def __init__(self, frequency1, zeta1, frequency2, zeta2, deltaT = 0.01):
        """ Shaper Initialization function

        Overrides the Shaper class __init__ to add 2nd mode parameters

        Parses the user inputs and calls solve_for_shaper method
        to get amplitudes and times

        Arguments:
          frequency1 : the design frequency for  first mode (Hz)
          zeta1 : damping ratio for the first mode
          frequency2 : design frequency for the second mode (Hz)
          zeta2 : damping ratio for the second mode
          deltaT : the samping time (s), used for the digitized version, default = 0.01
        """
        self.design_freq_1 = frequency1
        self.design_freq_2 = frequency2
        self.design_damping_1 = zeta1
        self.design_damping_2 = zeta2
        self.design_deltaT = deltaT
        self.shaper, self.digitized_shaper = self.solve_for_shaper(frequency1, zeta1, frequency2, zeta2, deltaT)
        self.times = self.shaper[:,0]
        self.amps = self.shaper[:,1]
        self.duration = self.times[-1]


    def __str__(self):
        """ Set up pretty printing of the shaper """
        type = 'Shaper Type \t \t \t {}\n'.format(self.type)
        designfreq1 = 'Mode 1 Design Frequency \t {:0.4f}\t Hz\n'.format(self.design_freq_1)
        designdamp1 = 'Mode 1 Damping Ratio \t \t {:0.4f}\n'.format(self.design_damping_1)
        designfreq2 = 'Mode 2 Design Frequency \t {:0.4f}\t Hz\n'.format(self.design_freq_2)
        designdamp2 = 'Mode 2 Damping Ratio \t \t {:0.4f}\n'.format(self.design_damping_2)
        duration = 'Duration \t \t \t {:0.4f}\t s \n'.format(self.duration)
        shaper = '\n' + '     ti      Ai \n{}\n'.format(self.shaper)

        return '\n' + type + designfreq1 + designdamp1 + designfreq2 + designdamp2 + duration + shaper

class ZV_2mode(Two_Mode_Shaper):
    """ Class describing a two-mode ZV shaper - created by convolving two ZV shapers"""
    type = 'Two-Mode ZV'
    amp_type = 'Positive'
    isPositive = True

    def solve_for_shaper(self, frequency1, zeta1, frequency2, zeta2, deltaT):
        """ Return the shaper impulse amplitudes and times

        Arguments:
          The created instance, uses instance variables for calculation

        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """

        zv1 = ZV(frequency1, zeta1, deltaT)
        zv2 = ZV(frequency2, zeta2, deltaT)

        shaper = seqconv(zv1.shaper, zv2.shaper)

        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper

class ZVD_2mode(Two_Mode_Shaper):
    """ Class describing a two-mode ZVD shaper - created by convolving two ZVD shapers"""
    type = 'Two-Mode ZVD'
    amp_type = 'Positive'
    isPositive = True

    def solve_for_shaper(self, frequency1, zeta1, frequency2, zeta2, deltaT):
        """ Return the shaper impulse amplitudes and times

        Arguments:
          The created instance, uses instance variables for calculation

        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """

        zvd1 = ZVD(frequency1, zeta1, deltaT)
        zvd2 = ZVD(frequency2, zeta2, deltaT)

        shaper = seqconv(zvd1.shaper, zvd2.shaper)

        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper


class ZVDD_2mode(Two_Mode_Shaper):
    """ Class describing a two-mode ZVDD shaper - created by convolving two ZVDD shapers"""
    type = 'Two-Mode ZVDD'
    amp_type = 'Positive'
    isPositive = True

    def solve_for_shaper(self, frequency1, zeta1, frequency2, zeta2, deltaT):
        """ Return the shaper impulse amplitudes and times

        Arguments:
          The created instance, uses instance variables for calculation

        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """

        zvdd1 = ZVDD(frequency1, zeta1, deltaT)
        zvdd2 = ZVDD(frequency2, zeta2, deltaT)

        shaper = seqconv(zvdd1.shaper, zvdd2.shaper)

        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper

class ZVDDD_2mode(Two_Mode_Shaper):
    """ Class describing a two-mode ZVDDD shaper - created by convolving two ZVDDD shapers"""
    type = 'Two-Mode ZVDDD'
    amp_type = 'Positive'
    isPositive = True

    def solve_for_shaper(self, frequency1, zeta1, frequency2, zeta2, deltaT):
        """ Return the shaper impulse amplitudes and times

        Arguments:
          The created instance, uses instance variables for calculation

        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """

        zvddd1 = ZVDDD(frequency1, zeta1, deltaT)
        zvddd2 = ZVDDD(frequency2, zeta2, deltaT)

        shaper = seqconv(zvddd1.shaper, zvddd2.shaper)

        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper

class ZV_EI_2mode(Two_Mode_Shaper):
    """ Class describing a two-mode ZV shaper - created by convolving two ZV shapers"""
    type = 'Two-Mode ZV'
    amp_type = 'Positive'
    isPositive = True

    def solve_for_shaper(self, frequency1, zeta1, frequency2, zeta2, deltaT):
        """ Return the shaper impulse amplitudes and times
        Arguments:
          The created instance, uses instance variables for calculation
        Returns:
          shaper : The shaper solution
          digitized_shaper : the digitized version of the shaper
        """

        zv1 = ZV(frequency1, zeta1, deltaT)
        ei = EI(frequency2, zeta2, deltaT)

        shaper = seqconv(zv1.shaper, ei.shaper)

        digitized_shaper = digseq(shaper, deltaT)

        return shaper, digitized_shaper


