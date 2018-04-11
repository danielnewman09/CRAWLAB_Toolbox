#------------------------------------------------------------------------------
# examples.py
#
# Demonstrate the crawlab_plots package using some simple 2D and 3D plots 
#
# Created: 4/4/18 - Daniel Newman -- dmn3669@louisiana.edu
#
# Modified:
#   * 4/4/18 - DMN -- dmn3669@louisiana.edu
#			- Added documentation for this script
#
#------------------------------------------------------------------------------

import numpy as np

# Import the crawlab_plots functions
#from crawlab_toolbox.plotting import generate_plot
#from crawlab_toolbox.plotting import plot_3d

import plotting

# Specify the folder where we will save the pdfs of the plots
folder = 'Figures/'

# Generate some basic responses
DT = 0.01
TMAX = 5.
TIME = np.arange(0,TMAX,DT)

def response_1(t):
	return np.sin(t)

def response_2(t):
	return 2 * np.sin(t)

def response_3(t):
	return 2 * np.sin(2 * t)

# combine the responses and labels to put on our 2D plot
responses = np.vstack((response_1(TIME),response_2(TIME),response_3(TIME)))
labels = ['First', 'Second', 'Third']

# Create a 2D plot
plotting.generate_plot(TIME, 	# X - coordinate on the plot
			  responses,	# Y - coordinates on the plot
			  labels,	# Labels for the plot legend
			  'Time (s)',	# X - axis label
			  'Position (m)',	# Y - axis label
			  filename='Test_Response',	 # Plot filename
			  folder=folder,	# Specify the folder where the filename will be saved
			  num_col=2,	# Specify the number of columns in the legend
			  legend_loc='upper right',	# Specify the location of the legend
			  ymax=0.5,
			  transparent=True,
			  save_plot=True
			 )

# load and parse some pre-generated 3-Dimensional data
data = np.genfromtxt('siic_phase_freq_pulse.txt',skip_header=1,delimiter=',')
freq = data[:,0]
phase = data[:,1]
amp = data[:,2]

# Create a 3D plot
plotting.plot_3d(freq,phase,amp,
		r'$\frac{\omega_n}{\omega_m}$',
		r'$\theta_e$',
		r'Percent Vibration',
		folder=folder,
		filename='SIIC_Phase_Freq_Sens',
		elevation=40,
		azimuth=250,
		xticks=0.1,yticks=30,zticks=20,
		labelsize=30,labelpad=30,
		transparent=False,
		save_plot=True)
