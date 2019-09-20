
#------------------------------------------------------------------------------
# Plotting.py
#
# Create publication-ready 3D and 2D plots using matplotlib
# 
#
# Created: 4/4/18 - Daniel Newman -- dmn3669@louisiana.edu
#
# Modified:
#   * 4/4/18 - DMN -- dmn3669@louisiana.edu
#           - Added documentation for this script
#
#------------------------------------------------------------------------------


import matplotlib as mpl

from matplotlib import pyplot as plt
from matplotlib import ticker as mtick

from matplotlib import rc
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D

import numpy as np

import os

from scipy.interpolate import griddata

from cycler import cycler

### MATPLOTLIBRC FORMAT
#mpl.rcParams['backend'] = 'MacOSX'

# LINES
mpl.rcParams['lines.linewidth'] = 2.0     # line width in points
mpl.rcParams['lines.dash_capstyle'] = 'round'          # butt|round|projecting

# FONT
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.weight'] = 'normal'
#font.size           : 12.0
mpl.rcParams['font.serif'] = 'CMU Serif', 'Bitstream Vera Serif', 'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Times New Roman', 'Times', 'Palatino', 'Charter', 'serif'

# TEXT
mpl.rcParams['text.hinting_factor'] = 8 # Specifies the amount of softness for hinting in the
                         # horizontal direction.  A value of 1 will hint to full
                         # pixels.  A value of 2 will hint to half pixels etc.
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preview'] = True


# AXES
mpl.rcParams['axes.labelsize'] = 18  # fontsize of the x any y labels
mpl.rcParams['axes.labelweight'] = 'medium'  # weight of the x and y labels
mpl.rcParams['axes.prop_cycle'] = cycler('color', ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628'])
                      ## color cycle for plot lines  as list of string
                      ## colorspecs: single letter, long name, or web-style hex
                      ## Note the use of string escapes here ('1f77b4', instead of 1f77b4)

# TICKS
mpl.rcParams['xtick.labelsize'] = 18      # fontsize of the tick labels
mpl.rcParams['ytick.labelsize'] = 18      # fontsize of the tick labels

# GRID
mpl.rcParams['grid.color'] = '0.75'   # grid color
mpl.rcParams['grid.linestyle'] = ':'       # dotted

# LEGEND
mpl.rcParams['legend.fancybox'] = True  # if True, use a rounded box for the
                               # legend, else a rectangle
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['legend.borderaxespad'] = 0.1   # the border between the axes and legend edge in fraction of fontsize

# FIGURE
mpl.rcParams['figure.figsize'] = 6,4    # figure size in inches
mpl.rcParams['figure.subplot.left'] = 0.2  # the left side of the subplots of the figure
mpl.rcParams['figure.subplot.right'] = 0.9    # the right side of the subplots of the figure
mpl.rcParams['figure.subplot.bottom'] = 0.2    # the bottom of the subplots of the figure
mpl.rcParams['figure.subplot.top'] = 0.85    # the top of the subplots of the figure
mpl.rcParams['figure.subplot.wspace'] = 0.2    # the amount of width reserved for blank space between subplots
mpl.rcParams['figure.subplot.hspace'] = 0.2    # the amount of height reserved for white space between subplots

# SAVEFIG
mpl.rcParams['savefig.dpi'] = 600      # figure dots per inch
mpl.rcParams['savefig.format'] = 'svg'      # png, ps, pdf, svg

# To generically create multiple plots
plot_linestyle = ['-','--','-.',':']

marker_weight = [30,60,40,40]
plot_markerstyle = ['o','x','v','^']

def set_lims(ax,X,Y,xmin,xmax,ymin,ymax):
    
    if xmax == 0.:
        xmax += 0.3

    # Determine the lower and upper bounds of the horizontal axis
    if xmax == None:
        xmax = np.amax(X)
    if xmin == None:
        xmin = np.amin(X)

    # Set the limits of the plot
    plt.xlim(xmin, xmax)

    if not isinstance(ymax,np.ndarray):
        # Set the window limits
        plt.ylim(np.amin(Y) - ymin * abs(np.amin(Y)),
                 np.amax(Y) + ymax * abs(np.amax(Y)-np.amin(Y)))
    else:
        plt.ylim(ymin[0],ymax[0])

# Container for all plots
def generate_plot(
                X,Y,labels,xlabel,ylabel,
                plot_type = 'Plot',
                ymax = 0.1, 
                ymin = 0.1,
                xmax = None,
                xmin = None,
                tick_increment = None,
                showplot = False,
                save_plot = False,
                log_y = False,
                log_x = False,
                transparent = False,
                grid = False, 
                folder = None,
                filename = 'Plot',
                num_col = 2,
                legend_loc = 'upper right',
                experimental_args = None,
                xlabelpad = 5,       
                hide_origin = False,  
                for_notebook=False,
                template='publication',
                file_type='pdf'
                 ):    
    '''
    This is a function which accepts a series of data and plots it based on preset defaults
    as well as user-defined, custom inputs.
    
    Creator : Daniel Newman - Danielnewman09@gmail.com
    
    Mandatory Inputs:
        X - x-coordinate of the plot
        Y - y-coordinates of the plot. Must have an axis of the same length as X
        labels - list of strings which form the labels we will use for the legend
        xlabel - Label along the X-axis
        ylabel - Label along the Y-axis
    
    Optional Inputs:
        plot_type - String indicating the type of plot
        ymax - multiplicative value for the maximum Y value
        ymin - multiplicative value for the minimum Y value
        xmax - maximum X value
        xmin - minimum X value
        tick_increment - spacing between y-axis ticks
        showplot - boolean indicating whether the plot is displayed
        log_y - boolean indicating whether the y-axis should be on a log scale
        transparent - boolean indicating whether to save a transparent .png
        grid - boolean indicating whether to show the grid
        folder - subfolder in which to save the figure
        filename - string indicating the name of the saved file
        num_col - number of columns in the legend
        legend_loc - string indicating the location of the legend
        experimental_args - experimental values to show on the plot
        xlabelpad - spacing between the x-axis and the x-label
    '''
    
    if template.lower() == 'large':
        plt.figure(figsize=(10,6.67))
    elif template.lower() == 'wide':
        plt.figure(figsize=(12,4))
    else:
        plt.figure()

    # Customize the axes
    ax = plt.gca()
    
    # Make sure the Y data is at least 2-D
    Y = np.atleast_2d(Y)
    
    # Ensure the compatibility of the X and Y data
    if Y.shape[0] != X.shape[0] and Y.shape[1] != X.shape[0]:
        raise ValueError(
            '''The Shape of X, [{}], is not compatible 
             with the shape of Y, [{}]...\n Exiting'''
            .format(X.shape,Y.shape))
        return 
    elif Y.shape[0] != X.shape[0]: 
        Y = Y.T

    if Y.shape[1] != len(labels):
        raise ValueError('Please ensure the number of legend labels matches the number of data plots.')
    
    if plot_type.lower() == 'plot':
        # Plot all of the available data
        for i in np.arange(0,len(labels)):
            if labels[i].lower() == 'vtol':
                plt.plot(x, Y[:,i],
                color='k',
                linestyle=plot_linestyle[1], # Linestyle given from array at the beginning of this document
                linewidth=1)
            else:
                if log_y:
                    plt.semilogy(X, Y[:,i],
                        label= '{}'.format(labels[i]),
                        linestyle=plot_linestyle[i], # Linestyle given from array at the beginning of this document
                        linewidth=2)    
                else:
                    plt.plot(X, Y[:,i],
                        label= '{}'.format(labels[i]),
                        linestyle=plot_linestyle[i], # Linestyle given from array at the beginning of this document
                        linewidth=2)  

        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')


        if tick_increment is not None:
            loc = mtick.MultipleLocator(base=tick_increment) # this locator puts ticks at regular intervals
            ax.yaxis.set_major_locator(loc)
  

        set_lims(ax,X,Y,xmin,xmax,ymin,ymax)
      

        # Show the grid, if desired
        ax.grid(grid)

        ax.set_axisbelow(True)

        # If we want to plot experimental data
        if experimental_args is not None: 

            if len(np.atleast_2d(data)[:,0]) > 1:

                # This code is for closely grouped experimental data that doesn't ened a box and whisker plot
                means = np.average(data,axis=0)
                maxes = np.amax(data,axis=0)
                mins = np.amin(data,axis=0)
            else:
                means = data
                maxes = data
                mins = data

            plt.errorbar(positions,means,yerr=[maxes-means,means-mins],fmt='D', 
                         ecolor='C1',mfc='C1',mec='C1',capsize=5, capthick=1,lw=1,label='Experimental'
                        )

    elif plot_type.lower() == 'scatter': 
        for i in range(0,len(labels)):
                plt.scatter(X[:,i], Y[:,i],
                    label= '{}'.format(labels[i]),s=marker_weight[i],#facecolors='none',#edgecolors='k',
                    marker=plot_markerstyle[i], # Linestyle given from array at the beginning of this document
                    linewidth=2) 
                
        set_lims(X,Y,xmin,xmax,ymin,ymax,log_y,log_x)
        plt.margins(1)
        
        #ax.spines['left'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_label_position('right')
        ax.yaxis.set_ticks_position('right')

        # set the x-spine (see below for more info on `set_position`)
        ax.spines['right'].set_position('zero')

        # turn off the right spine/ticks
        ax.spines['left'].set_color('none')
        ax.yaxis.tick_right()

        # set the y-spine
        ax.spines['bottom'].set_position('zero')

        # turn off the top spine/ticks
        ax.spines['top'].set_color('none')
        ax.xaxis.tick_bottom()
        
    else:
        raise ValueError('Invalid plot_type value. Please provide a valid plot type')

    if not log_y:
        # X tick locations
        xloc = mtick.MaxNLocator(
                        nbins=7, # Maximum number of bins
                        steps = [1 , 2, 2.5, 5, 10], # valid step increments
                        prune='both').tick_values(*plt.xlim()) 

        # Y tick locations
        yloc = mtick.MaxNLocator(
                        nbins=6, # Maximum number of bins
                        steps = [1, 2, 2.5, 5, 10], # valid step increments
                        prune='upper').tick_values(*plt.ylim())

        if hide_origin:
            # Hide the origin
            yloc = yloc[np.argwhere(yloc != 0.)]
            xloc = xloc[np.argwhere(xloc != 0.)]

        # Set the tickmarks at the given x and y locations
        ax.set_xticks(xloc)
        ax.set_yticks(yloc)
    
    if labels[0]:
        # Show the legend
        ax.legend(ncol=num_col,loc=legend_loc,framealpha=float(not transparent)).get_frame().set_edgecolor('k')
        
    # Create the axis labels
    plt.xlabel('{}'.format(xlabel), labelpad=xlabelpad)
    plt.ylabel('{}'.format(ylabel), labelpad=5)

    # Adjust the page layout filling the page using the new tight_layout command
    plt.tight_layout(pad=1.2) 

    if save_plot:
        if folder is not None:
            # Ensure that the folder we want to save to exists
            if not os.path.exists(folder):
                os.makedirs(folder)

            filename = folder + '/' + filename

        # Save the pdf of the plot    
        if transparent:
            plt.savefig('{}.png'\
                    .format(filename),transparent=transparent)             
        elif file_type == 'pdf':
            plt.savefig('{}.pdf'\
                    .format(filename))    
        elif file_type == 'svg':
            plt.savefig('{}.svg'\
                    .format(filename)) 

    if showplot:
        plt.show()

    # Clear the axes and figure
    plt.clf()
    plt.cla()
    plt.close()

def plot_3d(
    X,Y,Z,
    xlabel,ylabel,zlabel,
    azimuth=225,elevation=30,
    showplot=True,
    save_plot=False,
    folder='Figures/Miscellaneous',
    filename='3d_plot',
    xticks=1,yticks=1,zticks=1,
    enablelog=False,
    labelsize=24,
    labelpad=20,
    rotated=False,
    transparent=False,
    file_type='pdf'):
    '''
    Plot data in three dimensions

   Creator : Daniel Newman - Danielnewman09@gmail.com
    
    Mandatory Inputs:
        X - x-coordinate of the plot
        Y - y-coordinates of the plot
        Z - z-coordinates of the plot
        xlabel - Label along the X-axis
        ylabel - Label along the Y-axis
        zlabel - Label along the Z-axis
    
    Optional Inputs:
        azimuth - rotation of the plot about the z axis
        elevation - vertical rotation of the plot
        tick_increment - spacing between y-axis ticks
        showplot - boolean indicating whether the plot is displayed
        rotated - boolean indicating whether the axis labels are rotated
        transparent - boolean indicating whether to save a transparent .png
        folder - subfolder in which to save the figure
        filename - string indicating the name of the saved file 
    '''

    # Ensure that the folder we want to save to exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create the figure
    fig = plt.figure(figsize=(10,6.67))
    plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)
    ax1 = Axes3D(fig)
    ax1.view_init(elevation,azimuth)

    ax1.xaxis.set_major_locator(mtick.MultipleLocator(xticks))
    ax1.yaxis.set_major_locator(mtick.MultipleLocator(yticks))

    multipliers = {
        r'$(\times 10^{9})$': 1e-9,
        r'$(\times 10^{6})$': 1e-6,
        r'$(\times 10^{3})$': 1e-3,
        r'$(\times 10^{-3})$': 1e3,
        r'$(\times 10^{-6})$': 1e6,
        r'$(\times 10^{-9})$': 1e9,
    }

    if np.abs(np.amax(Z)) < 0.01 or np.abs(np.amin(Z)) > 1000:

        thisKey = ''
        thisValue = 1

        for key,value in multipliers.items():
            if (np.abs(np.amax(Z)) * value) // 1e3 < 1:
                thisKey = key
                thisValue = value

        zlabel += ' ' + thisKey
        Z *= thisValue


    # format the tick labels
    plt.setp(ax1.get_ymajorticklabels(), family='serif',fontsize=22)
    plt.setp(ax1.get_xmajorticklabels(), family='serif',fontsize=22)
    plt.setp(ax1.get_zmajorticklabels(), family='serif',fontsize=22)

    # Put a grid in the background
    ax1.grid(True)
    ax1.xaxis.pane.set_edgecolor('black')
    ax1.yaxis.pane.set_edgecolor('black')

    # Let the background of the plot be white
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    if 200 < azimuth < 240 or 20 < azimuth < 60:
        # adjusts the padding around the 3D plot
        ax1.dist = 11

        # Format the ticks
        ax1.xaxis._axinfo['tick']['inward_factor'] = 0
        ax1.xaxis._axinfo['tick']['outward_factor'] = 0.4
        ax1.yaxis._axinfo['tick']['inward_factor'] = 0
        ax1.yaxis._axinfo['tick']['outward_factor'] = 0.4
        ax1.zaxis._axinfo['tick']['inward_factor'] = 0
        ax1.zaxis._axinfo['tick']['outward_factor'] = 0.4

        # Vertically and horizontally align the tick labels
        [t.set_va('center') for t in ax1.get_yticklabels()]
        [t.set_ha('right') for t in ax1.get_yticklabels()]
        [t.set_va('top') for t in ax1.get_xticklabels()]
        [t.set_ha('center') for t in ax1.get_xticklabels()]
        [t.set_va('top') for t in ax1.get_zticklabels()]
        [t.set_ha('center') for t in ax1.get_zticklabels()]

    else:
        raise ValueError('The specified viewing angle is likely to yield suboptimal results. '
                         'Please choose an azimuth between (200,240) or (20,60).')
        # The tick locations and axis labels are aligned based on being viewed from a certain
        # angle. If you NEED to view the plot from a different angle, you will have to update this
        # alignment. 

    # Create a linear grid for x and y
    yi = np.linspace(min(Y), max(Y))
    xi = np.linspace(min(X), max(X))

    # Interpolate the Z data based on the X and Y grid
    Z = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='cubic')
    X, Y = np.meshgrid(xi, yi)

    if np.any(np.isnan(Z)):
        ax1.scatter(X, Y, Z) 
        plt.show()
        raise ValueError('The requested values cannot be shown as a smooth surface. \n'
                         'Please double-check your data. Generating point cloud of \n '
                         'requested values...')


    else:
        # Plot the surface data
        surf = ax1.plot_surface(
                    X, Y, Z, 
                    rstride=1, linewidth=0, alpha=0.85, 
                    cstride=1,cmap=cm.bwr, shade=False, antialiased=True)

        # Format the color bar
        color_bar = plt.colorbar(surf,shrink=0.5,aspect=8,pad=0.)
        cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
        plt.setp(cbytick_obj,  family='serif',fontsize=22)

    # Set the Z limits
    ax1.set_zlim3d(np.min(Z), np.max(Z))

    # Determine whether the axis labels are rotated
    ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax1.xaxis.set_rotate_label(rotated)  
    ax1.yaxis.set_rotate_label(rotated)  

    # Set the axis labels
    ax1.set_xlabel(xlabel,family='serif',fontsize=labelsize,labelpad=labelpad)
    ax1.set_ylabel(ylabel,family='serif',fontsize=labelsize,labelpad=labelpad)
    ax1.set_zlabel(zlabel,family='serif',fontsize=labelsize,labelpad=15,rotation=90)
    

    if save_plot:
        if folder is not None:
            # Ensure that the folder we want to save to exists
            if not os.path.exists(folder):
                os.makedirs(folder)

            filename = folder + '/' + filename

        # Save the pdf of the plot    
        if transparent:
            plt.savefig('{}.png'\
                    .format(filename),transparent=transparent)             
        elif file_type == 'pdf':
            plt.savefig('{}.pdf'\
                    .format(filename))    
        elif file_type == 'svg':
            plt.savefig('{}.svg'\
                    .format(filename))      

    if showplot:
        plt.show()

    plt.clf()
    plt.cla()
    plt.close()