import matplotlib as mpl

# LINES
mpl.rcParams['lines.linewidth'] = 2.0     # line width in points
mpl.rcParams['lines.dash_capstyle'] = 'round'          # butt|round|projecting

# FONT
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.weight'] = 'normal'
#font.size           : 12.0
mpl.rcParams['font.serif'] = 'DejaVu Serif', 'CMU Serif', 'Bitstream Vera Serif', 'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Times New Roman', 'Times', 'Palatino', 'Charter', 'serif'

# TEXT
mpl.rcParams['text.hinting_factor'] = 8 # Specifies the amount of softness for hinting in the
                         # horizontal direction.  A value of 1 will hint to full
                         # pixels.  A value of 2 will hint to half pixels etc.
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preview'] = True
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath} \boldmath"]


# AXES
mpl.rcParams['axes.labelsize'] = 22  # fontsize of the x any y labels
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
