"""
Demo of TeX rendering.

You can use TeX to render all of your matplotlib text if the rc
parameter text.usetex is set.  This works currently on the agg and ps
backends, and requires that you have tex and the other dependencies
described at http://matplotlib.org/users/usetex.html
properly installed on your system.  The first time you run a script
you will see a lot of output from tex and associated tools.  The next
time, the run may be silent, as a lot of the information is cached in
~/.tex.cache

"""
import matplotlib
matplotlib.use('PS')
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['text.latex.unicode']=True
import pylab as plt
plt.switch_backend('PS')

string = r'z=${value}^{upper}_{lower}$'.format(
                value='{' + str(0.27) + '}',
                upper='{+' + str(0.01) + '}',
                lower='{-' + str(0.01) + '}')
print(string)

fig = plt.figure(figsize=(3,1))
fig.text(0.1,0.5,string,size=24,va='center')
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('issue5076.pdf')
pp.savefig(fig)
pp.close()