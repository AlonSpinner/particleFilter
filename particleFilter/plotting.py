import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from particleFilter.geometry import pose2

def plot_cov_ellipse(pos, cov, nstd=1, ax=None, facecolor = 'none',edgecolor = 'b' ,  **kwargs):
        #slightly edited from https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
        '''
        Plots an `nstd` sigma error ellipse based on the specified covariance
        matrix (`cov`). Additional keyword arguments are passed on to the 
        ellipse patch artist.

        Parameters
        ----------
            pos : The location of the center of the ellipse. Expects a 2-element
                sequence of [x0, y0].
            cov : The 2x2 covariance matrix to base the ellipse on
            nstd : The radius of the ellipse in numbers of standard deviations.
            ax : The axis that the ellipse will be plotted on. If not provided, we won't plot.
            Additional keyword arguments are pass on to the ellipse patch.

        Returns
        -------
            A matplotlib ellipse artist
        '''
        eigs, vecs = np.linalg.eig(cov)
        theta = np.degrees(np.arctan2(vecs[1,0],vecs[0,0])) #obtain theta from first axis. second axis is just perpendicular to it

        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(eigs)
        ellip = Ellipse(xy=pos, 
                        width=width, 
                        height=height, 
                        angle=theta,
                        facecolor = facecolor, 
                        edgecolor=edgecolor, **kwargs)

        if ax is not None:
            ax.add_patch(ellip)
        
        return ellip

def spawnWorld(xrange = None, yrange = None):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x'); ax.set_ylabel('y'); 
    ax.set_aspect('equal'); ax.grid()

    if xrange is not None: ax.set_xlim(xrange)
    if yrange is not None: ax.set_ylim(yrange)

    return fig, ax

def plot_pose2(ax, x : list[pose2],  color = 'k'):
    locals = np.array([xi.local() for xi in x])
    u = np.cos(locals[:,2])
    v = np.sin(locals[:,2])
    return ax.quiver(locals[:,0],locals[:,1],u,v, color = color)
