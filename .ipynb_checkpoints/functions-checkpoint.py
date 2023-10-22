import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from typing import List
from obspy.geodetics.base import gps2dist_azimuth

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

def normalize_trace(trace):
    return trace/np.max(np.abs(trace))
    

def handle_polarity(lat1,lon1,lat2,lon2):
    # az_mecha = 178 #azimuth du mechanisme   assime que mechanisme est pile NS  -> suffit de vérifier si la lon du receiver est plus petite ou plus grande que celle de source
    if lon2 >= lon1 :  #lon1 = lon de la source 
        return 1 #si station à droite
    else:
        return -1
    

def shift(obs, n_shift):   
    return np.roll(trace,-n_shift) #doit shift de moins n pour que ça aille vers la gauche   
    