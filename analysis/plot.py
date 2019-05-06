import numpy as np



def plot_data(data, ax):
    D = data['R'] == 1
    I = data['R'] == 0
    
    if np.sum(D) > 0:
        ax.scatter(x=data['DB'][D],
                   y=data['A'][D]/data['B'][D],
                   c='k',
                   edgecolors='k',
                   label='chose delayed prospect')
    if np.sum(I) > 0:    
        ax.scatter(x=data['DB'][I],
                   y=data['A'][I]/data['B'][I],
                   c='w',
                   edgecolors='k',
                   label='chose immediate prospect')

    ax.legend()
