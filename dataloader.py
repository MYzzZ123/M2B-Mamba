import numpy as np

def get_networks():
    fc = np.load('data/fc.npy')
    sc = np.load('data/sc.npy')
    y = np.load('data/label.npy')

    y = np.where(y == 1, 0, np.where(y == 2, 1, y))

    return fc, sc, y