#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def read_spectra(filename):
    f = open(filename)
    data = []
    for line in f:
        data.append(float(line))
    f.close()
    return data
    
binding_energies = read_spectra('x_axis.txt')
spectra = {}
for T in (175,200,225,250,275,300,325):
    spectra[T] = read_spectra('C1s_111_' + str(T) + 'C.txt')
    print(T,np.trapz(spectra[T],binding_energies)/-17.63*0.96)
    plt.plot(binding_energies,spectra[T])
    
plt.show()
