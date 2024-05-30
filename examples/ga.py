#!/usr/bin/env python 

import numpy as np
#from geneticalgorithm8 import geneticalgorithm as ga
import os, sys, shutil, time
import pickle
import math
import matplotlib.pyplot as plt
from gamkm.ga_mkmv12_4 import GA_MKMV, read_pckl
import pprint

#run_type
# ga - run genetic algorithm
# single - run single calculation, reading the variables from vars.py (and other things from mkm.py) in a
#          GA calculation in the directory above
# strict_param_single - same as single, but with stricter convergence criteria in the MKM. Generally necessary
#                       for converged rates
# point - run MKM with DFT values. Reads mkm.py from directory above

run_type = 'ga' #'ga','point','single','strict_param_single'

rest_or_total = {'C':'total','O':'rest'} #'total','rest','neither'

aai = True #Adsorbate-adsorbate interaction
average_cross_interaction = True

variable_BE = True #variable XPS binding energies
variable_sigma = True #variable sigmas, i.e. broadening of gaussian XPS peaks

if run_type in ('point','single','strict_param_single'):
    mkm_file = '../mkm.py'
elif run_type == 'ga':
    mkm_file = 'mkm.py'
elif run_type == 'read':
    mkm_file = None
else:
    raise Exception('Invalid run_type')
    
if 'single' in run_type:
    source_energy_slope_file = '../vars.py'
elif run_type in ('read','point'):
    source_energy_slope_file = None
else:
    source_energy_slope_file = 'vars.py'
    
use_strict_params = 'strict_param' in run_type

#If manual starting population is True, I have read a pickle file with the best individuals from 50 runs
#and used them to initiate a new population

manual_start_population = True
if manual_start_population:
    try:
        with open('elite_pop.pckl','rb') as f:
            start_pop = pickle.load(f)
        print('elite_pop read')
    except:
        start_pop = None

#----------sys arguments-------------
if len(sys.argv) == 1:
	cores = 1
elif len(sys.argv) == 2:
	cores = sys.argv[1] #number of cores to parallelize over
else:
	raise Exception('Too many arguments')

#----------------------------------------------------------------------------

if run_type != 'ga' and os.path.isfile('gm.pckl'):
    gm = read_pckl()
    print('gm.pckl read. To rerun calculation, delete file.')
else:
    gm = GA_MKMV(aai=aai,variable_BE=variable_BE,variable_sigma=variable_sigma,
                 mkm_file=mkm_file,
                 average_cross_interaction=average_cross_interaction,
                 rest_or_total_covs=rest_or_total,
                 exp_type='spectra',
                 use_strict_params=use_strict_params)

    if run_type == 'ga':
        gm.set_variable_boundaries(self_slope=1.0,cross_slope=1.0)
        gm.run_GA(cores=cores,manual_start_population=start_pop)

    elif run_type == 'point':
        var_slopes = {}
        gm.run_MK_and_calculate_fitness(energies=None,var_slopes=var_slopes,final_calculation=True)
    elif 'single' in run_type:
        gm.average_cross_interaction = True
        gm.single_calculation(var_file=source_energy_slope_file)
    gm.finish()

if run_type != 'point':
    gm.plot_bar()
    gm.plot_bar(plotted_species='minima',filename='minima.pdf')
    gm.plot_bar(plotted_species='ts',activation_energy=True,filename='act.pdf')
#gm.plot_total_coverage()
gm.plot_total_coverage(file_prefix='stacked_cov',stacked_bars=True,specific_species=[])
gm.plot_spectra(write_species=False)
gm.plot_rxns()
#gm.write_coverages()
#gm.write_coverages(filename='c_covs.txt',exclude_species=['OH_s','H_h'])
gm.write_rates()
gm.write_aai_slopes(ts_coads=['CO_s','COH_s','CH2_s','CH3_s','OH_s','H_h'])
#The intention of choice of co-adsorbates above was to pick the ones with significant coverage. This could be
#automatized, of course
