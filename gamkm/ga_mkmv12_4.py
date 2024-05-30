#!/usr/bin/env python

#from catmap import ReactionModel,analyze
import numpy as np
import shutil,os,sys
import pickle
import time
import math
import copy
import pprint
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ase.data import chemical_symbols
from chemparse import parse_formula
from .colors import *
from .microkinetics7_3 import MicroKinetic
from .geneticalgorithm8_5 import geneticalgorithm as ga

def read_pckl(filename='gm.pckl'):
    try:
        fd = open(filename,'rb')
        content = pickle.load(fd)
        print(filename,'read. Delete to rerun calculation.')
    except Exception as e:
        print(filename,'could not be read.')
        print(e)
        content = None
    return content


class GA_MKMV(object):
    def __init__(self,mkm_file,exp_type='peaks',aai=False,average_cross_interaction=True,
                 variable_BE=False,variable_sigma=False,rest_or_total_covs=None,use_strict_params=False):
        #mkm_file - <str>, .py file containing data. It must be written with python syntax, where every variable
        #           is a key of a 'master' dictionary.
        #aai - <bool>, adsorbate-adsorbate interaction turned on or off
        #average_cross_interaction - <bool>, whether the linear AAI coefficients for species A and B is
        #                            calculated as an average of self-interaction (True) or is given as
        #                            an independent parameter (False)
        #rest_or_total_covs - <dict>. The keys are chemical elements corresponding to coverages, and the values
        #                     can be 'total','rest', 'neither' or 'only_specified_ads_groups'.
        #                     See the function 'resolve_rest_tot' for details.
        #exp_type - <str>, 'peaks' or 'spectra'. 'peaks' assumes that the experimental coverage is given for
        #           groups of adsorbates for each element, while 'spectra' uses the direct spectra. In the 
        #           latter case, computed coverages are used to simulate the spectra by assuming broadening.
        #use_strict_params - <bool>. Use long integration time length and low error tols in the ODE if
        #                    defined. It takes stricter parameters to converge rates than coverages, so if
        #                    the GA only needs coverages for the fitness, it can use loose parameters,
        #                    while stricter is needed to converge the rates for a certain GA solution.
        self.init() #create the file 'incomplete'
        self.start_time = time.time()
        self.slope_bounds = {}
        if os.path.exists(mkm_file):
            print('Mikrokinetic parameter file:',mkm_file)
            self.param = self.read_dict_file_exec(mkm_file)
        else:
            raise Exception('No mkm file found. Maybe run_type is incorrect?')
        if use_strict_params:
            if 'strict_ode_tols' in self.param:
                self.ode_tols = self.param['strict_ode_tols']
            else:
                print('Warning: Strict ODE tols not found. Normal ODE tols used')
                self.ode_tols = self.param.get('ode_tols',(1e-6,1e-9))
            if 'long_integration_time_length' in self.param:
                self.integration_time_length = self.param['long_integration_time_length']
            else:
                print('Warning: Long integration time length not found. Normal time length used')
                self.integration_time_length = self.param.get('integration_time_length',1e8)
        else:
            self.ode_tols = self.param.get('ode_tols',(1e-6,1e-9))
            self.integration_time_length = self.param.get('integration_time_length',1e8)
            
        self.aai = aai
        self.average_cross_interaction = average_cross_interaction
        self.variable_BE = variable_BE
        self.variable_sigma = variable_sigma
        self.read_dft_energies()
        self.Tps = []
        self.Ts = []
        self.ps = []
        self.exp_type = exp_type
        if self.exp_type == 'peaks':
            self.read_exp_covs() #Sets Tps
            self.resolve_rest_tot(rest_or_total_covs)
        elif exp_type == 'spectra':
            self.read_exp_spectra() #Sets Tps
        elif exp_type != None:
            raise Exception('Unrecognised exp_type')
        self.default_coverages = {}
        for ads in self.adsorbates:
            self.default_coverages[ads] = 0.

        #Default plot parameters
        rxn_plot_params = {'width': 0.5,
                           'energy_type': 'G',
                           'unify': True,
                           'labels': {'DFT': 'DFT',
                                      'no_AAI': 'GA w/o AAI',
                                      'AAI': 'GA w/ AAI'},
                           'Tp': 'extremes'} #'extremes' means combinations of maxT,minT,maxp,minp. 'all' possible
        rxn_plot_params['energy_type'] = self.param.get('landscape_energy_type',rxn_plot_params['energy_type'])
        rxn_plot_params.update(self.param.get('rxn_plot_params',{}))

        if rxn_plot_params['Tp'] == 'all':
            rxn_plot_params['Tp'] = self.Tps.copy()
        elif rxn_plot_params['Tp'] == 'extremes':
            Tps = []
            for Tp in [(min(self.Ts),min(self.ps)),(min(self.Ts),max(self.ps)),
                       (max(self.Ts),min(self.ps)),(max(self.Ts),max(self.ps))]:
                if Tp in self.Tps and Tp not in Tps:
                    Tps.append(Tp)
            rxn_plot_params['Tp'] = tuple(Tps)
        elif not isinstance(rxn_plot_params['Tp'],(tuple,list)):
            raise Exception("Plotted Tps must be tuple, list, 'all', or 'extremes'")
        self.rxn_plot_params = rxn_plot_params

        return
    
    def init(self,running_file='incomplete',complete_file='completed'):
        if os.path.isfile(complete_file):
    	    os.remove(complete_file)
        f = open(running_file,'w')
        f.close()
        return

    def parse_elements(self,formula):
        comp = parse_formula(formula)
        keys = list(comp.keys())
        for key in keys:
            if key not in chemical_symbols:
                del comp[key]
        return comp   

    def read_exp_spectra(self,conv_from_celsius=False):
        #Assume self.param['spectra'] including key 'species_BE'
        #spectra files, xaxis file, total coverages, species binding energies
        #Let's have everything in the same folder.
        #Spectra files on the form "C_273.15_15000.txt"
        exp_dir = self.param['spectra_dir']
        if exp_dir[-1] != '/':
            exp_dir += '/'
        master = self.read_dict_file_exec(exp_dir + 'exp_param.py')
        self.exp_data = master['exp_data']
        BE = master['BE']      #XPS binding energies. <dict> with element as key and <dict> as keys.
                               #Subdict has adsorbates as keys and BE (<float> or <list> of <float>) as values.
                               
        new_BE = {}
        
        self.BE_axis = {}
        self.element_weighted_ads = {}
        for element in self.exp_data:
            #Spectrum X axis
            axis_path = exp_dir + element + '_x_axis.txt'
            if os.path.exists(axis_path):
                #Read x axis
                f = open(axis_path)
                axis = []
                for line in f:
                    axis.append(float(line))
                f.close()
                self.BE_axis[element] = np.array(axis)
            else:
                self.BE_axis[element] = None
                
            BEs_needed = False
            for Tp in self.exp_data[element]:
                #Tp should be tuple (Temperature_in_Kelvin,pressure_in_Pa)
                self.add_Tps(Tp,conv_from_celsius)
                if 'cov' not in self.exp_data[element][Tp]:
                    raise Exception('Total overage lacking for ' + element + ', ' + str(Tp))
                if ('spectrum_weight' in self.exp_data[element][Tp]) and \
                    (self.exp_data[element][Tp]['spectrum_weight'] > 0.): #Changed line
                    BEs_needed = True
                    if self.BE_axis[element] is None: #not os.path.exists(axis_path):
                        raise Exception("BE axis file '" + axis_path + "' lacking")
                    spectrum_path = exp_dir + element + '_' + str(Tp[0]) + '_' + str(Tp[1]) + '.txt'
                    if not os.path.exists(spectrum_path):
                        raise Exception("Spectrum file '" + spectrum_path + "' missing")
                    f = open(spectrum_path)
                    spectrum = []
                    for line in f:
                        spectrum.append(float(line))
                    f.close()
                    norm = -np.trapz(spectrum,self.BE_axis[element])
                    spectrum = np.array(spectrum) / norm * self.exp_data[element][Tp]['cov']
                    self.exp_data[element][Tp]['spectrum'] = spectrum  
                    
                else:
                    self.exp_data[element][Tp]['spectrum_weight'] = 0.
                    
                                 
            #Weighted ads list
            #Adsorbates with several atoms of 'element' should appear multiple times, e.g.
            #['CH2OH','C2H2','C2H2',...]
            if BEs_needed:
                new_BE[element] = {}
            ads_list = []
            for ads in self.adsorbates:
                comp = self.parse_elements(ads)
                if element in comp: #ads:
                    for i in range(int(comp[element])):
                        ads_list.append(ads)
                    if BEs_needed:
                        if ads not in BE[element]:
                            raise Exception('XPS binding energy for ' + element + ', ads ' + ads + ' missing')
                        if isinstance(BE[element][ads],float):
                            #Better as a list. Could be several element atoms.
                            new_BE[element][ads] = [BE[element][ads]]
                            #self.BE[element][ads] = [self.BE[element][ads]]
                        elif isinstance(BE[element][ads],list):
                            if len(BE[element][ads]) != comp[element]:
                                raise Exception('# of BE does not match # ' + element + ' in adsorbate ' + ads)
                            new_BE[element][ads] = BE[element][ads]
                        else:
                            raise Exception('BE must be float or list')
            self.element_weighted_ads[element] = tuple(ads_list)
        
        #self.sigma = master['sigma'] #Assuming gaussian broadening of XPS peaks, 'sigma' is std deviation
        sigma = master['sigma']
        new_sigma = {}
        if isinstance(sigma,float):
            sigma = {'default': sigma}
        elif not isinstance(sigma,dict):
            raise Exception("sigma not dict nor float")
        
        for element in new_BE:
            #new_sigma[element] = {}
            if element not in sigma:
                if 'default' not in sigma:
                    raise Exception("'default' value needed in sigma")
                #new_sigma[element]['default'] = sigma['default']
                sigma[element] = {'default': sigma['default']} #Might crash
            elif isinstance(sigma[element],float):
                #new_sigma[element]['default'] = sigma[element]
                sigma[element] = {'default': sigma[element]}
            elif not isinstance(sigma[element],dict):
                raise Exception("sigma['" + element + "'] must be float or dict")

            for ads in new_BE[element]:
                if not isinstance(new_BE[element][ads],list):
                    raise Exception('BEs should have been parsed to lists')
                nb_peaks = len(new_BE[element][ads])

                if ads not in sigma[element]:
                    if 'default' not in sigma[element]:
                        raise Exception("'default' value needed in sigma['" + element + "']")
                    sigma[element][ads] = {'default': [sigma[element]['default']] * nb_peaks} #Might crash
                elif isinstance(sigma[element][ads],float):
                    sigma[element][ads] = {'default': [sigma[element][ads]] * nb_peaks}
                elif isinstance(sigma[element][ads],list):
                    if len(sigma[element][ads]) != nb_peaks:
                        raise Exception("Wrong number of peaks for sigma['" + element + "']['" + ads + "']")
                    sigma[element][ads] = {'default': sigma[element][ads]}
                elif not isinstance(sigma[element][ads],dict):
                    raise Exception("sigma['" + element + "']['" + ads + "'] must be float, list or dict")
                
                if 'default' in sigma[element][ads]:
                    if isinstance(sigma[element][ads]['default'],float):
                        sigma[element][ads]['default'] = [sigma[element][ads]['default']] * nb_peaks
                    elif isinstance(sigma[element][ads]['default'],list):
                        if len(sigma[element][ads]['default']) != nb_peaks:
                            raise Exception("Wrong number of peaks for sigma['" + element + "']" + \
                                            "['" + ads + "']['default']")
                    else:
                        raise Exception("sigma['" + element + "']['" + ads + "']['default'] " + \
                                        "must be list or float")
                    
                for Tp in self.exp_data[element]:
                    if Tp not in sigma[element][ads]:
                        if 'default' not in sigma[element][ads]:
                            raise Exception("'default' value needed in sigma['" + element + "']['" + ads + "']")
                        sigma[element][ads][Tp] = sigma[element][ads]['default']
                    elif isinstance(sigma[element][ads][Tp],float):
                        sigma[element][ads][Tp] = [sigma[element][ads][Tp]] * nb_peaks
                    elif isinstance(sigma[element][ads][Tp],list):
                        if len(sigma[element][ads][Tp]) != nb_peaks:
                            raise Exception("Wrong number of peaks for sigma['" + element + "']['" + ads + \
                                            "']['" + str(Tp) + "'] ")
                    else:
                        raise Exception("sigma['" + element + "']['" + ads + "']['" + str(Tp) + \
                                  "'] must be list or float")
                        
        
        self.BE = new_BE
        self.sigma = sigma
        return
        
    def add_Tps(self,Tp,conv_from_celsius=False):
        if conv_from_celsius:
            Tp[0] += 273.15
        if Tp not in self.Tps:
            self.Tps.append(Tp)
        (T,p) = Tp
        if T not in self.Ts:
            self.Ts.append(T)
        if p not in self.ps:
            self.ps.append(p)
        
    def read_exp_covs(self,conv_from_celsius=False):
        conv_from_celsius = self.param.get('conv_from_celsius',conv_from_celsius)
        master = self.read_dict_file_exec(self.param['exp_cov_file'])
        self.exp_covs = master['exp_covs']

        for element in self.exp_covs:
            for ads_group in list(self.exp_covs[element]):
                if ads_group not in ('Tot','Rest'):
                    if isinstance(ads_group,str):
                        ads_tuple = tuple([ads_group])
                        self.exp_covs[element][ads_tuple] = self.exp_covs[element][ads_group]
                        del self.exp_covs[element][ads_group]
                        ads_group = ads_tuple
                    elif not isinstance(ads_group,tuple):
                        raise Exception("Ads groups must be tuples, 'Tot', or 'Rest'")
                    for ads in ads_group:
                        if ads not in self.adsorbates:
                            raise Exception('Adsorbate ' + ads + ' exists for exp but not DFT.')
                for Tp in self.exp_covs[element][ads_group]:
                    self.add_Tps(Tp,conv_from_celsius)
                        #not perfect, but probably
                        #error will be caught elsewhere
                    if isinstance(self.exp_covs[element][ads_group][Tp]['cov'],list):
                        #Several samples of coverages. The factor 1.5 for marg is taken ad hoc
                        #without any deeper statistical analysis
                        cov = np.mean(self.exp_covs[element][ads_group][Tp]['cov'])
                        marg = (max(self.exp_covs[element][ads_group][Tp]['cov']) - \
                                min(self.exp_covs[element][ads_group][Tp]['cov'])) * 0.5 * 1.5
                        self.exp_covs[element][ads_group][Tp]['cov'] = cov
                        self.exp_covs[element][ads_group][Tp]['marg'] = marg
        #'Manual' corrections in mkm.py.
        if 'exp_covs' in self.param:
            for element in self.param['exp_covs']:
                if element not in self.exp_covs:
                    self.exp_covs[element] = {}
                for ads_group in self.param['exp_covs'][element]:
                    if ads_group not in self.exp_covs[element]:
                        self.exp_covs[element][ads_group] = {}
                    for Tp in self.param['exp_covs'][element][ads_group]:
                        if Tp not in self.exp_covs[element][ads_group]:
                            self.exp_covs[element][ads_group][Tp] = {}
                        for q in ('cov','marg','weight'):
                            self.exp_covs[element][ads_group][Tp][q] = \
                            self.param['exp_covs'][element][ads_group][Tp][q]
        else:
            self.exp_covs = None
        return 
      
    def read_dft_energies(self):
        #The DFT energy file should contain the dict 'master['species']'
        #The keys of this dict is chemical species, and the value is a dict with keys
        # 'energy' - <float> Formation energy relative to gas phase 'reference states' in mkm_file)
        # 'freqs' - <list> of <floats>/<ints> Vibration frequencies in inv cm. No imaginary numbers
        #for gases also
        # 'Atoms' - <ase.atoms> Easiest is ase.build.molecule
        # 'geometry' - <str> 'linear','nonlinear'
        # 'symmetry' - <int> Symmetry number, e.g. 12 for methane and 2 for water
        master = self.read_dict_file_exec(self.param['dft_energy_file'])
        species_data = master['species']
        self.adsorbates = []
        self.transition_states = []
        self.gases = []
        self.vibs_etc = {}
        self.dft_energies = {}
        for species in species_data:
            if '_' not in species:
                print("Warning! It is strongly recommended to have '_' + a site marker or gas in input files")
            self.vibs_etc[species] = {}
            if species[-2:] == '_g':
                self.gases.append(species)
            elif '-' in species:
                self.transition_states.append(species)
            else:
                self.adsorbates.append(species)
            for variable in species_data[species]:
                if variable == 'energy':
                    self.dft_energies[species] = species_data[species][variable]
                elif variable == 'freqs':
                    freqs = species_data[species]['freqs']
                    nb_atoms = sum(self.parse_elements(species).values())
                    if not all([isinstance(freq,(int,float)) for freq in freqs]):
                        raise Exception('Freqencies not all real for ' + species)
                    if species in self.adsorbates and len(freqs) != 3 * nb_atoms:
                        print('Number of frequencies for ' + species + ' is not ' + str(3*nb_atoms))
                    elif species in self.transition_states and len(freqs) != 3 * nb_atoms - 1:
                        print('Number of frequencies for ' + species + ' is not ' + str(3*nb_atoms-1))
                    elif species in self.gases and (len(freqs) > 3 * nb_atoms or len(freqs) == 0):
                        print('Number of frequencies for ' + species + ' is incorrect')
                    invcm_in_eV = 1.23981e-4
                    self.vibs_etc[species]['freqs'] = [freq * invcm_in_eV for freq in freqs]
                else:
                    self.vibs_etc[species][variable] = species_data[species][variable]
        
        #Products and reactants:
        self.reactants = self.param.get('reactants')
        self.products = self.param.get('products')
        if self.reactants == None and self.products == None:
            #Guess from fractional_pressures
            self.reactants = list(self.param['fractional_pressures'].keys())
            self.products = [gas for gas in self.gases if gas not in self.reactants]
            print('Reactants:',self.reactants,'(override possible in mkm file)')
            print('Products:',self.products,'(override possible in mkm file)')
        elif self.products == None:
            self.products = [gas for gas in self.gases if gas not in self.reactants]
            print('Products:',self.products,'(override possible in mkm file)')
        elif self.reactants == None:
            self.reactants = [gas for gas in self.gases if gas not in self.products]
            print('Reactants:',self.reactants,'(override possible in mkm file)')
        
        #Overriding DFT energies
        if 'overriding_energies' in self.param:
            for species in self.param['overriding_energies']:
                if species in self.dft_energies:
                    self.dft_energies[species] = self.param['overriding_energies'][species]
                    print('DFT energy overriden for ' + species)
                else:
                    raise Exception(species + ' not in DFT energies')
        self.name_list = self.adsorbates + self.transition_states + self.gases
        #Deviation coefficients for each species
        coefs = {}
        if 'dft_deviation_coefs' in self.param:
            if isinstance(self.param['dft_deviation_coefs'],dict):
                for species in self.dft_energies:
                    if species in self.param['dft_deviation_coefs']:
                        coefs[species] = self.param['dft_deviation_coefs'][species]
                    else:
                        coefs[species] = self.param['dft_deviation_coefs']['default']
            elif isinstance(self.param['dft_deviation_coefs'],(float,int)):
                for species in self.dft_energies:
                    coefs[species] = self.param['dft_deviation_coefs']
        else:
            for species in self.dft_energies:
                coefs[species] = 1.0
                
        self.coefs = coefs
        return
        
    def resolve_rest_tot(self,rest_or_total):
        #The possible values of the rest_or_total dict are:
        # 'total': Use the 'Tot' coverage. Check that coverages sum up.
        # 'rest': Use the 'Rest' coverage. Check that coverages sum up.
        # 'neither': Use neither. There must be no rest species. Check that coverages sum up.
        # 'only_specified_ads_groups'. Ignore any 'Tot' or 'Rest'. No check that coverages add up.
        #Both Total and Rest should be defined for each element
        #Unless Rest == Tot or Rest is None
        
        if type(rest_or_total) != dict:
            raise Exception("rest_or_total must be a dict with elements as keys")
        
        def coverage_sums_up(element,tol=1e-3,raise_error=True):
            Tps = list(self.exp_covs[element]['Tot'].keys()) #Temperatures and pressures
            tmp = {}
            for Tp in Tps:
                tmp[Tp] = 0
            for ads_group in self.exp_covs[element]:
                if ads_group != 'Tot':
                    for Tp in self.exp_covs[element][ads_group]:
                        tmp[Tp] += self.exp_covs[element][ads_group][Tp]['cov']
            equal = True
            for Tp in Tps:
                if abs(tmp[Tp] - self.exp_covs[element]['Tot'][Tp]['cov']) >= tol:
                    equal = False
            if not equal and raise_error:
                for Tp in Tps:
                    print('All:',tmp[Tp],'Tot',self.exp_covs[element]['Tot'][Tp]['cov'])
                raise Exception('Coverage inequality')
            return equal
        
        rest_tot_dict = {}
        
        for element in self.exp_covs:
            rest_tot_dict[element] = {}
            if rest_or_total[element] not in ('rest','total','neither','only_specified_ads_groups'):
                raise Exception("'rest_or_total' must be 'rest', 'total', 'neither' or 'only_specified_ads_groups'")
            if rest_or_total[element] == 'only_specified_ads_groups':
                ads_groups = list(self.exp_covs[element])
                for ads_group in ads_groups:
                    if ads_group in ('Rest','Tot'):
                        del self.exp_covs[element][ads_group]
                continue
                    
            total_adsorbates = []
            rest_adsorbates = []
            for ads in self.adsorbates:
                composition = self.parse_elements(ads)
                if element in composition:
                    total_adsorbates.append(ads)
                    rest_adsorbates.append(ads)
            if len(total_adsorbates) == 0:
                raise Exception('tot resolution error. Empty list.')
            
            for ads_group in self.exp_covs[element]:
                if ads_group not in ('Rest','Tot'):
                    for ads in ads_group:
                        rest_adsorbates.remove(ads)
            
            if rest_adsorbates == total_adsorbates:
                #There are no specified ads groups
                if rest_or_total[element] == 'total' and 'Tot' in self.exp_covs[element]:
                    self.exp_covs[element][tuple(total_adsorbates)] = self.exp_covs[element]['Tot']
                    del self.exp_covs[element]['Tot']
                    rest_tot_dict[element][tuple(total_adsorbates)] = 'Tot'
                else:
                    raise Exception("No specified ads groups -> Must have only 'Tot'")
                    
            elif len(rest_adsorbates) == 0:
                #The specified ads groups covers all
                if rest_or_total[element] == 'total':
                    if 'Rest' in self.exp_covs[element]:
                        print("WARNING - Redundant 'Rest' in exp_covs[" + element + "]")
                        del self.exp_covs[element]['Rest']
                    if 'Tot' in self.exp_covs[element]:
                        coverage_sums_up(element,raise_error=True)
                        self.exp_covs[element][tuple(total_adsorbates)] = self.exp_covs[element]['Tot']
                        del self.exp_covs[element]['Tot']
                        rest_tot_dict[element][tuple(total_adsorbates)] = 'Tot'
                    else:
                        raise Exception("'rest_or_total = 'total' -> 'Tot' must exist")
                elif rest_or_total[element] == 'neither':
                    del self.exp_covs[element]['Tot']
                    del self.exp_covs[element]['Rest']
                    continue
                else:
                    raise Exception("'rest_or_total' cannot be 'rest' if there are no rest adsorbates")
            else:
                if not ('Rest' in self.exp_covs[element] and 'Tot' in self.exp_covs[element]):
                    raise Exception("Rest adsorbates neither all or None -> Both 'Rest' and 'Tot' must be defined")
                if rest_or_total[element] == 'neither':
                    print('WARNING - all adsorbates are not accounted for in exp_covs[' + element + ']')
                    del self.exp_covs[element]['Rest']
                    del self.exp_covs[element]['Tot'] 
                    continue
                elif rest_or_total[element] == 'rest':
                    coverage_sums_up(element,raise_error=True)
                    self.exp_covs[element][tuple(rest_adsorbates)] = self.exp_covs[element]['Rest']
                    del self.exp_covs[element]['Rest']
                    del self.exp_covs[element]['Tot']
                    rest_tot_dict[element][tuple(total_adsorbates)] = 'Rest'            
                elif rest_or_total[element] == 'total':
                    coverage_sums_up(element,raise_error=True)
                    self.exp_covs[element][tuple(total_adsorbates)] = self.exp_covs[element]['Tot']
                    del self.exp_covs[element]['Rest']
                    del self.exp_covs[element]['Tot']
                    rest_tot_dict[element][tuple(total_adsorbates)] = 'Tot'               
                else:
                    raise Exception('Error in resolve_rest_tot')
            
            ads_groups = list(self.exp_covs[element].keys())
            for ads_group in ads_groups:
                if ads_group not in rest_tot_dict:
                    zero_weight = True
                    for Tp in self.exp_covs[element][ads_group]:
                        if self.exp_covs[element][ads_group][Tp]['weight'] > 0:
                            zero_weight = False
                    if zero_weight:
                        del self.exp_covs[element][ads_group]
        self.rest_tot_dict = rest_tot_dict
        return
        
    def run_GA(self,cores,manual_start_population=None):
        timeouts = self.param.get('function_timeout',{'aai':18.,'no_aai':12.})
        if self.aai:
            function_timeout = timeouts['aai']
        else:
            function_timeout = timeouts['no_aai']
        
        var_bounds,var_frame = self.set_variable_boundaries()
        
        #self.var_bounds = var_bounds
        self.var_frame = var_frame
        
        #var_bounds to list with help of var_frame
        var_bound_list = []
        def subdivide(frame_dict,actual_dict,new_list):
            for key in frame_dict:
                if isinstance(frame_dict[key],(float,int)):
                    new_list.append(actual_dict[key])
                elif isinstance(frame_dict[key],list):
                    new_list.extend(actual_dict[key])
                elif isinstance(frame_dict[key],dict):
                    new_list = subdivide(frame_dict[key],actual_dict[key],new_list)
                else:
                    raise Exception('Type error in var_bounds')
            return new_list
            
        var_bound_list = []
        subdivide(self.var_frame,var_bounds,var_bound_list)
                    
        self.var_bound_array = np.array(var_bound_list)
               
        ga_model = ga(function=self.target_function,
                      dimension=len(self.var_bound_array),
                      variable_type='real',
                      variable_boundaries=self.var_bound_array,
                      algorithm_parameters=self.param.get('algorithm_parameters',{}),
                      parallel_cores=cores,
                      manual_start_population=manual_start_population,
                      read_continuation_population=self.param.get('read_continuation_population',False),
                      logfile=self.param.get('logfile','eval.log'),
                      func_exec_report_file=self.param.get('func_exec_report_file'),
                      save_timelapse=self.param.get('save_timelapse',False),
                      progress_bar=self.param.get('progress_bar',False),
                      cont_file_writing_frequency=self.param.get('cont_file_writing_frequency',100),
                      function_timeout=function_timeout,
                      timeout_objective_function=self.param.get('timeout_objective_function',150.))
        ga_model.run()
        sys.stdout.flush()
        self.ga_fitness = ga_model.output_dict['function']
        ga_output = self.list_to_dict(ga_model.output_dict['variable'],self.var_frame)
        """
        energies = copy.copy(ga_output['energies'])
        
        for state in self.param['reference_states']:
            energies[state] = 0.
        
        if 'slopes' in ga_output:
            var_slopes = ga_output['slopes']
        else:
            var_slopes = {}
        """
        self.write_dict_file('vars.py',ga_output)
        fitness, cov_calc_report = self.run_MK_and_calculate_fitness(ga_output,final_calculation=True)
        #print(cov_calc_report)

        return
        
    def single_calculation(self,var_file,print_slopes=False):
        var_dict = self.read_dict_file_exec(var_file)
        self.set_variable_boundaries()
        
        fitness, cov_calc_report = self.run_MK_and_calculate_fitness(var_dict,final_calculation=True)
        #print(cov_calc_report)
        
        return
        
    def target_function(self,inputs,index): #or fitness function
        var_dict = self.list_to_dict(inputs,self.var_frame)
        
        fitness, cov_calc_report = self.run_MK_and_calculate_fitness(var_dict,final_calculation=False)
        #print('Individual',index,'completed with fitness',round(fitness))
        return fitness, cov_calc_report
   
    def run_MK_and_calculate_fitness(self,variables,final_calculation=False):
        energies = variables.get('energies')
        for state in self.param['reference_states']:
            energies[state] = 0.
        var_slopes = variables.get('slopes')
        var_BE = variables.get('BEs')
        var_sigma = variables.get('sigmas')
        
        if energies == None:
            mk_data = 'DFT'
            energies = copy.copy(self.dft_energies)
        elif var_slopes:
            mk_data = 'AAI'
        else:
            mk_data = 'no_AAI'
        
        slopes = self.parse_slopes(var_slopes)
        
        for species in slopes:
            if species not in self.adsorbates + self.transition_states:
                raise Exception('Non-species in slopes: ' + species)
        """
        species_coverages, rates, failure, cov_calc_report = self.calculate_coverages(energies,slopes,
                                                                                      plot_rxns=final_calculation,
                                                                                      mk_data=mk_data,
                                                                                      verbose=final_calculation)
        """
        species_coverages, reactions, failure, cov_calc_report = self.run_MK(energies,slopes,verbose=final_calculation)
        
        if failure:
            cov_deviation = 100.
            spectra_deviation = 100.
        else:
            
            if self.exp_type == 'peaks':
                delta_covs, calc_covs = self.calc_coverage_deviation(species_coverages)
                cov_deviation = self.cov_deviation_fitness(delta_covs)
                spectra_deviation = 0.
            elif self.exp_type == 'spectra':
                delta_data, calc_data = self.calc_spectra_deviation(species_coverages,var_BE=var_BE,var_sigma=var_sigma)
                spectra_deviation, cov_deviation = self.calc_spectra_fitness(delta_data)
            else:
                raise Exception()
            
        dft_deviation, max_deviation = self.dft_deviation_penalty_fitness(energies)
                                                                      
        aai_penalty = self.aai_penalty_fitness(var_slopes)
        
        fitness = cov_deviation + spectra_deviation + dft_deviation + aai_penalty
        
        if final_calculation:
            self.output = {}
            self.max_deviation = max_deviation
            self.fitness = fitness
            self.fitness_composition = {'exp_cov_deviation':cov_deviation,
                                        'dft_deviation':dft_deviation,
                                        'aai_penalty':aai_penalty}

            self.species_coverages = species_coverages
            self.ga_energies = copy.copy(energies)
            self.reactions = reactions
            self.mk_data = mk_data
            self.var_slopes = var_slopes
            self.BE = var_BE
            
            if self.exp_type == 'peaks':            
                self.calc_covs = calc_covs
                self.delta_covs = delta_covs
            if self.exp_type == 'spectra':
                self.calc_data = calc_data
                self.fitness_composition['exp_spectra_deviation'] = spectra_deviation

            if slopes:
                self.save_corr_details(slopes,species_coverages)
        return fitness, cov_calc_report
        
    def get_production_rates(self,rxns):
        #Calculate rates for gases
        rates = {}
        for gas in self.gases:
            rates[gas] = 0.
        for rxn in rxns:
            gas_stoich = {}
            for species in rxns[rxn]['l']:
                if species in self.gases:
                    if species in gas_stoich:
                        gas_stoich[species] -= 1
                    else:
                        gas_stoich[species] = -1
            for species in rxns[rxn]['r']:
                if species in self.gases:
                    if species in gas_stoich:
                        gas_stoich[species] += 1
                    else:
                        gas_stoich[species] = 1
            for gas in gas_stoich:
                rates[gas] += gas_stoich[gas] * rxns[rxn]['rate']
        return rates
    
    def set_variable_boundaries(self,energy_bounds={'gas':0.3,'transition_state': 0.5, 'adsorbate':0.5},
                                self_slope=1.0,cross_slope=1.0,
                                BE_margin=0.15):
                                
        self.energy_bounds = self.param.get('energy_bounds',energy_bounds)

        var_bounds = {'energies':{}}
        var_frame = {'energies':{}}
        for species in self.name_list:
            if species not in self.param['reference_states']: #CO_g,H2O_g,H2_g
                if species in self.energy_bounds:
                    #var_bounds['energies'][species] = [self.dft_energies[species] - energy_bounds[species],
                    #                                   self.dft_energies[species] + energy_bounds[species]]
                    deltaE = self.energy_bounds[species]
                elif species[-2:] == '_g':
                    #var_bounds['energies'][species] = [self.dft_energies[species] - energy_bounds['gas'],
                    #                                   self.dft_energies[species] + energy_bounds['gas']]
                    deltaE = self.energy_bounds['gas']
                elif '-' in species:
                    #var_bounds['energies'][species] = [self.dft_energies[species] - energy_bounds['transition_state'],
                    #                                   self.dft_energies[species] + energy_bounds['transition_state']]
                    deltaE = self.energy_bounds['transition_state']
                else:
                    #var_bounds['energies'][species] = [self.dft_energies[species] - energy_bounds['adsorbate'],
                    #                                   self.dft_energies[species] + energy_bounds['adsorbate']]
                    deltaE = self.energy_bounds['adsorbate']
                var_bounds['energies'][species] = [self.dft_energies[species] - deltaE,
                                                   self.dft_energies[species] + deltaE]
                var_frame['energies'][species] = self.dft_energies[species] - 2 * abs(deltaE) #Not to be in interval
                

        
        #average_cross_interaction=True means that the self-interaction slopes are defined for
        #adsorbates, while adsorbate cross interaction is averaged.
        #Otherwise, all ads-ads interactions are defined (triangularly, since A&B == B&A
        #Transition states are always specied as cross interaction
        
        #self.adsorbates = ['H_h','CO_s','OH_s']
        #self.transition_states = ['CO-H_s','O-H_s']
        
        if self.variable_BE:
            BE_bounds = {}
            BE_frame = {}
            BE_margin = self.param.get('BE_margin',BE_margin)

            #pprint.pprint(self.BE)
            #quit()
            for element in self.BE:
                BE_bounds[element] = {}
                BE_frame[element] = {}
                for species in self.BE[element]:
                    BE_bounds[element][species] = []
                    BE_frame[element][species] = []
                    for peak in self.BE[element][species]:
                        bound = [peak - BE_margin, peak + BE_margin]
                        BE_bounds[element][species].append(bound)
                        BE_frame[element][species].append(bound[0] - abs(BE_margin))
            var_bounds['BEs'] = BE_bounds
            var_frame['BEs'] = BE_frame
        
        if self.variable_sigma: #Temperature independent
            sigma_bounds = {}
            sigma_frame = {}
            sigma_bound = self.param.get('sigma_bounds',[0.2,0.5])
            for element in self.BE: #BE intentially
                sigma_bounds[element] = {}
                sigma_frame[element] = {}
                for species in self.BE[element]:
                    sigma_bounds[element][species] = []
                    sigma_frame[element][species] = []
                    for peak in self.BE[element][species]:
                        sigma_bounds[element][species].append(sigma_bound)
                        sigma_frame[element][species].append(bound[1])
            var_bounds['sigmas'] = sigma_bounds
            var_frame['sigmas'] = sigma_frame
            
        if self.aai:
            self_slope = self.param.get('self_slope',self_slope)
            cross_slope = self.param.get('cross_slope',cross_slope)
            max_slopes = {}
            slope_bounds = {}
            slope_frame = {}
       
            #self.average_cross_interaction = average_cross_interaction
            if self.average_cross_interaction:
                for ads in self.adsorbates:
                    max_slopes[ads] = self_slope
                    slope_bounds[ads] = [-self_slope,self_slope]
                    slope_frame[ads] = -self_slope - abs(self_slope)
            else:
                for i in range(len(self.adsorbates)):
                    max_slopes[self.adsorbates[i]] = {}
                    slope_bounds[self.adsorbates[i]] = {}
                    slope_frame[self.adsorbates[i]] = {}
                    for j in range(i,len(self.adsorbates)):
                        if i == j:
                            slope = self_slope
                            #max_slopes[self.adsorbates[i]][self.adsorbates[j]] = self_slope
                            #slope_bounds[self.adsorbates[i]][self.adsorbates[j]] = [-self_slope, self_slope]
                        else:
                            slope = cross_slope
                            #max_slopes[self.adsorbates[i]][self.adsorbates[j]] = cross_slope
                            #slope_bounds[self.adsorbates[i]][self.adsorbates[j]] = [-cross_slope, cross_slope]
                        max_slopes[self.adsorbates[i]][self.adsorbates[j]] = slope
                        slope_bounds[self.adsorbates[i]][self.adsorbates[j]] = [-slope, slope]
                        slope_frame[self.adsorbates[i]][self.adsorbates[j]] = -slope - abs(slope)
                        

            for ts in self.transition_states:
                max_slopes[ts] = {}
                slope_bounds[ts] = {}
                slope_frame[ts] = {}
                for co_ads in self.adsorbates:
                    max_slopes[ts][co_ads] = cross_slope
                    slope_bounds[ts][co_ads] = [-cross_slope, cross_slope]
                    slope_frame[ts][co_ads] = -cross_slope - abs(cross_slope)
        
            var_bounds['slopes'] = slope_bounds
            var_frame['slopes'] = slope_frame
        
        return var_bounds,var_frame
    
    def parse_slopes(self,var_slopes,cutoff=0.0):
        #var_dict is a dict with keys 'energies' and 'slopes'. Slopes could be a dict of dicts,
        #but then it must be so to speak 'triangular', so that a combination of surface species
        #
        #slopes below cutoff are set to 0.0 and should give no penalty
        #A and B only appear once
        if var_slopes == None:
            return {}
        slopes = {}
        if self.average_cross_interaction:
            tmp_slopes = {}
            for ads in self.adsorbates:
                if ads in var_slopes and var_slopes[ads] > cutoff:
                    #slopes < cutoff are ignored
                    tmp_slopes[ads] = var_slopes[ads]
                else:
                    tmp_slopes[ads] = 0
            for ads in self.adsorbates:
                slopes[ads] = {}
                for co_ads in self.adsorbates:
                    slope = (tmp_slopes[ads] + tmp_slopes[co_ads]) * 0.5
                    if slope > cutoff: #This should always happen if cutoff >= 0, but not tested
                        slopes[ads][co_ads] = slope
            for ts in self.transition_states:
                if ts in var_slopes:
                    slopes[ts] = {}
                    for co_ads in var_slopes[ts]:
                        if var_slopes[ts][co_ads] > cutoff:
                            slopes[ts][co_ads] = var_slopes[ts][co_ads]
            for species in list(slopes.keys()):
                if len(slopes[species]) == 0:
                    del slopes[species]
        else:
            for species in var_slopes:
                if species not in slopes:
                    slopes[species] = {}
                for co_ads in var_slopes[species]:
                    if var_slopes[species][co_ads] > cutoff:
                        slopes[species][co_ads] = var_slopes[species][co_ads]
                        if '-' not in species:
                            ads = species
                            if co_ads not in slopes:
                                slopes[co_ads] = {}
                            
                            slopes[co_ads][ads] = slopes[ads][co_ads]

        return slopes
 
    def run_MK(self,energies,slopes=None,verbose=False):
        report = ""
        if energies == None:
            raise Exception('Energies are None')
            
        coverages = {}
        reactions = {}
        failure = False
        
        ads_area = self.param.get('ads_area',6.47e-20)
        scale_factor = self.param.get('ode_scale_factor',1.0)
        steps = self.param.get('steps',10000)
        adsorption_model = self.param.get('adsorption_model','HK') #Hertz-Knudsen
        Xrc= self.param.get('Xrc',False)
        ode_solver = self.param.get('ode_solver','super_stiff')
        ode_algorithm = self.param.get('ode_algorithm','Radau')     
        
        guess_init_covs = self.param.get('guess_init_covs',True)
        
        
        for Tp in reversed(self.Tps):
            (T,p) = Tp
            Tp_report = "\n"
            
            #Search already calculated coverages for suitable initial coverages
            if guess_init_covs:
                min_deltaT = np.inf
                min_deltap = np.inf
                closest_i = None
                calc_Tps = list(coverages.keys())
                for i,Tp0 in enumerate(calc_Tps):
                    (T0,p0) = Tp0
                    deltaT = abs(T - T0)
                    deltap = abs(p - p0)
                    if deltaT <= min_deltaT and deltap <= min_deltap:
                        min_deltaT = deltaT
                        min_deltap = deltap
                        closest_i = i
                        
                if closest_i == None or not coverages[calc_Tps[closest_i]]:
                    init_covs = self.default_coverages.copy()
                else:
                    init_covs = coverages[calc_Tps[closest_i]]

                    init_cov_str = "Init covs from " + str(calc_Tps[closest_i])
                    Tp_report += init_cov_str + "\n"
                        #if print_report:
                        #    print(init_cov_str)
            else:
                init_covs = self.default_coverages.copy()
            
            time1 = time.time()
            mk = MicroKinetic(T,p,fractional_pressures=self.param['fractional_pressures'],eps_p=1e-20,
                              ads_Area=ads_area,
                              t=self.integration_time_length,steps=steps,
                              theta0=init_covs,energies=energies, slopes=slopes,
                              vibs_etc=self.vibs_etc,rxns=self.param['rxns'],
                              adsorption_model=adsorption_model,
                              Xrc=Xrc,ode_solver=ode_solver,
                              algorithm=ode_algorithm,tols=self.ode_tols,
                              scale_factor=scale_factor,
                             )
            covs, rxns, species_sites, mkm_report = mk.solve_microkinetic_model(print_report=False,
                                                                                report_runtimes=verbose)
            #if plot_rxns: #Should just be for final calculation
            #    for rxn in rxns:
            #        print(rxn,rxns[rxn]['rate'])
            Tp_report += mkm_report
            tp_runtime_str = "Tp runtime: " + str(round(time.time()-time1,3)) + " s"
            Tp_report += tp_runtime_str + "\n"
            #outputs[mode]['covs'] = covs
            if covs:
                for species in self.param.get('printed_coverages',[]):
                    Tp_report += species + " " + str(round(covs[species],5)) + " ML\n"
            else:
                Tp_report += "MK run failed. No coverages.\n"
            if verbose:
                print(Tp_report)
            report += Tp_report
        
            if not covs:
                failure = True
                coverages[Tp] = {}
                break
            coverages[Tp] = covs
            reactions[Tp] = rxns
        return coverages, reactions, failure, report
        
    def plot_rxns(self):
        if 'plotted_rxns' not in self.param:
            print("No 'plotted_rxns' in mkm file")
            return
        else:
            plotted_rxns = self.param['plotted_rxns']
            
        rxn_rates = {}
        opt_rxns = {}
        rxn_name_len = 0
        for Tp in self.rxn_plot_params['Tp']:
            rxn_rates[Tp] = {}
            for rxn_name in plotted_rxns:
                rxn_name_len = max(rxn_name_len,len(rxn_name))
                rxn_rate = self.get_rxn_rate(plotted_rxns[rxn_name],Tp)
                rxn_rates[Tp][rxn_name] = rxn_rate
                if rxn_rate < 0:
                    print(rxn_name,'at',Tp,'has a negative rate')
                sorted_rates = sorted(rxn_rates[Tp].items(), key=lambda x:x[1])
                opt_rxns[Tp] = sorted_rates[-1][0]
                
        unified = self.rxn_plot_params['energy_type'] == 'E' and self.rxn_plot_params['unify']
        #print('unified',unified)
        
        if unified:
            plot_dir = 'energy_landscapes'
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)
            
            for rxn_name in plotted_rxns:
                self.plot_rxn2(rxn_name,rxn_path=plotted_rxns[rxn_name],Tps=self.rxn_plot_params['Tp'],plot_dir=plot_dir)
                #with open(plot_dir + '/rxn_rates.txt') as fd:
                #    for Tp in self.rxn_plot_params['Tp']:
                #        Tp_str = str(Tp)
                
            if len(set(opt_rxns.values())) == 1: #the same opt rxn at all Tps
                rxn_name = list(opt_rxns.values())[0]
                self.plot_rxn2(rxn_name,rxn_path=plotted_rxns[rxn_name],Tps=self.rxn_plot_params['Tp'],
                               plot_dir=plot_dir,file_prefix='opt_')
            else:
                for Tp in self.rxn_plot_params['Tp']:
                    T,p = Tp
                    self.plot_rxn2(opt_rxns[Tp],rxn_path=plotted_rxns[opt_rxns[Tp]],Tps=self.rxn_plot_params['Tp'],
                                   plot_dir=plot_dir,file_prefix='opt_' + str(T)+   'K'+str(p)+'Pa_')

            with open(plot_dir + '/rxn_rates.txt','w') as fd:
                Tp_list = [str(Tp) for Tp in self.rxn_plot_params['Tp']]
                Tp_lengths = [len(Tp_str) for Tp_str in Tp_list]
                Tp_line = ' '.join(Tp_list)
                fd.write(''.ljust(rxn_name_len + 1) + Tp_line + '\n')
                for rxn_name in plotted_rxns:
                    line = rxn_name.ljust(rxn_name_len + 1)
                    for i,Tp in enumerate(self.rxn_plot_params['Tp']):
                        s = '{:0.3e}'.format(rxn_rates[Tp][rxn_name])
                        line += (s + ' ').rjust(Tp_lengths[i]) + ' '
                    fd.write(line + '\n')
        else:
            for Tp in self.rxn_plot_params['Tp']:
                T,p = Tp
                plot_dir = str(T)+'K'+str(p)+'Pa'
                if os.path.isfile(plot_dir + '/rxn_rates.txt'):
                    os.remove(plot_dir + '/rxn_rates.txt')
                if not os.path.exists(plot_dir):
                    os.mkdir(plot_dir)
                for rxn_name in plotted_rxns:
                    self.plot_rxn2(rxn_name,rxn_path=plotted_rxns[rxn_name],Tps=[Tp],plot_dir=plot_dir)
                    with open(plot_dir + '/rxn_rates.txt','a') as fd:
                        line = rxn_name.ljust(rxn_name_len + 1)
                        line += '{:0.3e}'.format(rxn_rates[Tp][rxn_name]).rjust(10)
                        fd.write(line + '\n')
                self.plot_rxn2(opt_rxns[Tp],rxn_path=plotted_rxns[opt_rxns[Tp]],Tps=[Tp],
                               plot_dir=plot_dir,file_prefix='opt_')

            

        """
        rxn_rates = {}
        for rxn_name in plotted_rxns:
            for Tp in self.rxn_plot_params['Tp']:
                if Tp not in rxn_rates:
                    rxn_rates[Tp] = {}
                rxn_rate = self.get_rxn_rate(plotted_rxns[rxn_name],Tp)
                rxn_rates[Tp][rxn_name] = rxn_rate
                #rxn_rates[rxn_name]
                if rxn_rate < 0:
                    print(rxn_name,'at',Tp,'has a negative rate')
                    
                if not unified:
                    plot_dir = str(T)+'K'+str(p)+'Pa'
                    if not os.path.exists(plot_dir):
                        os.mkdir(plot_dir)
                    
                    self.plot_rxn2(rxn_path=plotted_rxns[rxn_name],Tps=[Tp],plot_dir=plot_dir)
            if unified:
                plot_dir = 'energy_landscapes'
                if not os.path.exists(plot_dir):
                    os.mkdir(plot_dir)
                self.plot_rxn2(rxn_path=plotted_rxns[rxn_name],Tps=self.rxn_plot_params['Tp'],plot_dir=plot_dir)
        opt_rxns = {}
        for Tp in self.rxn_plot_params['Tp']:
            sorted_rates = sorted(rxn_rates[Tp].items(), key=lambda x:x[1])
            opt_rxn = sorted_rates[-1][0]
            opt_rxns[Tp] = opt_rxn
            if not unified:
                (T,p) = Tp
                plot_dir = str(T)+'K'+str(p)+'Pa'
                self.plot_rxn2(rxn_path=plotted_rxns[opt_rxn],Tps=[Tp],plot_dir=plot_dir,prefix="opt_")
        if unified:
            if set(opt_rxns)
        """
            
                
    def plot_rxn2(self,rxn_name,rxn_path,Tps,plot_dir,celsius=True,file_prefix=""):
        Ts = [Tp[0] for Tp in Tps]
        print_T = len(set(Ts)) > 1 #If more than one temperature, write in legend
        ps = [Tp[1] for Tp in Tps]
        print_p = len(set(ps)) > 1 #If more than one pressure, write in legend
        (T,p) = Tps[0]
        energy_type = self.rxn_plot_params['energy_type']
        if len(Tps) > 1 and energy_type != 'E':
            raise Exception('Unifying plots with different Tp only possible with E')

        width = self.rxn_plot_params['width'] #0.5 default
        colors = {'AAI0': Dark2['turquoise'], #'#66c2a5',
                  'AAI1': Dark2['magenta'], #e7298a
                  'AAI2': Dark2['lime_green'], #66a61e
                  'AAI3': Dark2['yellow'], #e6ab02
                  'no_AAI': Dark2['rust'],  #'#d95f02',
                  'DFT': Dark2['lavender'], #'#7570b3',
                 }
        labels = {}
        if len(Tps) > len(colors) - 2:
            raise Exception('Too few colors')
            
        rxnss = {}
        #Plot DFT
        mk_dft = MicroKinetic(T,p,energies=copy.copy(self.dft_energies),slopes=None,
                              vibs_etc=self.vibs_etc,rxns=self.param['rxns'])
        rxnss['DFT'] = mk_dft.get_thermodynamics(print_report=False)[0]
        labels['DFT'] = self.rxn_plot_params['labels']['DFT']
        
        
        #if self.mk_data != DFT: plot no_AAI
        if self.mk_data != 'DFT':
            mk_no_aai = MicroKinetic(T,p,energies=copy.copy(self.ga_energies),slopes=None,
                                     vibs_etc=self.vibs_etc,rxns=self.param['rxns'])
            rxnss['no_AAI'] = mk_no_aai.get_thermodynamics(print_report=False)[0]
            labels['no_AAI'] = self.rxn_plot_params['labels']['no_AAI'] #Could improve depending on AAI or no_AAI
        
        
        #if self.mk_data == AAI: plot AAI(s)
        if self.mk_data == 'AAI':
            for i,Tp in enumerate(Tps):
                rxnss['AAI' + str(i)] = self.reactions[Tp]
                #colors[Tp] = colors['AAI' + str(i)]
                label = self.rxn_plot_params['labels']['AAI']
                if print_T:
                    T = Tp[0]
                    if celsius:
                        T -= 273.15
                        T = int(round(T))
                    if print_p:
                        p = Tp[1]
                        label += str((T,p))
                    else:
                        label += ' (' + str(T) + u'\u00B0C)'
                elif print_p:
                    p = Tp[1]
                    label += ' (' + str(p) + ' Pa)'
                labels['AAI' + str(i)] = label
        
                
        
        #colors['std'] = colors[self.rxn_plot_params['std']]
        
        #modes = sorted(outputs) #std should be plotted last
        #plt.clf()
        plt.figure()

        for mode in rxnss:
            rxns = rxnss[mode]
            color = colors[mode]
            label = labels[mode]
        
            energy_l = 0 #either free energy or energy
            x0 = 1 #rxn coordinate
            plt.plot((x0-0.5*width,x0+0.5*width),(energy_l,energy_l),lw=2.,
                      c=color,label=label)
            ticks = []
            species_labels = []
            for i,rxn in enumerate(rxn_path):           
                x_A = x0+i+0.5*width #connection point left side of rxn and barrier
                x_r = x0+1+i #center for right side of rxn
                x_C = x_r-0.5*width #connection point barrier and right side of rxn
                
                energy_r = energy_l + rxns[rxn]['d' + energy_type]

                ticks.append(x_r-0.5)
                species_labels.append(rxn.replace('to','->'))
                
                Qf = energy_type + 'a_f' #activation energy
                
                if rxns[rxn][Qf] < 0:
                    raise Exception('Negative barrier in rxn plot')
                elif rxns[rxn][Qf] == 0 or rxns[rxn][Qf] <= rxns[rxn]['d' + energy_type]:
                    plt.plot((x_A,x_C),(energy_l,energy_r),'--',lw=1.5,c=color)
                else:
                    energy_B = energy_l + rxns[rxn][Qf] #energy of barrier
                    x_B = (x_C * np.sqrt(energy_B-energy_l) + x_A * np.sqrt(energy_B-energy_r)) / \
                          (np.sqrt(energy_B - energy_l) + np.sqrt(energy_B - energy_r))
                    #x_B Corresponds to maximum of parabola
                    a = -(energy_B - energy_l) / (x_B - x_A)**2
                    b = -2*a*x_B
                    c = energy_B + a*x_B**2
                    xs = np.linspace(x_A,x_C,20,endpoint=True)
                    ys = a * xs**2 + b * xs + c
                    plt.plot(xs,ys,'--',lw=1.5,c=color)           

                plt.plot((x_r-0.5*width,x_r+0.5*width),(energy_r,energy_r),lw=2.,c=color)
                
                energy_l = energy_r
        
        plt.legend()
        if energy_type == 'G':
            plt.ylabel('Free energy (eV)')
        elif energy_type == 'E':
            plt.ylabel('Energy (eV)')
        #plt.title(rxn_name + '(' + str(T) + ' K, ' + str(p) + ' Pa), r=' + "{:.3e}".format(production_rate))
        plt.xticks(ticks,species_labels,rotation=90)
        plt.tight_layout()
        plt.savefig(plot_dir + '/' + file_prefix + rxn_name + '_' + energy_type + '.pdf')
        plt.close()
        return
        
    def calculate_coverages(self,energies=None,slopes=None,plot_rxns=False,mk_data=None,
                            verbose=False):
    
        report = ""
        if energies == None:
            raise Exception('Energies are None')

        coverages = {}
        rates = {}
        failure = False
        
        inputs = {'std':{'energies':energies,'slopes':slopes}}
        if plot_rxns:
            self.rxn_plot_params['labels']['std'] = self.rxn_plot_params['labels'][mk_data] #elegant
            self.rxn_plot_params['std'] = mk_data
            if mk_data != 'DFT':
                inputs['DFT'] = {'energies':self.dft_energies,'slopes':None}
                if mk_data != 'no_AAI':
                    inputs['no_AAI'] = {'energies':energies,'slopes':None}
            #cmp_coverages = {}
            #for mode in inputs:
            #    if mode != 'std':
            #        cmp_coverages[mode] = {}
        
        for Tp in reversed(self.Tps):
            (T,p) = Tp
            
            if plot_rxns and 'plotted_rxns' in self.param and self.param['plotted_rxns']:
                if not 'Tp' in self.rxn_plot_params or Tp in self.rxn_plot_params['Tp']:
                    plot_dir = str(T)+'K'+str(p)+'Pa'
                    self.rxn_plot_params['plot_dir'] = plot_dir
                    plotted_rxns = self.param['plotted_rxns']
                    if not os.path.exists(plot_dir):
                        os.mkdir(plot_dir)
                else:
                    self.rxn_plot_params['plot_dir'] = None
                    plotted_rxns = {}
            else:
                plotted_rxns = {}

            outputs = {}
            
            ads_area = self.param.get('ads_area',6.47e-20)

            scale_factor = self.param.get('ode_scale_factor',1.0)
            
            #init_cov_params = self.param.get('init_covs',{'type':'closest','T_cutoff':})
            guess_init_covs = self.param.get('guess_init_covs',True)
                
            for mode in inputs:
                outputs[mode] = {}
                mode_report = "\n" 
                if len(inputs) > 1:
                    mode_report += mode + "\n"
                #Search already calculated coverages for suitable initial coverages
                #if mode == 'std': #and 'init_covs' in self.param and self.param['init_covs']['type'] == 'closest':
                if mode == 'std' and guess_init_covs:
                    min_deltaT = np.inf
                    min_deltap = np.inf
                    closest_i = None
                    calc_Tps = list(coverages.keys())
                    for i,Tp0 in enumerate(calc_Tps):
                        (T0,p0) = Tp0
                        deltaT = abs(T - T0)
                        deltap = abs(p - p0)
                        if deltaT <= min_deltaT and deltap <= min_deltap:
                            min_deltaT = deltaT
                            min_deltap = deltap
                            closest_i = i
                        
                    if closest_i == None or not coverages[calc_Tps[closest_i]]:
                        init_covs = self.default_coverages.copy()
                    else:
                        init_covs = coverages[calc_Tps[closest_i]]

                        init_cov_str = "Init covs from " + str(calc_Tps[closest_i])
                        mode_report += init_cov_str + "\n"
                        #if print_report:
                        #    print(init_cov_str)
                else:
                    init_covs = self.default_coverages.copy()
                
                steps = self.param.get('steps',10000)
                adsorption_model = self.param.get('adsorption_model','HK') #Hertz-Knudsen
                Xrc= self.param.get('Xrc',False)
                ode_solver = self.param.get('ode_solver','super_stiff')
                ode_algorithm = self.param.get('ode_algorithm','Radau')
                
                time1 = time.time()
                mk = MicroKinetic(T,p,fractional_pressures=self.param['fractional_pressures'],eps_p=1e-20,
                                  ads_Area=ads_area,
                                  t=self.integration_time_length,steps=steps,
                                  theta0=init_covs,energies=inputs[mode]['energies'], slopes=inputs[mode]['slopes'],
                                  vibs_etc=self.vibs_etc,rxns=self.param['rxns'],
                                  adsorption_model=adsorption_model,
                                  Xrc=Xrc,ode_solver=ode_solver,
                                  algorithm=ode_algorithm,tols=self.ode_tols,
                                  scale_factor=scale_factor,
                                 )
                if mode == 'std':
                    covs, rxns, species_sites, mkm_report = mk.solve_microkinetic_model(print_report=False,
                                                                                        report_runtimes=verbose)
                    if plot_rxns: #Should just be for final calculation
                        for rxn in rxns:
                            print(rxn,rxns[rxn]['rate'])
                    mode_report += mkm_report
                    tp_runtime_str = "Tp runtime: " + str(round(time.time()-time1,3)) + " s"
                    mode_report += tp_runtime_str + "\n"
                    outputs[mode]['covs'] = covs
                    if covs:
                        for species in self.param.get('printed_coverages',[]):
                            mode_report += species + " " + str(round(covs[species],5)) + " ML\n"
                    else:
                        mode_report += "MK run failed. No coverages.\n"
                        
                else:
                    rxns, species_sites, thermo_report = mk.get_thermodynamics(print_report=False)
                    #pprint.pprint(rxns)
                    mode_report += "Thermodynamics read\n"
                    mode_report += thermo_report
                    covs = False
                    
                outputs[mode]['rxns'] = rxns
                
                if len(inputs) > 1:
                    mode_report += "-------------------------------------\n"
                if verbose:
                    print(mode_report)
                report += mode_report
            
            if plotted_rxns:
                production_rates = {}
                for rxn_name in plotted_rxns:
                    try:
                        production_rate = self.plot_rxn(rxn_name,outputs,T,p,
                                                        energy_type=self.param.get('landscape_energy_type','G'))
                    except Exception as e:
                        report += "Reaction " + rxn_name + " could not be plotted\n"
                        raise Exception(e)
                    production_rates[rxn_name] = production_rate
     
                sorted_rates = sorted(production_rates.items(), key=lambda x:x[1])
                #print(sorted_rates)
                #print(sorted_rates[-1][0])
                #quit()
                self.plot_rxn(sorted_rates[-1][0],outputs,T,p,
                              energy_type=self.param.get('landscape_energy_type','G'),file_prefix='opt_')
                
                
            if plotted_rxns:
                for rxn in outputs['std']['rxns']:
                    #print(rxn,"{:.3e}".format(outputs['std']['rxns'][rxn]['rate']))
                    report += rxn + "{:.3e}".format(outputs['std']['rxns'][rxn]['rate'])
                    
            if not outputs['std']['covs']:
                failure = True
                coverages[Tp] = {}
                break
            coverages[Tp] = outputs['std']['covs']
            
            if plot_rxns:
                #for mode in inputs:
                    #if mode != 'std':
                    #    cmp_coverages[mode][Tp] = outputs[mode]['covs']
                rates[Tp] = self.get_production_rates(outputs['std']['rxns'])
            else:
                rates = None
        return coverages, rates, failure, report

    def get_rxn_rate(self,rxn_path,Tp):
        rxns = self.reactions[Tp]
        intermediates = {}
        tot_rxn = {}
        deletion_flag = False
        rxn_rate = np.inf
        for rxn in rxn_path:  
            if rxn not in rxns:
                #Check if backward_rxn exists
                if 'to' in rxn:
                    left,right = rxn.split('to')
                    backward_rxn = right + 'to' + left
                elif 'ads' in rxn:
                    backward_rxn = rxn.replace('ads','des')
                elif 'des' in rxn:
                    backward_rxn = rxn.replace('des','ads')
                else:
                    raise Exception('Rxn ' + rxn + ' not in rxns, and no reverse rxn found')
                for mode in outputs.keys():                 
                    frxn = {}
                    brxn = rxns[backward_rxn]
                    
                    for pair in [('Ga0_f','Ga0_rev'),('Ga_f','Ga_rev'),
                                 ('Ea0_f','Ea0_rev'),('Ea_f','Ea_rev'),
                                 ('kr','kf'),('l','r'),('m','m')]:
                        frxn[pair[0]] = brxn[pair[1]]
                        frxn[pair[1]] = brxn[pair[0]]
                    for q in ['dEc','dG','dG0','dE','dE0','rate']:
                        frxn[q] = -brxn[q]
                    rxns[rxn] = frxn
                
            if rxns[rxn]['rate'] < rxn_rate:
                rxn_rate = rxns[rxn]['rate']
            for species in rxns[rxn]['l']:
                if not '*' in species:
                    if species in self.reactants:
                        if not species in tot_rxn:
                            tot_rxn[species] = -1
                        else:
                            tot_rxn[species] -= 1
                    elif species not in intermediates or intermediates[species] == 0:
                        print('Error in rxn_plot: Species',species, 'not on surface')
                    else:
                        intermediates[species] -= 1
                        if intermediates[species] == 0:
                            del intermediates[species]
                            deletion_flag = True
            for species in rxns[rxn]['r']:
                if not '*' in species:
                    if species in self.products:
                        if not species in tot_rxn:
                            tot_rxn[species] = 1
                        else:
                            tot_rxn[species] += 1
                    elif species not in intermediates:
                        intermediates[species] = 1
                    else:
                        intermediates[species] += 1
        if intermediates:
            print('Error in rxn_plot. Intermediates:',intermediates)
        if not deletion_flag:
            print('Error in rxn_plot. No intermediates in reaction?')
        elements = {}
        deletion_flag = False
        for species in tot_rxn:
            composition = self.parse_elements(species)
            for element in composition:
                if element not in elements:
                    elements[element] = composition[element] * tot_rxn[species]
                else:
                    elements[element] += composition[element] * tot_rxn[species]
                if elements[element] == 0:
                    del elements[element]
                    deletion_flag = True
        if elements:
            print('Error in rxn_rate. Stoichiometric imbalance:',elements)
        if not deletion_flag:
            print('Error in rxn_rate. No elements?')
        return rxn_rate

    def plot_rxn(self,rxn_name,outputs,T,p,energy_type='G',pressure_correction=False,file_prefix=''):
        rxns = outputs['std']['rxns']
        #------Check that rxn is OK------
        rxn_path = self.param['plotted_rxns'][rxn_name]
        intermediates = {}
        tot_rxn = {}
        deletion_flag = False
        production_rate = 1e10
        for rxn in rxn_path:  
            if rxn not in rxns:
                #Check if backward_rxn exists
                if 'to' in rxn:
                    left,right = rxn.split('to')
                    backward_rxn = right + 'to' + left
                elif 'ads' in rxn:
                    backward_rxn = rxn.replace('ads','des')
                elif 'des' in rxn:
                    backward_rxn = rxn.replace('des','ads')
                else:
                    raise Exception('Rxn ' + rxn + ' not in rxns, and no reverse rxn found')
                for mode in outputs.keys():                 
                    frxn = {}
                    brxn = outputs[mode]['rxns'][backward_rxn]
                    
                    for pair in [('Ga0_f','Ga0_rev'),('Ga_f','Ga_rev'),
                                 ('Ea0_f','Ea0_rev'),('Ea_f','Ea_rev'),
                                 ('kr','kf'),('l','r'),('m','m')]:
                        frxn[pair[0]] = brxn[pair[1]]
                        frxn[pair[1]] = brxn[pair[0]]
                    for q in ['dEc','dG','dG0','dE','dE0','rate']:
                        frxn[q] = -brxn[q]
                    outputs[mode]['rxns'][rxn] = frxn
                
            if rxns[rxn]['rate'] < production_rate:
                production_rate = rxns[rxn]['rate']
            for species in rxns[rxn]['l']:
                if not '*' in species:
                    if species in self.reactants:
                        if not species in tot_rxn:
                            tot_rxn[species] = -1
                        else:
                            tot_rxn[species] -= 1
                    elif species not in intermediates or intermediates[species] == 0:
                        print('Error in rxn_plot: Species',species, 'not on surface')
                    else:
                        intermediates[species] -= 1
                        if intermediates[species] == 0:
                            del intermediates[species]
                            deletion_flag = True
            for species in rxns[rxn]['r']:
                if not '*' in species:
                    if species in self.products:
                        if not species in tot_rxn:
                            tot_rxn[species] = 1
                        else:
                            tot_rxn[species] += 1
                    elif species not in intermediates:
                        intermediates[species] = 1
                    else:
                        intermediates[species] += 1
        if intermediates:
            print('Error in rxn_plot. Intermediates:',intermediates)
        if not deletion_flag:
            print('Error in rxn_plot. No intermediates in reaction?')
        elements = {}
        deletion_flag = False
        for species in tot_rxn:
            composition = self.parse_elements(species)
            for element in composition:
                if element not in elements:
                    elements[element] = composition[element] * tot_rxn[species]
                else:
                    elements[element] += composition[element] * tot_rxn[species]
                if elements[element] == 0:
                    del elements[element]
                    deletion_flag = True
        if elements:
            print('Error in rxn_plot. Stoichiometric imbalance:',elements)
        if not deletion_flag:
            print('Error in rxn_plot. No elements?')
            
        #-----------Plot----------
        
        plt.clf()
        width = self.rxn_plot_params['width'] #0.5 default
        colors = {'AAI': Dark2['turquoise'], #'#66c2a5', #green
                  'no_AAI': Dark2['rust'],  #'#d95f02', #rust
                  'DFT': Dark2['lavender'], #'#7570b3', #lavender
                 }
        colors['std'] = colors[self.rxn_plot_params['std']]
        
        modes = sorted(outputs) #std should be plotted last
        for mode in modes:
            rxns = outputs[mode]['rxns']
            color = colors[mode]
        
            energy_l = 0 #either free energy or energy
            x0 = 1 #rxn coordinate
            plt.plot((x0-0.5*width,x0+0.5*width),(energy_l,energy_l),lw=2.,
                      c=color,label=self.rxn_plot_params['labels'][mode])
            ticks = []
            labels = []
            for i,rxn in enumerate(rxn_path):           
                x_A = x0+i+0.5*width #connection point left side of rxn and barrier
                x_r = x0+1+i #center for right side of rxn
                x_C = x_r-0.5*width #connection point barrier and right side of rxn
                
                energy_r = energy_l + rxns[rxn]['d' + energy_type]

                ticks.append(x_r-0.5)
                labels.append(rxn.replace('to','->'))
                
                Qf = energy_type + 'a_f' #activation energy
                
                if rxns[rxn][Qf] < 0:
                    raise Exception('Negative barrier in rxn plot')
                elif rxns[rxn][Qf] == 0 or rxns[rxn][Qf] <= rxns[rxn]['d' + energy_type]:
                    plt.plot((x_A,x_C),(energy_l,energy_r),'--',lw=1.5,c=color)
                else:
                    energy_B = energy_l + rxns[rxn][Qf] #energy of barrier
                    x_B = (x_C * np.sqrt(energy_B-energy_l) + x_A * np.sqrt(energy_B-energy_r)) / \
                          (np.sqrt(energy_B - energy_l) + np.sqrt(energy_B - energy_r))
                    #x_B Corresponds to maximum of parabola
                    a = -(energy_B - energy_l) / (x_B - x_A)**2
                    b = -2*a*x_B
                    c = energy_B + a*x_B**2
                    xs = np.linspace(x_A,x_C,20,endpoint=True)
                    ys = a * xs**2 + b * xs + c
                    plt.plot(xs,ys,'--',lw=1.5,c=color)           

                plt.plot((x_r-0.5*width,x_r+0.5*width),(energy_r,energy_r),lw=2.,c=color)
                
                energy_l = energy_r
        
        plt.legend()
        if energy_type == 'G':
            plt.ylabel('Free energy (eV)')
        elif energy_type == 'E':
            plt.ylabel('Energy (eV)')
        plt.title(rxn_name + '(' + str(T) + ' K, ' + str(p) + ' Pa), r=' + "{:.3e}".format(production_rate))
        plt.xticks(ticks,labels,rotation=90)
        plt.tight_layout()
        plt.savefig(self.rxn_plot_params['plot_dir'] + '/' + file_prefix + rxn_name + '_' + energy_type + '.pdf')
        return production_rate

    def calc_spectra_deviation(self,coverages,var_BE=None,var_sigma=None):
        if var_BE == None:
            BEs = self.BE
        else:
            BEs = var_BE
        if var_sigma == None:
            sigmas = self.sigma
        else:
            sigmas = var_sigma #can have incorrect depth
        calc_data = {}
        delta_data = {}
        for element in self.exp_data:
            calc_data[element] = {}
            delta_data[element] = {}
            for Tp in self.exp_data[element]:
                calc_data[element][Tp] = {'cov':0.0}
                delta_data[element][Tp] = {}
                for ads in self.element_weighted_ads[element]:
                    calc_data[element][Tp]['cov'] += coverages[Tp][ads]
                delta_data[element][Tp]['cov'] = calc_data[element][Tp]['cov'] - self.exp_data[element][Tp]['cov']
                
                if self.exp_data[element][Tp]['spectrum_weight']:
                    calc_spectrum = np.zeros(len(self.BE_axis[element]))
                    for ads in BEs[element]:
                        #sigma_list
                        if isinstance(sigmas[element],list):
                            #True if only one, variable sigma
                            sigma_list = sigmas[element]
                        elif isinstance(sigmas[element][ads],list):
                            #True if temperature-independent sigmas
                            sigma_list = sigmas[element][ads]
                        else:
                            sigma_list = sigmas[element][ads][Tp]
                            
                        for BE,sigma in zip(BEs[element][ads],sigma_list):
                            gaussian = coverages[Tp][ads] * 1./(np.sqrt(2. * np.pi) * sigma) * \
                                       np.exp(-(self.BE_axis[element] - BE)**2 / (2 * sigma**2))
                            calc_spectrum += gaussian
                    calc_data[element][Tp]['spectrum'] = calc_spectrum
                    delta_data[element][Tp]['spectrum'] = self.exp_data[element][Tp]['spectrum'] - calc_spectrum

        return delta_data, calc_data
    
    def calc_coverage_deviation(self,coverages,delta_logarithm=False):
        calc_covs = copy.deepcopy(self.exp_covs)
        delta_logarithm = self.param.get('delta_logarithm',delta_logarithm)
        
        for element in calc_covs:
            for ads_group in calc_covs[element]:
                for Tp in calc_covs[element][ads_group]:
                    #Tp_exp_set.add(Tp)
                    calc_covs[element][ads_group][Tp] = 0
                
        delta_covs = copy.deepcopy(calc_covs)
        
        def dev(lin_exp_cov,lin_calc_cov,exp_marg,log=delta_logarithm):
            if log:
                exp_cov = np.log10(lin_exp_cov)
                calc_cov = np.log10(lin_calc_cov)
            else:
                exp_cov = lin_exp_cov
                calc_cov = lin_calc_cov
	        
            #exp_margs must both be >= 0
            if exp_marg < 0:
                raise Exception('Exp margs must be non-negative.')

            min_exp_cov = exp_cov - exp_marg
            max_exp_cov = exp_cov + exp_marg

            if min_exp_cov < calc_cov < max_exp_cov:
                deviation = 0.0
            elif calc_cov <= min_exp_cov:
                deviation = min_exp_cov - calc_cov
            elif calc_cov >= max_exp_cov:
                deviation = max_exp_cov - calc_cov
		        
            return deviation

        for element in calc_covs:
            for ads_group in calc_covs[element]:
                for Tp in calc_covs[element][ads_group]:
                    for ads in ads_group:
                        calc_covs[element][ads_group][Tp] += coverages[Tp][ads]
                    delta_covs[element][ads_group][Tp] = dev(self.exp_covs[element][ads_group][Tp]['cov'],
						                                     calc_covs[element][ads_group][Tp],
						                                     self.exp_covs[element][ads_group][Tp]['marg'])
        return delta_covs, calc_covs
        
    def calc_spectra_fitness(self,delta_data,spectra_coef=10.0,norm=1):
        spectra_coef = self.param.get('spectra_coef',spectra_coef)
        spectra_sum = 0.
        sp_wt_sum = 0.
        tot_cov_sum = 0.
        cov_wt_sum = 0.
        for element in self.exp_data:
            for Tp in self.exp_data[element]:
                if self.exp_data[element][Tp]['spectrum_weight']:
                    spectra_sum += self.exp_data[element][Tp]['spectrum_weight'] * \
                                       np.linalg.norm(delta_data[element][Tp]['spectrum'],ord=norm) / \
                                       len(delta_data[element][Tp]['spectrum'])
                    sp_wt_sum += self.exp_data[element][Tp]['spectrum_weight']
                tot_cov_sum += self.exp_data[element][Tp]['cov_weight'] * \
                               abs(delta_data[element][Tp]['cov'])**norm #self.exp_data[element][Tp]['cov']
                cov_wt_sum += self.exp_data[element][Tp]['cov_weight']
        spectra_deviation_fitness = spectra_coef * spectra_sum**(1./norm) / sp_wt_sum
        cov_deviation_fitness = tot_cov_sum**(1./norm) / cov_wt_sum
        return spectra_deviation_fitness, cov_deviation_fitness
                    
    def cov_deviation_fitness(self,delta_covs,norm=1):
        if not delta_covs:
            raise Exception('delta_covs empty')
        cov_sum = 0
        weight_sum = 0
        for element in delta_covs:
            for ads_group in delta_covs[element]:
                for Tp in delta_covs[element][ads_group]:
                    term = np.abs(self.exp_covs[element][ads_group][Tp]['weight'] *\
                                  delta_covs[element][ads_group][Tp])**norm
                    cov_sum += term
                    weight_sum += self.exp_covs[element][ads_group][Tp]['weight']

        cov_deviation_fitness = cov_sum**(1./norm) / weight_sum

        return cov_deviation_fitness
        
    def dft_deviation_penalty_fitness(self,energies,dft_deviation_penalty_model='quadratic'):
        dft_deviation_penalty_model = self.param.get('dft_deviation_penalty_model',dft_deviation_penalty_model)
        if energies == None:
            deviation_penalty = 0.
            max_deviation = 0.
        else:
            if not energies.keys() == self.dft_energies.keys():
                print(energies.keys(),self.dft_energies.keys())
                raise Exception('Mismatch energy labels')
            if dft_deviation_penalty_model is None:
                return 0
            elif dft_deviation_penalty_model == 'linear':
                order = 1
            elif dft_deviation_penalty_model == 'quadratic':
                order = 2
                
            deviation_penalty = 0
            max_deviation = 0
            for species in energies:
                deviation = (energies[species] - self.dft_energies[species])
                if deviation > max_deviation:
                    max_deviation = deviation
                deviation_penalty += self.coefs[species] * deviation ** order
            deviation_penalty /= len(energies)
        return deviation_penalty, max_deviation

    def aai_penalty_fitness(self,var_aai_slopes,max_penalty=0.05):
        max_penalty = self.param.get('max_aai_penalty',max_penalty)
        if var_aai_slopes == None:
            fitness = 0.
        else:
            penalty = 0.0
            nb_vars = 0
            for species in var_aai_slopes:
                if isinstance(var_aai_slopes[species],dict):
                    for co_ads in var_aai_slopes[species]:
                        nb_vars += 1
                        if var_aai_slopes[species][co_ads] > 0.:
                            penalty += 1.0
                elif isinstance(var_aai_slopes[species],float):
                    nb_vars += 1
                    if var_aai_slopes[species] > 0.:
                        penalty += 1.0
            if nb_vars > 0:
                fitness = penalty / nb_vars * max_penalty
            else:
                fitness = 0.
        return fitness

    """
    def list_to_dict2(self,inputs, frame_dict):
        #inputs is a list of variables
        #frame_dict is a dict of dicts with an ordered structure
        list_ptr = 0
        output_dict = {}
        for var_type in frame_dict:
            output_dict[var_type] = {}
            for species in frame_dict[var_type]:
                if isinstance(frame_dict[var_type][species],dict):
                    output_dict[var_type][species] = {}
                    for sub_key in frame_dict[var_type][species]:
                        output_dict[var_type][species][sub_key] = inputs[list_ptr]
                        list_ptr += 1
                else:
                    output_dict[var_type][species] = inputs[list_ptr]
                    list_ptr += 1
        if list_ptr != len(inputs):
            #total elements of frame_dict and inputs should be the same
            raise Exception('Length of list and dict of dicts do not agree')
        return output_dict
    """
    
    def list_to_dict(self,input_list,frame_dict):
        #pprint.pprint(frame_dict)
        #quit()
        def subdivide(input_list,new_dict,ptr=0):
            for key in new_dict:
                if isinstance(new_dict[key],(float,int)):
                    try:
                        new_dict[key] = input_list[ptr]
                    except IndexError:
                        pprint.pprint(new_dict)
                        pprint.pprint(input_list)
                        raise Exception('Too short input_list')
                    except Exception:
                        raise
                    ptr += 1
                elif isinstance(new_dict[key],list):
                    length = len(new_dict[key])
                    try:
                        new_dict[key] = [input_list[i] for i in range(ptr,ptr+length)]
                    except IndexError:
                        pprint.pprint(new_dict)
                        pprint.pprint(input_list)
                        raise Exception('Too short input_list')
                    except Exception:
                        raise  
                    ptr += length      
                elif isinstance(new_dict[key],dict):
                    ptr = subdivide(input_list,new_dict[key],ptr=ptr) #Recursion
                else:
                    raise Exception('Type error with input_list')
            return ptr
        new_dict = copy.deepcopy(frame_dict)
        ptr = subdivide(input_list,new_dict)
        if ptr != len(input_list):
            raise Exception('Too long input list')
        return new_dict
    
    def float_eq(a,b,tol=0.001):
        if isinstance(a,float):
            equal = abs(a - b) < tol
        else:
            equal = True 
            for i,j in zip(a,b):
                equal = equal and abs(i - j) < tol
        return equal
        
    def finish(self,running_file='incomplete',complete_file='completed'):
        if os.path.isfile(running_file):
            os.remove(running_file)
        f = open(complete_file,'w')
        f.close()
        
        self.write_report()
        self.write_report(filename=None)
        self.write_pckl()
        return
        
    #------------------------Writing/reading functions-------------------------------
    #
    #--------------------------------------------------------------------------------
    
    def read_dict_file_exec(self, dict_file):
        #Execute python file with the dictionary 'master'
        if os.path.exists(dict_file) and dict_file[-3:] == '.py':
            f = open(dict_file)
            content = f.read()
            exec(content)
        elif not os.path.exists(dict_file):
            raise Exception(dict_file + ' does not exist')
        else:
            raise Exception(dict_file + ' must have .py extension')
        return locals()['master']

    def write_dict_file(self,dict_file,master,width=120):
        #Write dictionary to file.
        if dict_file[-3:] != '.py':
            raise Exception('dict file must have .py extension')
        if os.path.exists(dict_file):
            f = open(dict_file,'r')
            lines = f.readlines()
            if '#NO OVERWRITING BY PYTHON\n' in lines:
                raise Exception('No overwriting allowed for ' + dict_file)
            f.close()
        else:
            lines = ['#Write below this line\n']
        
        f = open(dict_file,'w')
        end_of_copying = False
        for line in lines:  
            if line != '#Write below this line\n':
                f.write(line)
            else:
                f.write(line)
                end_of_copying = True
                break
        if not end_of_copying:
            raise Exception("Could not find '#Write below this line'")
        f.write('master = {}\n')
        for key in master:
            f.write("\n")
            if isinstance(master[key],(list,dict,np.ndarray)):
                f.write("master['" + str(key) + "'] = \\\n")
                f.write(pprint.pformat(master[key],width=width,sort_dicts=False) + "\n")
            else:
                f.write("master['" + str(key) + "'] = " + str(master[key]))
        f.close()

    def write_pckl(self,filename='gm.pckl'):
        #Save the object as pickle file
        fd = open(filename,'wb')
        try:
            pickle.dump(self,fd)
            fd.close()
        except TypeError as e:
            print('TypeError: Check that no dict_keys are saved')
            print(e)
            fd.close()
            os.remove(filename)
        except Exception as e:
            print(e)
            os.remove(filename)
        return
        
        
    def write_report(self,filename='report.txt',func_digits=3,comp_digits=4):
        if filename == None:
            fd = sys.stdout
        else:
            fd = open(filename,'w')
        
        #Fitness
        fitness = round(self.fitness,func_digits)
        print('Fitness:','{:.3f}'.format(self.fitness,func_digits),file=fd)
        comp = self.fitness_composition.copy()
        for key in comp:
            comp[key] = round(comp[key],comp_digits)
        print('Composition:',file=fd)
        pprint.pprint(comp,stream=fd)
        
        #Runtime
        total_seconds = time.time() - self.start_time
        total_minutes = np.floor(total_seconds / 60)
        seconds = round(total_seconds - 60 * total_minutes,2)
        hours = round(np.floor(total_minutes / 60))
        minutes = round(total_minutes - 60 * hours)
        print("\n--- %s hours %s minutes %s seconds ---" % (hours, minutes, seconds),file=fd)
        
        if filename != None:
            fd.close()
        
        return
        
    def write_coverages(self,filename='coverages.txt',exclude_species=[],cutoff=0.005,digits=2):
        if isinstance(exclude_species,str):
            exclude_species = [exclude_species] 

        Tps = list(self.species_coverages.keys())
        Tps.sort()
        
        #adsorbates = [t[0] for t in sorted(self.species_coverages[Tps[0]].items(),key=lambda x:x[1])]
        #adsorbates.reverse()
        
        f = open(filename,'w')            
        f.write('Temperature (K), Pressure (mbar)\n')

        for Tp in Tps:
            f.write(str(Tp))
            f.write('\n')
            adsorbates = [t[0] for t in sorted(self.species_coverages[Tp].items(),key=lambda x:x[1])]
            adsorbates.reverse()
            for ads in adsorbates: #self.species_coverages[Tp]:
                if ads not in exclude_species and self.species_coverages[Tp][ads] > cutoff:
                    f.write('\t' + ads + ': ' + str(round(self.species_coverages[Tp][ads],digits)) + '\n')
        f.close()
        return
        
    def save_corr_details(self,slopes,species_coverages,filename='aai_details.py'):
        corrs = {}
        for Tp in species_coverages:
            corrs[Tp] = {}
            for species in self.adsorbates + self.transition_states:
                if species in slopes:
                    Ec = 0.
                    corrs[Tp][species] = {}
                    for co_ads in slopes[species]:
                        if species_coverages[Tp][co_ads] > 0.001:
                            slope = slopes[species][co_ads]
                            coverage = species_coverages[Tp][co_ads]
                            corrs[Tp][species][co_ads] = "{:.3f}".format(slope) + " * " + "{:.3f}".format(coverage)
                            Ec += slope * coverage
                    corrs[Tp][species]['Ec'] = round(Ec,3)
        self.write_dict_file(filename,corrs,width=40)
        
    def write_aai_slopes(self,filename='aai_params.txt',ts_coads=[]):
        fd = open(filename,'w')
        slopes = self.parse_slopes(self.var_slopes)
        self_interaction = []
        for ads in self.adsorbates:
            if ads in slopes and ads in slopes[ads]:
                self_interaction.append((ads,slopes[ads][ads]))
        self_interaction.sort(key=lambda x: x[1])
        self_interaction.reverse()
        fd.write('Self interaction\n')
        for pair in self_interaction:
            fd.write(pair[0].ljust(8) + str(round(pair[1],2)) + '\n')
        fd.write('\nTS cross parameters\n')
        for ts in self.transition_states:
            if ts in slopes:
                for coads in ts_coads:
                    if coads in slopes[ts]:
                        fd.write(ts.ljust(10) + coads.ljust(8) + str(round(slopes[ts][coads],2)) + '\n')
        return
            

    def write_rates(self,filename='rates.txt',gases='all'):
        #Write production rates to file
        # gases <list> - Gases to write. The rest of the gases are skipped.
        # also possible <str> - 'all','products','reactants'
        if gases == 'all':
            gases = self.gases.copy()
        elif gases == 'products':
            gases = self.products
        elif gases == 'reactants':
            gases = self.reactants
        element_rate_coefs = {}
        for gas in self.gases:
            comp = self.parse_elements(gas)
            for element in comp:
                if element not in element_rate_coefs:
                    element_rate_coefs[element] = {gas: comp[element]}
                else:
                    element_rate_coefs[element][gas] = comp[element]
        #print(element_rate_coefs)
        f = open(filename,'w')
        f.write('Rates\n')
        f.write('Balance rates should be small. They have to be smaller than the major product(s)\n')
        f.write('Temperature (K), Pressure (mbar)\n')
        for Tp in self.reactions:
            rates = self.get_production_rates(self.reactions[Tp])
            balance = {}
            for element in element_rate_coefs:
                rate = 0
                for gas in element_rate_coefs[element]:
                    rate += element_rate_coefs[element][gas] * rates[gas]
                balance[element] = rate
                        
            f.write(str(Tp))
            f.write('\n')
            
            #--------Check if rates of reactants and products match--------
            warning_allowed = True
            for gas in gases:
                if gas in self.reactants:
                    max_log = -20.
                    add_str = '       log balance/rate: '
                    for el in balance:
                        log = np.log10(abs(balance[el])) - np.log10(abs(rates[gas]))
                        max_log = max(max_log,log)
                        add_str += el + ' ' + str(round(log,1)).rjust(4) + ' '
                    if max_log > -1.0 and warning_allowed:
                        print(Tp,'WARNING: Rates seem not to be converged! Balance > 10% of rate.')
                        warning_allowed = False
                else:
                    add_str = ''
                    
                f.write('    ' + (gas + ':').ljust(11) + "{:.3e}".format(rates[gas]).rjust(10) + add_str + '\n')
            

            for element in element_rate_coefs:
                f.write('    ' + element + ' balance: ' + "{:.3e}".format(balance[element]).rjust(10) + '\n')

        f.close()
        return
     

        
    #------------------------------Plotting functions--------------------------------
    #
    #--------------------------------------------------------------------------------
    
    #plt_rxn is found further above
    
    def format_species(self,species,subscript=True,no_site=True):
        #Take a species like 'CH3O_s' and format it for display
        #subscript - '3' will be subscripted in 'CH3O_s'
        #no_site - '_s' is removed from 'CH3O_s'. '_g' is not removed
        if no_site:
            spec,site = species.split('_')
            if site == 'g':
                formatted_species = species
            else:
                formatted_species = spec
        if subscript:
            new_species = ''
            sub = False
            for char in formatted_species:
                if char.isdigit():
                    if not sub:
                        new_species += '$_{'
                    sub = True
                else:
                    if sub:
                        new_species += '}$'
                    sub = False
                new_species += char
            if sub:
                new_species += '}$'
            formatted_species = fr"{new_species}"
        return formatted_species
    
    def plot_bar(self,filename='bar.pdf',title=None,margin=None,plotted_species='all',
                 activation_energy=False):
        #Plots a bar diagram of (zero-coverage) E_GA minus E_DFT.
        # margin <float> - Plot from -margin to margin on y axis
        # plotted_species <str> - 'all','ts','ads','minima' (=ads and gases)
        # activation_energy <bool> - True: Plot activation energy, i.e. barriers
        #                            False: Plot transition state rel. to gas refs
        
        #six_class_paired = ['#a6cee3','#1f78b4','#b2df8a',
        #                    '#33a02c','#fb9a99','#e31a1c']
        six_class_paired = list(Paired.values())[:6]
        
        species_dict = {'all': self.name_list,
                        'ts': self.transition_states,
                        'ads': self.adsorbates,
                        'minima': self.adsorbates + self.gases}
        species_list = species_dict[plotted_species]
        
        for state in self.param['reference_states']:
            if state in species_list:
                if self.ga_energies[state] != 0.:
                    raise Exception('Error with reference energies')
                species_list.remove(state) 

        nb_energies = len(species_list)
        
        fig_width = 10. * nb_energies / 36 + 0.85
        
        plt.clf()
        fig = plt.figure(facecolor=(1,1,1),figsize=(fig_width,5))
        
        if activation_energy:
            max_delta_E_a = 0.
        
        if len(self.ga_energies) != len(self.dft_energies):
            raise Exception('Number of DFT/GA energies do not match')
        X = np.arange(nb_energies)
        
        barrier_prestates = {}
        for r in self.param['rxns']:
            if 'm' in self.param['rxns'][r]:
                for species in self.param['rxns'][r]['m']:
                    if '-' in species:
                        ts = species
                prestates = []
                for species in self.param['rxns'][r]['l']:
                    if not '*' in species:
                        prestates.append(species)
                barrier_prestates[ts] = prestates   
        
        ticks = []
        for i, label in enumerate(species_list):
            j = i % 2
            if label[-2:] == '_g':
                tick = label
            else:
                tick = label[:-2]
            plotted = False
            if '-' in label:
                col = six_class_paired[j+2] #ts
                if activation_energy:
                    ga_ts = self.ga_energies[label]
                    dft_ts = self.dft_energies[label]
                    ga_prestates = 0.
                    dft_prestates = 0.
                    prelabel = ''
                    for ps in barrier_prestates[label]:
                        ga_prestates += self.ga_energies[ps]
                        dft_prestates += self.dft_energies[ps]
                        if ps[-2:] == '_g':
                            prelabel += ps
                        else:
                            prelabel += ps[:-2]
                        prelabel += ','
                    prelabel = prelabel[:-1] #rm last comma
                    tick = prelabel + '->' + tick
                    ticks.append(tick)
                    delta_E_a = ga_ts - ga_prestates - (dft_ts - dft_prestates)
                    if abs(delta_E_a) > max_delta_E_a:
                        max_delta_E_a = abs(delta_E_a)
                    plt.bar(i,delta_E_a,color=col)
                    plotted = True
            elif '_g' in label:
                col = six_class_paired[j+4] #gas
            else:
                col = six_class_paired[j] #ads
            if not plotted:
                plt.bar(i,self.ga_energies[label] - self.dft_energies[label],color=col)
                ticks.append(tick)
                plotted = True
        
        plt.xticks(X,rotation=90)
        
        if margin == None:
            if activation_energy:
                margin = max_delta_E_a * 1.05
            else:
                margin = max(self.energy_bounds.values())
     
        plt.ylim([-margin,margin])
        plt.gca().set_xticklabels(ticks)
        if plotted_species == 'ts' and activation_energy:
            plt.ylabel(r'$\Delta E_A$ GA/DFT (eV)')
        else:
            plt.ylabel(r'$\Delta E_0$ GA/DFT (eV)')
        plt.xlabel('Species')
        plt.title(title)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        return
                
    def init_subplots(self,nb_rows,nb_cols,
                      title_space=1.0,
                      subplot_height_space=1.25,
                      hspace=0.5,
                      bottom_space=0.8,
                      left_space=0.7,
                      right_space=0.4,
                      subplot_width_space=2.5,
                      wspace=0.25):
        #Supporting function for other plots
        
        height = title_space + nb_rows * subplot_height_space + \
                 (nb_rows - 1) * hspace + bottom_space

        width = left_space + nb_cols * subplot_width_space + \
                (nb_cols - 1) * wspace + right_space
                
        if nb_cols == 1:
            if nb_rows == 1:
                f, ax = plt.subplots(1, 1,figsize=(width,height))
                axs = np.array([[ax]])
            else:
                f, ax = plt.subplots(nb_rows, 1,figsize=(width,height))
                axs = np.array([list(ax)]).T
        else:
            if nb_rows == 1:
                f, ax = plt.subplots(1, nb_cols,figsize=(width,height))
                axs = np.array([list(ax)])
            else:
                f, ax = plt.subplots(nb_rows, nb_cols,figsize=(width,height))
                axs = np.array(list(ax))
            
        tspace = 1 - title_space / height
        bspace = bottom_space / height
        lspace = left_space / width
        rspace = 1 - right_space / width
        plt.subplots_adjust(left=lspace,right=rspace,top=tspace,
                            bottom=bspace,hspace=hspace,wspace=wspace)
        return axs,height
           
        
    def plot_total_coverage(self,file_prefix='cmp_cov',
                            resolve_rest_tot=True,
                            stacked_bars=False,stack_min_cov=0.01,avoid_bright=False,
                            celsius=True,mbar=True,
                            specific_species=['H_h'],
                            max_coverage=None, #float or dict, default 1.0
                            ads_group_exclusion_cov=None,
                            bar_width=8.0,
                            title_space=1.0,
                            subplot_height_space=1.25,
                            hspace=0.5,
                            bottom_space=0.8,
                            left_space=0.7,
                            right_space=0.4,
                            subplot_width_space=2.5,
                            wspace=0.25):
        #Plot total coverage for each element, for temperatures and pressures
        # file_prefix <str> - Filename prefix. Filename is followed by element or species
        # resolve_rest_tot <bool> - Whether to write 'Rest' or specify adsorbates
        # stacked_bars <bool> - Whether to split up the calculated bar in species
        # stack_min_cov <float> - Minimum coverage to have its own bar if stacked_bars
        # avoid_bright <bool> - Avoid brightest nuance of sequential colors
        # specific_species <list> - Plot total coverage of specific species
        # max_coverage <float> or <dict> - Plotted max coverage in ML. If dict, e.g. 
        #                                  max_coverage['C'] = 0.9
        # ads_group_exclusion_cov <float> - If an adsorbate group has lower coverage than
        #                                   this, it is not plotted, e.g. N2_s
        # 

        if self.exp_type == 'spectra':
            exp_covs = {}
            calc_covs = {}
            for element in self.exp_data:
                exp_covs[element] = {self.element_weighted_ads[element]:{}}
                calc_covs[element] = {self.element_weighted_ads[element]:{}}
                for Tp in self.exp_data[element]:
                    exp_covs[element][self.element_weighted_ads[element]][Tp] = {'cov': self.exp_data[element][Tp]['cov']}
                    if 'marg' in self.exp_data[element][Tp]:
                        exp_covs[element][self.element_weighted_ads[element]][Tp]['marg'] = self.exp_data[element][Tp]['marg']
                    else:
                        exp_covs[element][self.element_weighted_ads[element]][Tp]['marg'] = 0.
                    calc_covs[element][self.element_weighted_ads[element]][Tp] = self.calc_data[element][Tp]['cov']
            self.exp_covs = exp_covs
            self.calc_covs = calc_covs
        
        def format_ads_group(element,ads_group,resolve_rest_tot=resolve_rest_tot):
            if not resolve_rest_tot:
                ads_tuple = ads_group
            else:    
                if self.exp_type == 'spectra':
                    return 'Tot'
                elif ads_group in self.rest_tot_dict[element]:
                    ads_tuple = self.rest_tot_dict[element][ads_group]

            string = str(ads_tuple)
            string = string.replace("'","")
            string = string.replace(" ","")
            string = string.replace("("," ")
            string = string.replace(")"," ")
            return string
            
                            
        if max_coverage is None:
            max_coverage = {'default': 1.0}
        elif isinstance(max_coverage,float):
            max_coverage = {'default': max_coverage}
        elif isinstance(max_coverage,dict) and 'default' not in max_coverage:
            max_coverage['default'] = 1.0
            
        for element in self.exp_covs:
            if element not in max_coverage:
                max_coverage[element] = max_coverage['default']
        for species in specific_species:
            if species not in max_coverage:
                max_coverage[species] = max_coverage['default']
            
        
        plt.clf()
        calc_color = Reds[4][3] #i.e. '#cb181d' Earlier '#e41a1c', i.e. Set1['red']
        exp_color = Set1['blue'] #'#377eb8'
        
        yticks = {}
        for plot in max_coverage:       
            if 0.6 <= max_coverage[plot] <= 1.0:
                yticks[plot] = [0.0,0.2,0.4,0.6,0.8,1.0]
            elif 0.3 <= max_coverage[plot] < 0.6:
                yticks[plot] = [0.0,0.1,0.2,0.3,0.4,0.5]
            else:
                yticks[plot] = [0.0,0.05,0.1,0.15,0.20,0.25]           
        
        if celsius:
            Ts_label = [T - 273.15 for T in self.Ts]
            xlabel = u'Temperature (\u00B0C)'
        else:
            Ts_label = self.Ts
            xlabel = 'Temperature (K)'
        
        #----------Specific species, like 'H_h'----------
        for species in specific_species:
            axs,height = self.init_subplots(1,len(self.ps))
            i = 0
            for Tp in self.species_coverages:
                (T,p) = Tp
                if celsius:
                    T -= 273.15
                for j,p_listed in enumerate(self.ps):
                    if p == p_listed:
                        col = j
                axs[i,col].bar(T, self.species_coverages[Tp][species],
                               width=bar_width, color=calc_color)
            for j,p in enumerate(self.ps):
                subtitle = '' #species.split('_')[0]
                if i == 0:
                    if mbar:
                        p_text = str(int(round(p/100))) + ' mbar\n'
                    else:
                        p_text = str(p) + ' Pa\n'
                    subtitle = str(p_text + subtitle)
                axs[i][j].set_title(species.split('_')[0])
                axs[i][j].set_xticks(Ts_label)
                if len(Ts_label) == 1:
                    axs[i][j].set_xlim(Ts_label[0] - 3 * bar_width,Ts_label[0] + 3 * bar_width)
                axs[i][j].set_yticks(yticks[species])
                axs[i][j].set_ylim(0,max_coverage[species])
                calc_label = mpatches.Patch(color=calc_color,label='Calc')
                axs[i][j].legend(handles=[calc_label])
            axs[i][0].set_ylabel(r'$\theta$')
            for j in range(len(self.ps)):
                axs[-1][j].set_xlabel(xlabel)

            y_suptitle = 1 - 0.4 / height 
            plt.suptitle(species.split('_')[0] + ' coverage',size=20,y=y_suptitle)
            plt.savefig(file_prefix + '_' + species + '.pdf') #cmp_cov
        
        #------------------Ordinary plot-------------------
        for element in self.exp_covs:
            if ads_group_exclusion_cov:
                ads_groups = []
                for ads_group in self.exp_covs[element]:
                    max_cov = 0.
                    for Tp in self.exp_covs[element][ads_group]:
                        max_cov = max(self.exp_covs[element][ads_group][Tp]['cov'],
                                      self.calc_covs[element][ads_group][Tp],
                                      max_cov)
                    if max_cov >= ads_group_exclusion_cov:
                        ads_groups.append(ads_group)
            else:
                ads_groups = list(self.exp_covs[element].keys())

            axs,height = self.init_subplots(len(ads_groups),len(self.ps))
            
            # For stacked_bars
            #red_colors = {'a':'#fee5d9','b':'#fcbba1','c':'#fc9272','d':'#fb6a4a',
            #                  'e':'#ef3b2c','f':'#cb181d','g':'#99000d'}
            #priority = ['f','d','b','c','e','g','a']
            
            for i,ads_group in enumerate(ads_groups):
            
                if stacked_bars:
                    labelled_ads = []
                    ascending_Tps = list(self.species_coverages.keys())
                    ascending_Tps.sort()

                    for Tp in ascending_Tps:
                        sign_ads = []
                        for ads in ads_group:
                            element_cov = self.species_coverages[Tp][ads] * ads_group.count(ads) #e.g. if multiple C atoms
                            if element_cov > stack_min_cov and \
                               ads not in labelled_ads:
                                sign_ads.append((ads,element_cov))
                        sign_ads = sorted(sign_ads,key=lambda item: item[1])
                        sign_ads.reverse()
                        labelled_ads += [ads[0] for ads in sign_ads]
                    labelled_ads.reverse()
                    nb_fields = len(labelled_ads) + 1

                    if avoid_bright:
                        red_colors = Reds[nb_fields + 1][1:]
                    else:
                        red_colors = Reds[nb_fields]
                    #print(labelled_ads)
                    #---------------------------------------

                for Tp in self.exp_covs[element][ads_group]:
                    (T,p) = Tp
                    if celsius:
                        T -= 273.15
                    for j,p_listed in enumerate(self.ps):
                        if p == p_listed:
                            col = j
                    
                    self.exp_covs[element][ads_group][Tp]['cov']
                    yerr = self.exp_covs[element][ads_group][Tp]['marg']
                    if yerr == 0.0:
                        yerr = None
                    axs[i,col].bar(T - 0.5 * bar_width, self.exp_covs[element][ads_group][Tp]['cov'],
                                   width=bar_width,color=exp_color,
                                   yerr=yerr,capsize=2)
                    if stacked_bars:
                        bottom = 0
                        for k,ads in enumerate(labelled_ads):
                            element_cov = self.species_coverages[Tp][ads] * ads_group.count(ads)
                            if element_cov > stack_min_cov:
                                axs[i,col].bar(T + 0.5 * bar_width,element_cov,
                                               width=bar_width,bottom=bottom,
                                               #color=red_colors[nuances[k]])
                                               color=red_colors[k])
                                bottom += element_cov
                            rest_cov = self.calc_covs[element][ads_group][Tp] - bottom
                        axs[i,col].bar(T + 0.5 * bar_width,rest_cov,width=bar_width,bottom=bottom,
                                       #color=red_colors[nuances[-1]])
                                       color=red_colors[-1])
                    else:
                        axs[i,col].bar(T + 0.5 * bar_width, self.calc_covs[element][ads_group][Tp],
                                       width=bar_width, color=calc_color)
                for j,p in enumerate(self.ps):
                    subtitle = format_ads_group(element,ads_group)
                    if i == 0:
                        if mbar:
                            p_text = str(int(round(p/100))) + ' mbar\n'
                        else:
                            p_text = str(p) + ' Pa\n'
                        subtitle = p_text + subtitle
                    axs[i][j].set_title(subtitle)
                    axs[i][j].set_xticks(Ts_label)
                    if len(Ts_label) == 1:
                        axs[i][j].set_xlim(Ts_label[0] - 3 * bar_width,Ts_label[0] + 3 * bar_width)
                    axs[i][j].set_yticks(yticks[element])
                    axs[i][j].set_ylim(0,max_coverage[element])
                    exp_label = mpatches.Patch(color=exp_color,label='Exp')
                    handles = [exp_label]
                    if stacked_bars:
                        calc_label = mpatches.Patch(color=red_colors[-1],label='Rest')
                        handles.append(calc_label)
                        for k in range(nb_fields - 2,-1,-1):
                            #print(k)
                            calc_label = mpatches.Patch(color=red_colors[k],label=labelled_ads[k].split('_')[0])
                            handles.append(calc_label)
                    else: 
                        calc_label = mpatches.Patch(color=calc_color,label='Calc')
                        handles.append(calc_label)
                    axs[i][j].legend(handles=handles,fontsize='x-small')
                axs[i][0].set_ylabel(r'$\theta$')
            for j in range(len(self.ps)):
                axs[-1][j].set_xlabel(xlabel)
            y_suptitle = 1 - 0.2 / height 
            plt.suptitle(element + ' coverage',size=20,y=y_suptitle)
            plt.savefig(file_prefix + '_' + element + '.pdf')
        return

    def plot_spectra(self,filename='spectra',celsius=True,mbar=True,write_species=False):
        #Plot XPS spectra in subplots rows-temperatures, columns-pressures
        # write_species <bool> - Write species names in plot by their binding energy
        
        if self.exp_type != 'spectra':
            raise Exception("exp_type must be 'spectra'")
                
        plt.clf()
        calc_color=Set1['red'] #'#e41a1c'
        exp_color=Set1['blue'] #'#377eb8'
        
        for element in self.exp_data:
            axs,height = self.init_subplots(len(self.Ts),len(self.ps))
            save_plot = False
            #texts = []
            for i,T in enumerate(self.Ts):
                for j,p in enumerate(self.ps):
                    ax = axs[i][j]
                    texts = []
                    if (T,p) in self.exp_data[element] and 'spectrum' in self.exp_data[element][(T,p)]:
                        save_plot = True
                        ax.plot(self.BE_axis[element],self.exp_data[element][(T,p)]['spectrum'],
                                c=exp_color,label='Exp')
                        ax.plot(self.BE_axis[element],self.calc_data[element][(T,p)]['spectrum'],
                                c=calc_color,label='Calc')
                        ax.legend()
                        if write_species:
                            for species in self.BE[element]:

                                for BE_peak in self.BE[element][species]:
                                    Text = ax.text(BE_peak,0,self.format_species(species),
                                                   rotation=90,fontsize=4.5,ha='center')
                                    texts.append(Text)
                                #ax.get_ticks() #Should be 288,286,284,282
                            ax.figure.draw_without_rendering()
                            bbox_list = []
                            transform_display_to_data = ax.transData.inverted().transform
                            for text in texts:
                                curr_bbox = text.get_window_extent().get_points()
                                data_bbox = transform_display_to_data(curr_bbox)
                                offset = 0
                                current_ymin = curr_bbox[0][1]
                                current_ymax = curr_bbox[1][1]

                                placement_ok = False
                                counter = 0
                                while not placement_ok:
                                    overlap = False
                                    for bbox in bbox_list:
                                        #one's max minus other's. If one of these differences is negative,
                                        #the sets are disjoint and no overlap
                                        if curr_bbox[1][0] - bbox[0][0] > 0 and bbox[1][0] - curr_bbox[0][0] > 0 and \
                                           curr_bbox[1][1] - bbox[0][1] > 0 and bbox[1][1] - curr_bbox[0][1] > 0:
                                            overlap = True
                                            offset = max(offset,bbox[1][1] - current_ymin + 3)
                                            curr_bbox[0][1] = current_ymin + offset
                                            curr_bbox[1][1] = current_ymax + offset
                                            break
                                    placement_ok = not overlap
                                    counter += 1
                                    if counter >= 100:
                                        raise Exception()

                                bbox_list.append(curr_bbox)
                                pos = list(text.get_position())
                                new_data_bbox = transform_display_to_data(curr_bbox)
                                data_offset = new_data_bbox[0][1] - data_bbox[0][1]

                                pos[1] += data_offset
                                text.set_position(pos)
                            
                        ax.invert_xaxis()

                        ax.set_ylabel('Intensity')
                        if celsius:
                            T_text = str(T - 273.15) + u' \u00B0C, '
                        else:
                            T_text = str(T) + ' K,'
                        if mbar:
                            p_text = str(int(round(p/100))) + ' mbar'
                        else:
                            p_text = str(p) + ' Pa'
                            
                        ax.set_title(T_text + p_text)
                        axs[-1][j].set_xlabel('Binding energy (eV)')

            if save_plot:
                y_suptitle = 1 - 0.2 / height
                plt.suptitle(element + ' spectra',size=20,y=y_suptitle)
                plt.savefig(filename + '_' + element + '.pdf')
        return
        
    def plot_coverages(self,filename='coverages.pdf',cutoff=0.005):
        #Plot coverage of each species if greater than cutoff
        plt.clf()
        axs, height = self.init_subplots(len(self.Ts),len(self.ps))
        for i,T in enumerate(self.Ts):
            for j,p in enumerate(self.ps):
                if (T,p) in self.Tps:
                    save_plot = True
                    plotted_species = []
                    plotted_coverages = []
                    for ads in self.species_coverages[(T,p)]:
                        cov = self.species_coverages[(T,p)][ads]
                        if cov > cutoff:
                            plotted_species.append(ads.split('_')[0])
                            plotted_coverages.append(cov)
                    axs[i][j].bar(plotted_species,plotted_coverages)
        plt.savefig(filename)
        
        return

