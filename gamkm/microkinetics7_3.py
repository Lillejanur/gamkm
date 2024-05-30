import sys
import numpy as np
import math
from scipy import integrate
from scipy.integrate import solve_ivp #requires 1.4 or higher
np.float128()
import copy
import time
from .energy_calc7 import Gibbs
import pprint
#from ase.thermochemistry import HarmonicThermo
class MicroKinetic(object):
    '''MicroKinetics object.

        The MicroKinetics class for solving microkinetic models.

        Parameters:

        species: list of str
        Labels for species

        T: float
        Reaction temperature in Kelvin

        p: float
        Total pressure in Pa
        
        fractional_pressures: dict
        Fraction of total pressure. Keys are gas phase labels, e.g. 'CO_g'
        
        eps_p: float
        Default pressure (Pa) for product gases, since 0 would not work mathematically

        ads_Area: float
        Area of active site in m**2

        theta0: dict of floats
        Initial coverage of surface species
        
        slopes: dict
        Adsorbate-adsorbate interaction slopes. Keys are surface species. Values are either the
        parameters or subdicts with co-adsorbing adsorbates as keys and parameters as values
        
        energies: dict
        Energy of gas, adsorbate, transition-state species. Note that it is not barriers, but
        energies of transition states relative to gas phase references.
        
        vibs_etc: dict
        Vibrations for all species and necessary parameters to calculate gas partition functions
        
        rxns: dict
        Elementary reactions. Contains lists of species left, middle and right side of the reaction
        (rxns[rxn]['l'], etc.)
        
        sites: dict
        Surface sites as keys and max coverage as values.
        
        adsorption_model: str
        Adsorption model can be 'pure_gibbs', which means that gas rate constants have prefactor
        kBT/h like surface species, or 'HK' for Hertz-Knudsen, which assumes a 2D transition state
        for the adsorption process.

        t: int
        Simulation time in seconds

        steps: int
        Number of time steps

        Xrc: bool
        Switch on DRC analysis

        ode_solver: str
        'standard' uses scipy ode with bdf
        'super_stiff' uses ode_ivp with radau integrator

        algorithm: str
        Used only if ode_solver = 'super_stiff'
        'BDF' backwards differentiation (standard)
        'Radau' (slow but super stable, use if BDF fails)
        'RK45' (Runge-Kutte 45, sometimes works ok)
        
        tols: tuple
        Tolerances, relative and absolute. See scipy.integrate.solve_ivp
        
        scale_factor: float
        Factor to scale the variables (coverages) in the solver. I saw on stackoverflow
        that solve_ivp works best for variables in the range 0.1 to 100, so this might
        improve convergence.
        '''
    vibs_etc = None
    rxns = None
        
    def __init__(self,T,p,fractional_pressures=None,eps_p=1e-20,
                 ads_Area=None, #M=None,
                 theta0=None,slopes=None,energies=None,vibs_etc=None,
                 rxns=None,sites=None,adsorption_model='pure_gibbs',
                 t=1,steps=1000,
                 Xrc = False,ode_solver='standard', algorithm='BDF', tols = (1e-12,1e-14),
                 scale_factor=1.0,
                 #plotted_rxns={}, rxn_plot_params={}
                ):
                 
        self.run_times = {'theta_to_dict': 0,
                          'get_rate_constants': 0,
                          'get_rates': 0,
                          'recalc1': 0,
                          'recalc2': 0,
                          'rest': 0}
        self.T = T
        self.p = p
        self.standard_pressure = 1e5
        self.fractional_pressures = fractional_pressures
        self.eps_p = eps_p
        
        self.ads_Area = ads_Area
        #self.M = M
        
        self.theta0 = theta0
        self.aai = bool(slopes)
        self.slopes = slopes                           
                            
        self.energies = energies
        if energies == None:
            print('Warning! energies == None')
        
        if MicroKinetic.vibs_etc == None:
            MicroKinetic.vibs_etc = copy.deepcopy(vibs_etc)
            
        if MicroKinetic.rxns == None:
            for r in rxns:
                sites_l = {}
                for species in rxns[r]['l']:
                    site = species.split('_')[-1]
                    #Sites should be balanced, except gases
                    if site != 'g': 
                        if site in sites_l:
                            sites_l[site] += 1
                        else:
                            sites_l[site] = 1
                sites_r = {}
                for species in rxns[r]['r']:
                    site = species.split('_')[-1]
                    if site != 'g':
                        if site in sites_r:
                            sites_r[site] += 1
                        else:
                            sites_r[site] = 1
                if sites_l != sites_r:
                    raise Exception('Site imbalance: ' + str(r))  
            MicroKinetic.rxns = copy.deepcopy(rxns)
        self.sites = sites
        self.adsorption_model = adsorption_model
        
        self.t = t
        self.steps = steps
        self.tspan = np.linspace(0, t, self.steps)
     
        self.Xrc = Xrc                
        self.ode_solver = ode_solver
        self.algorithm = algorithm
        self.tols = tols
        self.alpha = scale_factor #Numerical scaling in the solver. Someone said values close to 0.1-100 is good.
        #self.plotted_rxns = plotted_rxns
        #self.rxn_plot_params = {'width':0.5,'color':'k'}
        #for param in rxn_plot_params:
        #    self.rxn_plot_params[param] = rxn_plot_params[param]
        
        J2eV = 6.24e18
        self.Na = 6.023e23
        self.h = 6.626e-34 * J2eV
        self.kb = 1.38e-23 * J2eV
        self.kbT = self.kb*self.T
        self.k = 1.38064852e-23 # J/K
        self.RT = 83.144598*self.T # (cm3*bar/mol*K) *K

        self.beta = (1/self.kbT)
        
        self.prefactors = {}
        
        #self.verbose = verbose

        
    def set_vibs_etc(self,vibs_etc):
        Gibbs.set_vibs_etc(vibs_etc)
        return
        
    def get_ads_prefactor(self,M,p):
        if self.adsorption_model == 'pure_gibbs':
            prefactor = p * self.kbT/self.h
        elif self.adsorption_model == 'HK':
            prefactor = p * self.ads_Area/(np.sqrt(2*np.pi*((M*0.001/self.Na)*self.T*self.k)))
        else:
            raise Exception('adsorption_model unknown. Please use HK or pure_gibbs')
        return prefactor
        
    def get_rate_constants(self, rxns):
        for r in rxns:
            rxns[r]['kf'] = rxns[r]['kf_prefactor']*math.exp(-rxns[r]['Ga_f']*self.beta)
            rxns[r]['kr'] = rxns[r]['kr_prefactor']*math.exp(-rxns[r]['Ga_rev']*self.beta)
            #rxns[r]['Keq'] = rxns[r]['kf'] / rxns[r]['kr']
        return
        
    def theta_prod(self,species_list):
        prod = 1
        for species in species_list:
            if species[-1] != 'g':
                prod *= self.thetas[species] #/ self.alpha
        return prod
    
    
    def get_rates(self, rxns):
        #thetas is assumed to be a dict  {'CO_s':0.2,...} It must include empty sites
        #thetas['H2_g'] can be defined as P_H2, or, if the pressure is in the free energy,
        #thetas['H2_g'] can be 1
            
        for r in rxns:
            rate_f = rxns[r]['kf']
            for species in rxns[r]['l']:
                if species[-1] != 'g':
                    rate_f *= self.thetas[species]
            rate_rev = rxns[r]['kr']
            for species in rxns[r]['r']:
                if species[-1] != 'g':
                    rate_rev *= self.thetas[species]
                    
            rxns[r]['rate'] = rate_f - rate_rev
                    
            #rxns[r]['rate'] = rxns[r]['kf'] * self.theta_prod(rxns[r]['l']) - rxns[r]['kr'] * self.theta_prod(rxns[r]['r'])
            
        return
   
    def theta_to_dict(self,theta_list,adsorbate_names):
        #Assume that we have a dict of sites, with max coverage, e.g. {'h':1.0,'s':1.0}
        for site in self.sites:
            self.thetas['tot_' + site] = 0.
        for ads in adsorbate_names:
            theta = theta_list[self.adsorbate_dict[ads]]
            self.thetas[ads] = theta
            site = ads.split('_')[-1]
            if site != 'g':
                self.thetas['tot_' + site] += theta
        for site in self.sites:
            self.thetas['*_' + site] = self.sites[site] - self.thetas['tot_' + site]

        return 
    
    def get_model_mikael(self,t,theta_list,gibbs):
        #gibbs object

        theta_scaled = theta_list / self.alpha
        if self.aai:
            dt1,dt2 = gibbs.recalculate_coverage_dependent_energy3(theta_scaled)
            self.run_times['recalc1'] += dt1
            self.run_times['recalc2'] += dt2            
        timem = time.time()
        self.theta_to_dict(theta_scaled,gibbs.adsorbate_names) #updates self.thetas
        
        #Recalculate Gibbs free energy and rate constants
        time0 = time.time()
        self.run_times['theta_to_dict'] += time0 - timem
        time1 = time.time()

        if self.aai:
            self.get_rate_constants(gibbs.rxns)
        time2 = time.time()
        self.run_times['get_rate_constants'] += time2 - time1
        
        #Calculate rates
        self.get_rates(gibbs.rxns)
        time3 = time.time()
        self.run_times['get_rates'] += time3 - time2
        
        dthetadt = np.zeros(len(gibbs.adsorbate_names)) 
        for r in gibbs.rxns:
            for specie in gibbs.rxns[r]['lss']:
                dthetadt[self.adsorbate_dict[specie]] -= gibbs.rxns[r]['rate']
            for specie in gibbs.rxns[r]['rss']:
                dthetadt[self.adsorbate_dict[specie]] += gibbs.rxns[r]['rate']
                   
        dthetadt *= self.alpha
        time4 = time.time()
        self.run_times['rest'] += time4 - time3
        return dthetadt 
        
    def stiff_ode_solver_mikael(self,gibbs):
        report = ""
        t_start = min(self.tspan)
        t_final = max(self.tspan)
        n_steps = len(self.tspan)
        delta_t = (t_final - t_start) / self.steps
        t_eval = np.arange(t_start, t_final, delta_t)    
        if False:
            r = solve_ivp(self.get_model_mikael, (t_start, t_final),
                          self.theta0*self.alpha, method=self.algorithm, rtol=self.tols[0], atol=self.tols[1],
                          t_eval=t_eval, args=(gibbs,))
            report += "solve_ivp passed. Remove True in stiff_ode_solver\n"
        else: 
            try:
                t18 = time.time()
                r = solve_ivp(self.get_model_mikael, (t_start, t_final),
                        self.theta0*self.alpha, method=self.algorithm, rtol=self.tols[0], atol=self.tols[1],
                        t_eval=t_eval, args=(gibbs,))
                #report += "solve_ivp_time: " + str(round(time.time() - t18,3)) + " s\n"
                if r.message == "The solver successfully reached the end of the integration interval.":
                    message = "ODE successful"
                else:
                    message = r.message
                report += message + "\n"
                if r.status == 0:
                    times = r.t
                    values = r.y.T / self.alpha
                else:
                    times = None
                    values = None
            except Exception as e:
                report += "Exception occurred in ODE solver (solve_ivp)\n"
                report += e + "\n"
                times = None
                values = None
        
        return times, values, report

    def get_thermodynamics(self,print_report=True):
        report = ''
        if self.aai:
            report += 'WARNING! Thermodynamics only valid for zero coverage'
        gibbs = Gibbs(self.T,self.energies,MicroKinetic.vibs_etc,MicroKinetic.rxns,slopes=self.slopes)
        gibbs.calculate_gibbs(standard_pressure=self.standard_pressure)
        gibbs.post_run_energy_update(theta_array=None)
        if print_report:
            print(report)
        return gibbs.rxns, gibbs.species_and_sites, report

    def solve_microkinetic_model(self,print_report=True,report_runtimes=False):
        #calculate PURE barriers using the Gibbs calculator
        
        report = ''
        #report += '#####################################\n'
        report += 'Simulation at ' + str(self.T) + ' K, ' + str(self.p) + ' Pa\n'

        gibbs = Gibbs(self.T,self.energies,MicroKinetic.vibs_etc,MicroKinetic.rxns,slopes=self.slopes)
            
        if self.sites == None:
            self.sites = {}
            for site in gibbs.site_names:
                self.sites[site] = 1
                
        self.adsorbate_dict = {}
        for i,ads in enumerate(gibbs.adsorbate_names):
            self.adsorbate_dict[i] = ads
            self.adsorbate_dict[ads] = i
        
        if self.theta0 is None:
            self.theta0 = np.zeros(len(gibbs.adsorbate_names))
        else:
            theta0 = np.zeros(len(gibbs.adsorbate_names))
            for i in range(len(gibbs.adsorbate_names)):
                theta0[i] = self.theta0[self.adsorbate_dict[i]]
            self.theta0 = theta0
        
        
        self.thetas = {}
        gibbs.calculate_gibbs(standard_pressure=self.standard_pressure)
        
        self.pressures = {}
        for gas in gibbs.gas_names:
            if gas in self.fractional_pressures:
                self.pressures[gas] = self.fractional_pressures[gas] * self.p
            else:
                self.pressures[gas] = self.eps_p
        
        if self.adsorption_model == 'pure_gibbs':
            surf_rxn_keys = list(gibbs.rxns.keys())         
        else:
            surf_rxn_keys = []
            for r in gibbs.rxns:
                gas_found = False
                
                for side, k, other_side, other_k in zip(['l','r'],['kf','kr'],['r','l'],['kr','kf']):
                    for species in gibbs.rxns[r][side]:
                        if species in gibbs.gas_names:
                            if gas_found:
                                raise Exception('Max one gas species per reaction')
                            else:
                                gas_found = True
                                M = gibbs.species_and_sites[species]['M']
                                gibbs.rxns[r][k + '_prefactor'] = self.get_ads_prefactor(M,self.pressures[species])
                                gibbs.rxns[r][other_k + '_prefactor'] = self.get_ads_prefactor(M,self.standard_pressure)
                
                if not gas_found:
                    gibbs.rxns[r]['kf_prefactor'] = self.kbT/self.h
                    gibbs.rxns[r]['kr_prefactor'] = self.kbT/self.h
                    
       
                    
        self.get_rate_constants(gibbs.rxns)
        
        self.times, self.coverage_array, ode_report = self.stiff_ode_solver_mikael(gibbs) #all times and coverages, timelapse
        
        report += ode_report
        
        #pprint.pprint(gibbs.rxns)
        
        #gibbs.post_run_energy_update(self.coverage_array[-1])
        
        coverages = {}
        if self.coverage_array is not None:
            gibbs.post_run_energy_update(self.coverage_array[-1])
            for adsorbate in gibbs.adsorbate_names:
                coverages[adsorbate] = self.coverage_array[-1][self.adsorbate_dict[adsorbate]]

        if report_runtimes:
            report += pprint.pformat(self.run_times) + "\n"
        if print_report:
            print(report)

        return coverages, gibbs.rxns, gibbs.species_and_sites, report
