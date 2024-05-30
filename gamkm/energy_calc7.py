import numpy as np
from .thermochemistry import HarmonicThermo,IdealGasThermo
from ase.build import molecule
import pprint
import copy
import time

class Gibbs(object):
    vibs_etc = None
    rxns = None
    def __init__(self,T,energies=None,vibs_etc=None,rxns=None,slopes=None):
        kb = 8.617332478e-5
        self.T = T
        #-----------Mikael------------
        #self.pressures = pressures #dict
        if energies == None:
            self.set_default_energies()
        else:
            self.energies = energies
            self.adsorbate_names = []
            self.transition_state_names = []
            self.gas_names = []
            self.site_names = []
            for species in energies:
                if species[-2:] == '_g':
                    self.gas_names.append(species)
                elif '-' in species:
                    self.transition_state_names.append(species)

                else:
                    self.adsorbate_names.append(species)
                    site = species.split('_')[-1]
                    if site not in self.site_names:
                        self.site_names.append(site)

            self.species_names = self.adsorbate_names + self.transition_state_names + self.gas_names
            self.surface_species = self.adsorbate_names + self.transition_state_names
        #slope_dict = slopes #assume dict
        self.set_slopes(slopes)
        #print(self.slopes)
        
        self.invcm_in_eV = 1.23981e-4
        if Gibbs.vibs_etc == None:
            Gibbs.vibs_etc = vibs_etc
        if Gibbs.vibs_etc == None:
            self.set_default_vibs_etc()
        else:
            Gibbs.vibs_etc = vibs_etc
        if Gibbs.rxns == None:
            Gibbs.rxns = rxns
        if Gibbs.rxns == None:
            self.set_default_rxns()

        #Make rxn sides with only surface species
        for r in rxns:
            sides = ['l','r']
            if 'm' in rxns[r]:
                sides.append('m')
            for side in sides:
                rxns[r][side + 'ss'] = []
                for species in rxns[r][side]:
                    if species in self.surface_species:
                        rxns[r][side + 'ss'].append(species)
        
    def eval_thermo(self,specie_list,quantity):
        value = 0
        for specie in specie_list:
            value += self.species_and_sites[specie][quantity]
        return value

    def set_slopes(self,slopes):
        #slopes dict
        if slopes is None:
            slopes = {}
        else:
            #--------
            for species in slopes:
                if '-' not in species:
                    ads = species
                    if not isinstance(slopes[ads],dict):
                        pprint.pprint(slopes)
                        raise Exception()
                    for co_ads in slopes[ads]:
                        #print(ads,co_ads)
                        if co_ads in slopes:
                            if ads in slopes[co_ads]:
                                if slopes[ads][co_ads] != slopes[co_ads][ads]:
                                    raise Exception('Slope discrepancy: ' + co_ads + '&' + ads + ' != ' + ads + '&' + co_ads)
                            else:
                                raise Exception('Slope for ' + co_ads + '&' + ads + ' does not exist')
                        else:
                            raise Exception('No slopes for ' + co_ads)
            #--------- 
        self.slopes = np.zeros((len(self.surface_species),len(self.adsorbate_names))) #array
        for i,ss in enumerate(self.surface_species):
            if ss in slopes:
                for j,co_ads in enumerate(self.adsorbate_names):
                    if co_ads in slopes[ss]:
                        self.slopes[i][j] = slopes[ss][co_ads]
        return
                
    def calculate_gibbs(self,standard_pressure=1e5):

        thermos = copy.deepcopy(Gibbs.vibs_etc)
        for species in self.species_names:
            thermos[species]['E'] = self.energies[species]

        for specie in thermos:
            if specie in self.gas_names:
                atoms = self.vibs_etc[specie]['Atoms']
                thermos[specie]['M'] = sum(atoms.get_masses())
                thermo = IdealGasThermo(vib_energies=thermos[specie]['freqs'],
                                        potentialenergy=thermos[specie]['E'],
                                        geometry=self.vibs_etc[specie]['geometry'],
                                        symmetrynumber=self.vibs_etc[specie]['symmetry'],
                                        spin=0,atoms=atoms)
                thermos[specie]['S'] = thermo.get_entropy(self.T,pressure=standard_pressure,verbose=False) #ref pressure
                #thermos[specie]['S'] = thermo.get_entropy(self.T,pressure=self.pressures[specie],verbose=False)
                thermos[specie]['H'] = thermo.get_enthalpy(self.T,verbose=False)
                #thermos[specie]['G_preal'] = thermos[specie]['H'] - self.T * thermos[specie]['S_preal']
                #thermos[specie]
            else:
                thermo = HarmonicThermo(vib_energies=thermos[specie]['freqs'],
                                        potentialenergy=thermos[specie]['E'])
                thermos[specie]['S'] = thermo.get_entropy(self.T,verbose=False)
                thermos[specie]['H'] = thermo.get_internal_energy(self.T,verbose=False)
            thermos[specie]['G'] = thermos[specie]['H'] - self.T * thermos[specie]['S']
            thermos[species]['Gc'] = thermos[specie]['G'] - thermos[specie]['E'] #E -> correction
            thermos[specie]['Ec'] = 0

        for site in self.site_names:
            thermos['*_' + site] = {'G': 0.0, 'Ec': 0.0, 'E': 0.0}
        self.species_and_sites = thermos

        rxns = copy.deepcopy(Gibbs.rxns)
        
        #Gibbs free energy arrays for matrix multiplication later
        self.G0l = np.zeros(len(self.rxns))
        self.G0r = np.zeros(len(self.rxns))
        self.G0m = np.zeros(len(self.rxns))
        
        #self.E0l = np.zeros(len(self.rxns))
        #self.E0r = np.zeros(len(self.rxns))
        #self.E0m = np.zeros(len(self.rxns))
                
        for i,r in enumerate(rxns):
            self.set_minima_barrier_energies(rxns[r],etype='G')
            rxns[r]['dG0'] = rxns[r]['dG']
            rxns[r]['Ga0_f'] = rxns[r]['Ga_f']
            rxns[r]['Ga0_rev'] = rxns[r]['Ga_rev']
            """
            lG = self.eval_thermo(rxns[r]['l'],'G')
            self.G0l[i] = lG
            rG = self.eval_thermo(rxns[r]['r'],'G')
            self.G0r[i] = rG

            #rxns[r]['dE0'] = rE - lE

            rxns[r]['dG'] = rG - lG
            rxns[r]['dG0'] = rxns[r]['dG'].copy()
            if 'm' in rxns[r]:
                mG = self.eval_thermo(rxns[r]['m'],'G')
                self.G0m[i] = mG

                Ga_f = mG - lG
                Ga_rev = mG - rG

                if Ga_f > 0 and Ga_rev > 0:
                    rxns[r]['Ga_f'] = Ga_f
                    rxns[r]['Ga_rev'] = Ga_rev
                    no_TS = False
                else:
                    no_TS = True
            if 'm' not in rxns[r] or no_TS:
                rxns[r]['Ga_f'] = max(0.0, rxns[r]['dG0'])
                rxns[r]['Ga_rev'] = max(0.0, -rxns[r]['dG0'])
                
            rxns[r]['Ga0_f'] = rxns[r]['Ga_f']
            rxns[r]['Ga0_rev'] = rxns[r]['Ga_rev']
            """
        self.rxns = rxns
        
        #Create matrices to optimize slope-corrected energy updates in
        #recalculate_coverage_dependent_energy()
        
        L = np.zeros([len(self.rxns),len(self.surface_species)])
        M = np.zeros([len(self.rxns),len(self.surface_species)])
        R = np.zeros([len(self.rxns),len(self.surface_species)])
        
        #Gibbs free energy arrays
        self.G0l = np.zeros(len(self.rxns))
        self.G0r = np.zeros(len(self.rxns))
        self.G0m = np.zeros(len(self.rxns))
        
        for i,r in enumerate(rxns):
            self.G0l[i] = self.eval_thermo(rxns[r]['l'],'G')
            self.G0r[i] = self.eval_thermo(rxns[r]['r'],'G')
            if 'm' in rxns[r]:
                self.G0m[i] = self.eval_thermo(rxns[r]['m'],'G')
            #-------------Build matrices---------------
            for side,matrix in zip(['lss','rss','mss'],[L,R,M]):
                if side in self.rxns[r]: #'m' may not exist
                    for species in self.rxns[r][side]: #ony surface species
                        j = self.surface_species.index(species)
                        matrix[i][j] += 1.
            #-------------------------------------------
        self.LS = np.matmul(L,self.slopes)
        self.RS = np.matmul(R,self.slopes)
        self.MS = np.matmul(M,self.slopes)
 
        return
        
    def set_minima_barrier_energies(self,rxn,etype='G'):
        lG = self.eval_thermo(rxn['l'],etype)
        rG = self.eval_thermo(rxn['r'],etype)
        dQ = 'd' + etype
        if dQ in rxn:
            #only time overwriting should be withe same values
            if abs(rxn[dQ] - (rG - lG)) > 1e8:
                print('WARNING! Inconsistency in energies')
                print(dQ,rxn)
            
        rxn[dQ] = rG - lG    
        
        if 'm' in rxn:
            mG = self.eval_thermo(rxn['m'],etype)
            Ga_f = mG - lG
            Ga_rev = mG - rG
            if Ga_f > 0 and Ga_rev > 0:
                rxn[etype + 'a_f'] = Ga_f
                rxn[etype + 'a_rev'] = Ga_rev
                no_TS = False
            else:
                no_TS = True
        if 'm' not in rxn or no_TS:
            rxn[etype + 'a_f'] = max(0.0, rxn['d' + etype])
            rxn[etype + 'a_rev'] = max(0.0, -rxn['d' + etype])
        return
    
    def post_run_energy_update(self,theta_array):
        if theta_array is None:
            Ecs = [0] * len(self.surface_species)
        else:
            Ecs = np.matmul(self.slopes,theta_array)
        for i, ss in enumerate(self.surface_species):
            ssd = self.species_and_sites[ss]
            ssd['Ec'] = Ecs[i]
            ssd['E0'] = self.species_and_sites[ss]['E']
            ssd['E'] += Ecs[i]
            ssd['G0'] = self.species_and_sites[ss]['G']
            ssd['G'] += Ecs[i]
        for r in self.rxns:
            self.set_minima_barrier_energies(self.rxns[r],etype='G')
            self.set_minima_barrier_energies(self.rxns[r],etype='E')
    
    def recalculate_coverage_dependent_energy3(self,theta_array):
        t0 = time.time()
        Ecl = np.matmul(self.LS,theta_array)
        Ecr = np.matmul(self.RS,theta_array)
        Ecm = np.matmul(self.MS,theta_array)
        
        dG = self.G0r + Ecr - self.G0l - Ecl
        Ga_f_TS = self.G0m + Ecm - self.G0l - Ecl
        Ga_rev_TS = self.G0m + Ecm - self.G0r - Ecr
        t1 = time.time()
        dt1 = t1-t0
        
        for i,r in enumerate(self.rxns):
            self.rxns[r]['dG'] = dG[i]
            if 'm' in self.rxns[r]:
                if Ga_f_TS[i] > 0 and Ga_rev_TS[i] > 0:
                    self.rxns[r]['Ga_f'] = Ga_f_TS[i]
                    self.rxns[r]['Ga_rev'] = Ga_rev_TS[i]
                    no_TS = False
                else:
                    no_TS = True
            if 'm' not in self.rxns[r] or no_TS:
                self.rxns[r]['Ga_f'] = max(0.0, self.rxns[r]['dG'])
                self.rxns[r]['Ga_rev'] = max(0.0, -self.rxns[r]['dG'])
        t2 = time.time()
        dt2 = t2-t1
        
        return dt1,dt2

    def get_adsorbate_dict(self):
        adsorbates = {}
        for i,ads in enumerate(self.adsorbate_names):
            adsorbates[i] = ads
            adsorbates[ads] = i
        return adsorbates        
        
    def set_default_energies(self):
        self.adsorbate_names = ('H_h','CH2OH_s','CH2O_s','CH2_s','CH3O_s','CH3_s','CHOH_s',
                                'CHO_s','CH_s','COH_s','CO_s','C_s','H2O_s','OH_s')
        self.transition_state_names = ('C-H_s','C-OH_s','CH-OH_s','CH2-H_s','CH3-H_s','CH3O-H_s','CO-H_s',
                         'H-CH2OH_s','H-CH2O_s','H-CHOH_s','H-CHO_s','H-CH_s','H-COH_s',
                         'H-CO_s','H-OH_s','HCO-H_s')
        self.gas_names = ('CH3OH_g','CH4_g','CO_g','H2O_g','H2_g')
        self.species_names = self.adsorbate_names + self.transition_state_names + self.gas_names      
        
        #original formation energies from DFT
        rE_orig = [-0.36, -1.41, -0.97, -1.62, -1.57, -2.16, -1.07, -1.08, -1.72, -1.33, -1.7,
                   -0.9, -0.34, 0.24, -0.26, -0.12, -0.32, -1.5, -1.89, -0.8, -0.12, -0.99,
                   -0.74, -0.7, -0.82, -1.4, -0.52, -0.67, 0.81, -0.49, -1.55, -2.83, 0.0, 0.0, 0.0]
        energies = {}
        for i, species in enumerate(self.species_names):
            energies[species] = rE_orig[i]   
        self.energies = energies
        return
        
    def set_default_vibs_etc(self):
        
        vibs = [0]*35
        vibs[0] = np.array([463.0, 716.0, 982.0])
        vibs[1] = np.array([65.0, 116.0, 214.0, 282.0, 481.0, 517.0, 660.0, 846.0, 1076.0,
                            1151.0, 1315.0, 1421.0, 3006.0, 3093.0, 3621.0])
        vibs[2] = np.array([143.0, 162.0, 291.0, 324.0, 466.0, 640.0, 898.0, 1103.0, 1153.0,
                            1440.0, 2966.0, 3044.0])
        vibs[3] = np.array([12.0, 306.0, 382.0, 468.0, 663.0, 790.0, 1356.0, 2737.0, 3004.0])
        vibs[4] = np.array([34.0, 73.0, 158.0, 273.0, 317.0, 360.0, 971.0, 1137.0, 1140.0,
                            1441.0, 1462.0, 1463.0, 2982.0, 3053.0, 3056.0])
        vibs[5] = np.array([12.0, 114.0, 168.0, 622.0, 686.0, 703.0, 1382.0, 1417.0, 1576.0,
                            3026.0, 3093.0, 3099.0])
        vibs[6] = np.array([12.0, 144.0, 202.0, 314.0, 367.0, 569.0, 761.0, 1044.0, 1203.0,
                            1426.0, 2983.0, 3650.0])
        vibs[7] = np.array([12.0, 72.0, 206.0, 336.0, 452.0, 660.0, 1238.0, 1262.0, 2832.0])
        vibs[8] = np.array([413.0, 437.0, 487.0, 710.0, 735.0, 3045.0])
        vibs[9] = np.array([109, 157, 178, 373, 429, 448, 1093, 1223, 3652])
        vibs[10] = np.array([60.0, 231.0, 256.0, 303.0, 470.0, 1747.0])
        vibs[11] = np.array([500, 502, 553])
        vibs[12] = np.array([12.0, 182.0, 250.0, 290.0, 486.0, 726.0, 1584.0, 3642.0, 3752.0])
        vibs[13] = np.array([12.0, 341.0, 396.0, 670.0, 718.0, 3682.0])
        vibs[14] = np.array([385, 461, 497, 535, 1816])
        vibs[15] = np.array([154, 353, 417, 446, 513, 543, 809, 3707])
        vibs[16] = np.array([194.0, 314.0, 411.0, 461.0, 643.0, 671.0, 723.0, 755.0, 849.0,
                             3162.0, 3676.0])
        vibs[17] = np.array([121, 332, 345, 534, 637, 853, 983, 1377, 1928, 2977, 3046])
        vibs[18] = np.array([12.0, 104, 118, 122, 408, 760, 829, 1206, 1414, 1445, 1814,
                             2996, 3087, 3125])
        vibs[19] = np.array([12.0, 8.0, 111.0, 197.0, 222.0, 319.0, 378.0, 961.0, 1116.0,
                             1134.0, 1237.0, 1403.0, 1463.0, 1467.0, 2965.0, 3029.0, 3040.0])
        vibs[20] = np.array([141, 226, 348, 372, 419, 545, 1260, 1458])
        vibs[21] = np.array([12.0, 103.0, 131.0, 198.0, 340.0, 406.0, 523.0, 761.0, 1008.0,
                             1084.0, 1207.0, 1344.0, 1447.0, 1796.0, 3007.0, 3096.0, 3654.0])
        vibs[22] = np.array([12.0, 73.0, 172.0, 235.0, 325.0, 412.0, 664.0, 1032.0, 1188.0,
                             1382.0, 1519.0, 1658.0, 2943.0, 3025.0])
        vibs[23] = np.array([53.0, 100.0, 187.0, 247.0, 517.0, 537.0, 662.0, 888.0, 1044.0,
                             1161.0, 1241.0, 1384.0, 3069.0, 3443.0])
        vibs[24] = np.array([122, 175, 257, 278, 371, 895, 1101, 1193, 1380, 2698, 2736])
        vibs[25] = np.array([270.0, 369.0, 382.0, 566.0, 681.0, 955.0, 1998.0, 3025.0])
        vibs[26] = np.array([84, 101, 249, 361, 474, 655, 1003, 1184, 1332, 2824, 3663])
        vibs[27] = np.array([71.0, 197.0, 315.0, 376.0, 537.0, 1047.0, 1445.0, 2114.0])
        vibs[28] = np.array([12.0, 273.0, 312.0, 399.0, 695.0, 831.0, 1111.0, 3615.0])
        vibs[29] = np.array([137.0, 231.0, 290.0, 327.0, 493.0, 594.0, 680.0, 911.0, 1105.0,
                             1166.0, 2929.0])
        vibs[30] = np.array([3739, 3077, 3013, 2975, 1496, 1487, 1459, 1365, 1160, 1066, 1028,
                             397])
        vibs[31] = np.array([2917.0, 1534.0, 1534.0, 3019.0, 3019.0, 3019.0, 1306.0, 1306.0,
                             1306.0])
        vibs[32] = np.array([2170.0])
        vibs[33] = np.array([3657.0, 1595.0, 3756.0])
        vibs[34] = np.array([4401.0])

        #convert vibs to eV
        vibs_eV = [i*self.invcm_in_eV for i in vibs]
        
        Gibbs.vibs_etc = {'CH3OH_g':{'Atoms':molecule('CH3OH'),'geometry':'nonlinear','symmetry':1},
                         'CH4_g':{'Atoms':molecule('CH4'),'geometry':'nonlinear','symmetry':12},
                         'CO_g':{'Atoms':molecule('CO'),'geometry':'linear','symmetry':1},
                         'H2O_g':{'Atoms':molecule('H2O'),'geometry':'nonlinear','symmetry':2},
                         'H2_g':{'Atoms':molecule('H2'),'geometry':'linear','symmetry':2}}
        for name,vib in zip(self.name_list,vibs_eV):
            if name in self.vibs_etc:
                Gibbs.vibs_etc[name]['freqs'] = vib
            else:
                Gibbs.vibs_etc[name] = {'freqs':vib}
                
        return
        
    def set_default_rxns(self):
        rxns = {'H2to2H':{'l':['H2_g','*_h','*_h'],'r':['H_h','H_h']},
                'COads':{'l':['CO_g','*_s'],'r':['CO_s']},
                'CO,HtoCHO':{'l':['CO_s','H_h'],'m':['H-CO_s','*_h'],'r':['CHO_s','*_h']},
                'CO,HtoCOH':{'l':['CO_s','H_h'],'m':['CO-H_s','*_h'],'r':['COH_s','*_h']},
                'COHtoC,OH':{'l':['COH_s','*_s'],'m':['C-OH_s','*_s'],'r':['C_s','OH_s']},
                'C,HtoCH':{'l':['C_s','H_h'],'m':['C-H_s','*_h'],'r':['CH_s','*_h']},
                'CH,HtoCH2':{'l':['CH_s','H_h'],'m':['H-CH_s','*_h'],'r':['CH2_s','*_h']},
                'CH2,HtoCH3':{'l':['CH2_s','H_h'],'m':['CH2-H_s','*_h'],'r':['CH3_s','*_h']},
                'CH3,HtoCH4':{'l':['CH3_s','H_h'],'m':['CH3-H_s','*_h'],'r':['CH4_g','*_h','*_s']},
                'COH,HtoCHOH':{'l':['COH_s','H_h'],'m':['H-COH_s','*_h'],'r':['CHOH_s','*_h']},
                'CHO,HtoCHOH':{'l':['CHO_s','H_h'],'m':['HCO-H_s','*_h'],'r':['CHOH_s','*_h']},
                'CHOHtoCH,OH':{'l':['CHOH_s','*_s'],'m':['CH-OH_s','*_s'],'r':['CH_s','OH_s']},
                'CHOH,HtoCH2OH':{'l':['CHOH_s','H_h'],'m':['H-CHOH_s','*_h'],'r':['CH2OH_s','*_h']},
                'CH2OH,HtoCH3OH':{'l':['CH2OH_s','H_h'],'m':['H-CH2OH_s','*_h'],'r':['CH3OH_g','*_h','*_s']},
                'CHO,HtoCH2O':{'l':['CHO_s','H_h'],'m':['H-CHO_s','*_h'],'r':['CH2O_s','*_h']},
                'CH2O,HtoCH3O':{'l':['CH2O_s','H_h'],'m':['H-CH2O_s','*_h'],'r':['CH3O_s','*_h']},
                'CH3O,HtoCH3OH':{'l':['CH3O_s','H_h'],'m':['CH3O-H_s','*_h'],'r':['CH3OH_g','*_h','*_s']},
                'OH,HtoH2O':{'l':['OH_s','H_h'],'m':['H-OH_s','*_h'],'r':['H2O_s','*_h']},
                'H2Odes':{'l':['H2O_s'],'r':['H2O_g','*_s']},
               }
        Gibbs.rxns = rxns
        return
