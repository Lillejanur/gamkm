#!usr/bin/env python3

import os
import pprint

ABS_PATH_PATTERNS = {
    'dft_energy_file': '{}/dft_and_exp_data/dft_energies.py',
    'spectra_dir': '{}/dft_and_exp_data/spectra_march2',
}

MASTER = {}

MASTER['rxns'] = {
    'C,HtoCH': {'l': ['C_s', 'H_h'], 'm': ['C-H_s', '*_h'], 'r': ['CH_s', '*_h']},
    'CH,HtoCH2': {'l': ['CH_s', 'H_h'], 'm': ['H-CH_s', '*_h'], 'r': ['CH2_s', '*_h']},
    'CH2,HtoCH3': {'l': ['CH2_s', 'H_h'], 'm': ['CH2-H_s', '*_h'], 'r': ['CH3_s', '*_h']},
    'CH2O,HtoCH3O': {'l': ['CH2O_s', 'H_h'], 'm': ['H-CH2O_s', '*_h'], 'r': ['CH3O_s', '*_h']},
    'CH2OH,HtoCH3OH': {'l': ['CH2OH_s', 'H_h'], 'm': ['H-CH2OH_s', '*_h'], 'r': ['CH3OH_g', '*_h', '*_s']},
    'CH3,HtoCH4': {'l': ['CH3_s', 'H_h'], 'm': ['CH3-H_s', '*_h'], 'r': ['CH4_g', '*_h', '*_s']},
    'CH3O,HtoCH3OH': {'l': ['CH3O_s', 'H_h'], 'm': ['CH3O-H_s', '*_h'], 'r': ['CH3OH_g', '*_h', '*_s']},
    'CHO,HtoCH2O': {'l': ['CHO_s', 'H_h'], 'm': ['H-CHO_s', '*_h'], 'r': ['CH2O_s', '*_h']},
    'CHO,HtoCHOH': {'l': ['CHO_s', 'H_h'], 'm': ['HCO-H_s', '*_h'], 'r': ['CHOH_s', '*_h']},
    'CHOH,HtoCH2OH': {'l': ['CHOH_s', 'H_h'], 'm': ['H-CHOH_s', '*_h'], 'r': ['CH2OH_s', '*_h']},
    'CHOHtoCH,OH': {'l': ['CHOH_s', '*_s'], 'm': ['CH-OH_s', '*_s'], 'r': ['CH_s', 'OH_s']},
    'CO,HtoCHO': {'l': ['CO_s', 'H_h'], 'm': ['H-CO_s', '*_h'], 'r': ['CHO_s', '*_h']},
    'CO,HtoCOH': {'l': ['CO_s', 'H_h'], 'm': ['CO-H_s', '*_h'], 'r': ['COH_s', '*_h']},
    'COH,HtoCHOH': {'l': ['COH_s', 'H_h'], 'm': ['H-COH_s', '*_h'], 'r': ['CHOH_s', '*_h']},
    'COHtoC,OH': {'l': ['COH_s', '*_s'], 'm': ['C-OH_s', '*_s'], 'r': ['C_s', 'OH_s']},
    'COads': {'l': ['CO_g', '*_s'], 'r': ['CO_s']},
    'H2Odes': {'l': ['H2O_s'], 'r': ['H2O_g', '*_s']},
    'H2to2H': {'l': ['H2_g', '*_h', '*_h'], 'r': ['H_h', 'H_h']},
    'OH,HtoH2O': {'l': ['OH_s', 'H_h'], 'm': ['H-OH_s', '*_h'], 'r': ['H2O_s', '*_h']},
}

MASTER['reference_states'] = ['CO_g','H2O_g','H2_g']

MASTER['energy_bounds'] = {'gas':0.3,'transition_state': 0.3, 'adsorbate':0.3}

MASTER['dft_deviation_coefs'] = {'default': 1.0}

MASTER['spectra_coef'] = 10.0

MASTER['total_pressure'] = 0.15*1e5

MASTER['fractional_pressures'] = {
    'H2_g': 2/3.,
    'CO_g': 1/3.,
}


MASTER['init_covs'] = {'type': 'closest', 'T_cutoff': 50., 'p_cutoff': 1e5}

MASTER['integration_time_length'] = 1e8

MASTER['steps'] = 10000

MASTER['theta0'] = None

MASTER['adsorption_model'] = 'HK'#'HK','pure_gibbs'

MASTER['Xrc'] = False

MASTER['ode_solver'] = 'super_stiff'

MASTER['ode_algorithm'] = 'Radau'

MASTER['ode_tols'] = (1e-6,1e-9)

MASTER['strict_ode_tols'] = (1e-9,1e-12)

MASTER['ode_scale_factor'] = 1e4

MASTER['algorithm_parameters'] = {
    'max_num_iteration':3,
    'initial_population_size':None,
    'population_size':20,
    'mutation_probability': 1.0,
    'max_mutations': 50,
    'individual_mutation_probability': 0.10,
    'mutation_algorithm': 'mutcoinflip',
    'elit_ratio': 0.01,
    'stochastic_pick_ratio': 0.1,
    'stochastic_method': 'unique_uniform',
    'obj_func_cutoff': 100.,
    'crossover_probability': 1.0,
    'parents_portion': 0.6,
    'crossover_type': 'uniform',
    'max_iteration_without_improv':250,
    'all_parents_to_new_gen': False,
}

MASTER['read_continuation_population'] = False

MASTER['logfile'] = 'eval.log'

MASTER['func_exec_report_file'] = 'func_exec_report.txt'

MASTER['save_timelapse'] = True

MASTER['progress_bar'] = False

MASTER['cont_file_writing_frequency'] = 250

MASTER['function_timeout'] = {
    'aai': 6.5,
    'no_aai': 5.0,
}

MASTER['timeout_objective_function'] = 150.

MASTER['delta_logarithm'] = False

MASTER['dft_deviation_penalty_model'] = 'quadratic'

MASTER['max_aai_penalty'] = 0.05

MASTER['plotted_rxns'] = {  #All rates > 0
    'MeOH_CH2O': [
        'COads',
        'H2to2H',
        'CO,HtoCHO',
        'CHO,HtoCH2O',
        'H2to2H',
        'CH2O,HtoCH3O',
        'CH3O,HtoCH3OH',
    ],
    'MeOH_COH': [
        'COads',
        'H2to2H',
        'CO,HtoCOH',
        'COH,HtoCHOH',
        'H2to2H',
        'CHOH,HtoCH2OH',
        'CH2OH,HtoCH3OH'
    ],
    'MeOH_CHO_CHOH': [
        'COads',
        'H2to2H',
        'CO,HtoCHO',
        'CHO,HtoCHOH',
        'H2to2H',
        'CHOH,HtoCH2OH',
        'CH2OH,HtoCH3OH',
    ],
    'CH4_CHO': [
        'COads',
        'H2to2H',
        'CO,HtoCHO',
        'CHO,HtoCHOH',
        'CHOHtoCH,OH',
        'H2to2H',
        'CH,HtoCH2',
       	'CH2,HtoCH3',
        'H2to2H',
        'CH3,HtoCH4',
        'OH,HtoH2O',
        'H2Odes',
    ],
    'CH4_COH': [
        'COads',
        'H2to2H',
        'CO,HtoCOH',
        'COHtoC,OH',
        'C,HtoCH',
        'H2to2H',
        'CH,HtoCH2',
       	'CH2,HtoCH3',
        'H2to2H',
        'CH3,HtoCH4',
        'OH,HtoH2O',
        'H2Odes',
    ],
    'CH4_COH_CHOH':[
        'COads',
        'H2to2H',
        'CO,HtoCOH',
        'COH,HtoCHOH',
        'CHOHtoCH,OH',
        'H2to2H',
        'CH,HtoCH2',
        'CH2,HtoCH3',
        'H2to2H',
        'CH3,HtoCH4',
        'OH,HtoH2O',
        'H2Odes',
    ],
    #COH,H->CHOH negative
    # 'CH4_CHOH_COH': [
    #     'COads',
    #     'H2to2H',
    #     'CO,HtoCHO',
    #     'CHO,HtoCHOH',
    #     'CHOHtoCOH,H',
    #     'COHtoC,OH',
    #     'C,HtoCH',
    #     'H2to2H',
    #     'CH,HtoCH2',
    #     'CH2,HtoCH3',
    #     'H2to2H',
    #     'CH3,HtoCH4',
    #     'OH,HtoH2O',
    #     'H2Odes',
    # ],
}
 
MASTER['landscape_energy_type'] = 'E'
	 
MASTER['rxn_plot_params'] = {
    'reactants': ['CO_g','H2_g'],
    'products': ['CH4_g','CH3OH_g','H2O_g'],
    'Tp':((448.15,15000),(598.15,15000)),
}

MASTER['printed_coverages'] = [
    'CO_s','H_h'
]


def main():
    master = dict(MASTER)
    curdir = os.path.abspath('.')
    master.update({
        key: fname_pattern.format(curdir)
        for key, fname_pattern in ABS_PATH_PATTERNS.items()
    })
    with open('mkm.py', mode='w') as fobj:
        fobj.write('master = \\\n')
        pprint.pprint(master, fobj)


if __name__ == '__main__':
    main()
