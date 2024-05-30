'''

Copyright 2020 Ryan (Mohammad) Solgi

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.

'''

###############################################################################
###############################################################################
###############################################################################

import numpy as np
import math #Mikael
import sys,os,shutil
import time
from func_timeout import func_timeout, FunctionTimedOut
import concurrent.futures as cf #Inserted by Mikael for parallelisation
import matplotlib.pyplot as plt
import time
import pickle
from pprint import pprint
#import matplotlib
#matplotlib.use('TkAgg')

###############################################################################
###############################################################################
###############################################################################

class geneticalgorithm():
    
    '''  Genetic Algorithm (Elitist version) for Python
    
    An implementation of elitist genetic algorithm for solving problems with
    continuous, integers, or mixed variables.
    
    
    
    Implementation and output:
        
        methods:
                run(): implements the genetic algorithm
                
        outputs:
                output_dict:  a dictionary including the best set of variables
            found and the value of the given function associated to it.
            {'variable': , 'function': }
            
                timelapse: a list including the record of the progress of the
                algorithm over iterations

    '''
    #############################################################
    def __init__(self, function, dimension, variable_type='bool',
                 variable_boundaries=None,
                 variable_type_mixed=None,
                 function_timeout=10,
                 timeout_objective_function=None,
                 manual_start_population=None,
                 continuation_file='continuation_population.pckl',
                 cont_file_writing_frequency=0,
                 read_continuation_population=False,
                 logfile=None,
                 func_exec_report_file=None,
                 save_timelapse=False,
                 progress_bar=True,
                 parallel_cores=1,
                 pkl_list=[],
                 verbose=False,
                 algorithm_parameters={}):

        '''
        @param function <Callable> - the given objective function to be minimized
        NOTE: This implementation minimizes the given objective function. 
        (For maximization multiply function by a negative sign: the absolute 
        value of the output would be the actual objective function). The function
        takes two arguments - the input to calculate the objective function, and
        an index to number the individuals.
        
        @param dimension <integer> - the number of decision variables
        
        @param variable_type <string> - 'bool' if all variables are Boolean; 
        'int' if all variables are integer; and 'real' if all variables are
        real value or continuous (for mixed type see @param variable_type_mixed)
        
        @param variable_boundaries <numpy array/None> - Default None; leave it 
        None if variable_type is 'bool'; otherwise provide an array of tuples 
        of length two as boundaries for each variable; 
        the length of the array must be equal dimension. For example, 
        np.array([0,100],[0,200]) determines lower boundary 0 and upper boundary 100
        for first and upper boundary 200 for second variable where dimension is 2.
        
        @param variable_type_mixed <numpy array/None> - Default None; leave it 
        None if all variables have the same type; otherwise this can be used to
        specify the type of each variable separately. For example if the first 
        variable is integer but the second one is real the input is: 
        np.array(['int'],['real']). NOTE: it does not accept 'bool'. If variable
        type is Boolean use 'int' and provide a boundary as [0,1] 
        in variable_boundaries. Also if variable_type_mixed is applied, 
        variable_boundaries has to be defined.
        
        @param function_timeout <float> - if the given function does not provide 
        output before function_timeout (unit is seconds) the algorithm raise error.
        For example, when there is an infinite loop in the given function.
       	#########CHANGE ABOVE####

        @param timeout_objective_function

        @param manual_start_population <numpy array/None> Default None

        @param continuation_file <file> File with population to start manually with

        @param read_continuation_population <bool> Default False. If true, read
        continuation file

	    @param parallel cores <int> Number of cores to parellize over. The program can
        evaluate one individual on one core.

        @param pkl_list <list of strings> Default Empty list. Each element is
        interpreted as a file produced after evaluation of the objective function, 
        and that can be used in the process of rerunning the function for the same
        individual. The string must contain a '*' to be replaced with the index (see
        function above), .e.g., 'data*.pkl' is written as 'data32.pkl' for individual
        number 32, which is continuously removed after each generation. A file will
        be written at the end without '*', e.g., 'data.pkl' for the optimal individual.

        @param algorithm_parameters <dict> - See default keys and values below.
        '''

        default_algorithm_parameters={'max_num_iteration': None,
                                      'population_size':100,
                                      'initial_population_size': None, #Default = population size
                                      'mutation_probability':0.1,
                                      'individual_mutation_probability':1.0,
                                      'max_mutations':None, #Incompatible with mutation_probability
                                      'mutation_algorithm':'mut+mutmidle',
                                      'mutation_factor': None,
                                      'elit_ratio': 0.01,
                                      'stochastic_pick_ratio': None,
                                      'stochastic_method': 'normalized_obj_func',
                                      'obj_func_cutoff': None,
                                      'crossover_probability': 0.5,
                                      'parents_portion': 0.3,
                                      'all_parents_to_new_gen': True,
                                      'crossover_type':'uniform',
                                      'max_iteration_without_improv':None}

        '''
        default algorithm_parameters:
            @ max_num_iteration <int> - stoping criteria of the genetic algorithm (GA)
            @ population_size <int> 
            @ mutation_probability <float in [0,1]>
            @ mutation_algoritm <string> - Choices are 'mut','mutmidle','mutcoinflip'
              and mutfactor. If only one algorithm is chosen, one child per chosen
              parents are created and mutated. If two are chosen, e.g., the default
              'mut+mutmidle', two children are generated per couple of parents where
              every parent gene appears in one of the children.
            @ elit_ratio <float in [0,1]>
            @ stochastic_pick_ratio <float in [0,1]> - Ratio of parents picked
              stochastically, as opposed to strictly by fitness. Default is None, which
              means that all parents except the elites are picked stochastically
            @ stochastic_method <string> - Default is 'normalized_obj_func';
              'uniform', 'unique_uniform' are options
            @ obj_func_cutoff <float> Default is None; if timeout_objective_function
              is set, obj_func_cutoff is the same by default
            @ crossover_probability <float in [0,1]>
            @ parents_portion <float in [0,1]>
            @ parents_to_new_gen <bool> - Default is True
            @ crossover_type <string> - Default is 'uniform'; 'one_point' or 
              'two_point' are other options
            @ max_iteration_without_improv <int> - maximum number of 
              successive iterations without improvement. If None it is ineffective
        
        for more details and examples of implementation please visit:
            https://github.com/rmsolgi/geneticalgorithm
  
        '''
        
        self.__name__=geneticalgorithm
        #############################################################
        # input function
        assert (callable(function)),"function must be callable"     
        
        self.f=function
        #############################################################
        #dimension
        
        self.dim=int(dimension)
        
        #############################################################
        # input variable type
        
        assert(variable_type=='bool' or variable_type=='int' or\
               variable_type=='real'), \
               "\n variable_type must be 'bool', 'int', or 'real'"
       #############################################################
        # input variables' type (MIXED)     

        if variable_type_mixed is None:
            
            if variable_type=='real': 
                self.var_type=np.array([['real']]*self.dim)
            else:
                self.var_type=np.array([['int']]*self.dim)            

 
        else:
            assert (type(variable_type_mixed).__module__=='numpy'),\
            "\n variable_type must be numpy array"  
            assert (len(variable_type_mixed) == self.dim), \
            "\n variable_type must have a length equal dimension."       

            for i in variable_type_mixed:
                assert (i=='real' or i=='int'),\
                "\n variable_type_mixed is either 'int' or 'real' "+\
                "ex:['int','real','real']"+\
                "\n for 'boolean' use 'int' and specify boundary as [0,1]"
                

            self.var_type=variable_type_mixed
        #############################################################
        # input variables' boundaries 

            
        if variable_type!='bool' or type(variable_type_mixed).__module__=='numpy':
                       
            assert (type(variable_boundaries).__module__=='numpy'),\
            "\n variable_boundaries must be numpy array"
        
            assert (len(variable_boundaries)==self.dim),\
            "\n variable_boundaries must have a length equal dimension"        
        
        
            for i in variable_boundaries:
                assert (len(i) == 2), \
                "\n boundary for each variable must be a tuple of length two." 
                assert(i[0]<=i[1]),\
                "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
            self.var_bound=variable_boundaries
        else:
            self.var_bound=np.array([[0,1]]*self.dim)
 
        ############################################################# 
        #Timeout
        self.funtimeout=float(function_timeout)
        if timeout_objective_function is None:
            self.timeout_obj_func=None
        else:
            self.timeout_obj_func=float(timeout_objective_function)

        #############################################################
        #Save timelapse as pdf
        self.save_timelapse=save_timelapse
        
        #############################################################
        #Show progress bar
        self.progress_bar=progress_bar
        
        #############################################################
        #Parallelization
        self.parallel_cores=int(parallel_cores) #Mikaels
        if parallel_cores == 1:
            self.parallelization = False
        else:
            self.parallelization = True  


        #############################################################
        #Files to be removed after each generation
        self.pkl_list = pkl_list
        #############################################################
        #Output dict for variable, function etc.
        self.output_dict = {'bounds': self.var_bound}
        #############################################################
        #Continuation file
        self.cont_file = continuation_file
        self.cont_file_writing_frequency = cont_file_writing_frequency
        #############################################################
        #Logfile to write obj functions and runtimes
        self.logfile=logfile
        #############################################################
        #Function execution text file
        self.func_exec_report_file = func_exec_report_file
        #############################################################
        #Suppress or allow some printouts
        self.verbose = verbose
        #############################################################
        # input algorithm's parameters
        
        self.param=algorithm_parameters

        for key in self.param:
            if key not in default_algorithm_parameters:
                raise Exception(key + ' is not a valid algorithm parameter')

        print('\nGenetic Algorithm Used Default Parameters:')
        for key in default_algorithm_parameters:
            if key not in self.param:
                self.param[key] = default_algorithm_parameters[key]
                print(key + ' = ' + str(default_algorithm_parameters[key]))
        print('\n')
        
        self.pop_s=int(self.param['population_size'])
        
        if self.param['initial_population_size'] != None:
            self.init_pop_s = self.param['initial_population_size']
            if self.init_pop_s < self.pop_s:
                raise Exception('Initial population must be equal or larger than population')
        else:
            self.init_pop_s = self.pop_s
        
        assert (self.param['parents_portion']<=1\
                and self.param['parents_portion']>=0),\
        "parents_portion must be in range [0,1]"

        self.par_to_new_gen = self.param['all_parents_to_new_gen']
        
        self.par_s=int(self.param['parents_portion']*self.pop_s)
        trl=self.pop_s-self.par_s
        if trl % 2 != 0:
            self.par_s+=1
        if self.par_s < 2:
            raise Exception('There must be at least 2 parents.' + \
                            ' Increase parents portion or population size')
                            
               
        self.prob_mut=self.param['mutation_probability']
        
        self.max_muts=self.param['max_mutations']
        if self.max_muts != None:
            if self.max_muts > self.dim:
                raise Exception('Max mutations must be smaller than dimension')
            if self.prob_mut != 1.0:
                print('Warning! If max mutations is set, it is recommended to set ' +
                      'mutation probability to 1.0')
            

        #if self.prob_mut != None and self.max_muts != None:
        #    raise Exception('Either mutation_probability or max_mutations can be set')
        
        self.individual_prob_mut=self.param['individual_mutation_probability']
        
        assert (self.prob_mut<=1 and self.prob_mut>=0), \
        "mutation_probability must be in range [0,1]"
        
        mutalgo=self.param['mutation_algorithm']
        self.mutalgos=mutalgo.split('+')
        mutation_algorithms = ('mut','mutmidle','mutfactor','mutcoinflip')
        
        assert all([algo in mutation_algorithms for algo in self.mutalgos]), \
        "mutation_algorithm must be in"+str(mutation_algorithms)
        
        assert (len(self.mutalgos)>=1 and len(self.mutalgos)<=2), \
        "one or two mutation_algorithms allowed, if two, then separated by '+'"
        
        if self.param['mutation_factor'] == None:
            self.mutation_factor = np.inf
        else:
            self.mutation_factor = mutation_factor
        
        self.prob_cross=self.param['crossover_probability']
        assert (self.prob_cross<=1 and self.prob_cross>=0), \
        "crossover_probability must be in range [0,1]"
        
        assert (self.param['elit_ratio']<=1 and self.param['elit_ratio']>=0),\
        "elit_ratio must be in range [0,1]"                
        
        trl=self.pop_s*self.param['elit_ratio']
        if trl<1 and self.param['elit_ratio']>0:
            self.num_elit=1
        else:
            self.num_elit=int(trl)

        if self.param['stochastic_pick_ratio'] == None: #New
            self.num_stoch=self.par_s-self.num_elit
        else:
            trl=self.pop_s*self.param['stochastic_pick_ratio']
            self.num_stoch=int(trl)

        assert(self.par_s>=self.num_elit+self.num_stoch), \
        "\n number of parents must be greater than number of elits and stoch picks"

        self.stoch_method=self.param['stochastic_method']
        assert (self.stoch_method in ['normalized_obj_func','uniform','unique_uniform',\
                                      'unique_triangular']),\
        "\n stochastic_method must be 'normalized_obj_func','uniform' or 'unique_uniform"

        if self.param['obj_func_cutoff'] is not None:
            self.f_cutoff = float(self.param['obj_func_cutoff'])
        elif self.timeout_obj_func is not None:
            self.f_cutoff = self.timeout_obj_func
        else:
            self.f_cutoff = None
            
        
        if self.param['max_num_iteration']==None:
            self.iterate=0
            for i in range (0,self.dim):
                if self.var_type[i]=='int':
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*self.dim*(100/self.pop_s)
                else:
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*50*(100/self.pop_s)
            self.iterate=int(self.iterate)
            if (self.iterate*self.pop_s)>10000000:
                self.iterate=10000000/self.pop_s
        else:
            self.iterate=int(self.param['max_num_iteration'])
        
        self.c_type=self.param['crossover_type']
        assert (self.c_type=='uniform' or self.c_type=='one_point' or\
                self.c_type=='two_point'),\
        "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string" 
        
        
        self.stop_mniwi=False
        if self.param['max_iteration_without_improv']==None:
            self.mniwi=self.iterate+1
        else: 
            self.mniwi=int(self.param['max_iteration_without_improv'])
        
        self.time_list = {} #Ms
        
        sys.stdout.write('Start Genetic Algorithm\n')
        sys.stdout.flush()

        #############################################################
        #Start population: Either manual or continuation file
        #Manual start population is assumed to be a 2D array
        if manual_start_population is not None:
            if read_continuation_population:
                raise Exception('Either manual start population or continuation file')
            else:
                nb_individuals,dim = np.shape(manual_start_population)
                if dim != self.dim:
                    raise Exception('Incorrect dimension of manual start population')
                else:
                    self.pop_start = nb_individuals
                    for i, individual in enumerate(manual_start_population):
                        for j, var_type in enumerate(self.var_type):
                            if var_type == 'real':
                                manual_start_population[i][j] = float(manual_start_population[i][j])
                            if var_type == 'int' and not isinstance(manual_start_population[i][j],int):
                                raise Exception('manual start pop, individual ' + str(i) + \
                                                ', gene ' + str(j)  + ' must be int.')
                    self.start_population = manual_start_population

        else:
            if read_continuation_population:
                cfile = open(self.cont_file,'rb')
                cont_dict = pickle.load(cfile)
                if 'pop' in cont_dict:
                    self.start_population = cont_dict['pop'][:, :self.dim] #entry incl. fitness
                else:
                    last_gen = 0
                    last_gen_key = None
                    for key in cont_dict:
                        if 'gen' in key:
                            gen = int(key.replace('gen',''))
                            if gen > last_gen:
                                last_gen = gen
                                last_gen_key = key
                    self.start_population = cont_dict[last_gen_key]['pop'] #entryonly genes
                cfile.close()
                self.pop_start = self.init_pop_s
            else:
                self.pop_start = 0
                #self.start_population = np.array([])
                
        #print('pop_start',self.pop_start)
        #raise Exception()

        ############################################################# 
    def run(self):
        self.clean_up_pkls()
        if self.logfile != None and os.path.isfile(self.logfile):
            os.remove(self.logfile)
            
        if self.func_exec_report_file != None and \
           os.path.isfile(self.func_exec_report_file):
            os.remove(self.func_exec_report_file)

        ############################################################# 
        # Initial Population
        
        self.manage_function_report(report_str="",
                                    title_str="Generation 0:\n",nb_hash=45)
        
        self.integers=np.where(self.var_type=='int')
        self.reals=np.where(self.var_type=='real')
        
        #pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
        pop = np.array([np.zeros(self.dim+1)]*self.init_pop_s)
        solo=np.zeros(self.dim+1)
        var=np.zeros(self.dim)

        #times=np.zeros(self.pop_s)
        times=np.zeros(self.init_pop_s)
        
        if self.parallelization:
            PEX = cf.ProcessPoolExecutor(self.parallel_cores)
            os.environ["OMP_NUM_THREADS"] = "1"
            processes=[]

        #raise Exception()
        for p in range(0, self.init_pop_s):
            if p < self.pop_start:
                #given population
                var = self.start_population[p]
                solo[: self.dim] = var.copy()
            else:
                #raise Exception()
                #generate new individuals
                for i in self.integers[0]:
                    var[i] = np.random.randint(self.var_bound[i][0],\
                             self.var_bound[i][1]+1)
                    solo[i] = var[i].copy()
                for i in self.reals[0]:
                    var[i] = self.var_bound[i][0]+np.random.random()*\
                             (self.var_bound[i][1]-self.var_bound[i][0])
                    solo[i] = var[i].copy()
                    
            if self.parallelization:
                solo[self.dim] = 1.0 #tmp, to find error easier
                processes.append(PEX.submit(self.sim,var.copy(),p))
            else:
                value, time, f_report = self.sim(var,p)
                self.manage_function_report(f_report,p)
                solo[self.dim] = value
                times[p] = time
                
            pop[p] = solo.copy() #         
            
        if self.parallelization:     
            results = [pr.result() for pr in processes]
            for i,result in enumerate(results):
                pop[i][self.dim] = result[0]
                times[i] = result[1]
                f_report = result[2]
                self.manage_function_report(f_report,i)
            #value = result[0]

        #############################################################
        # Sort
        ordered_indexes = pop[:,self.dim].argsort()
        pop = pop[ordered_indexes]
               
        #############################################################

        #############################################################
        # Report

        self.timelapse=[]
        #self.test_obj=pop[0,self.dim]
        self.best_variable=pop[0,: self.dim]
        self.best_function=pop[0,self.dim]
        
        ##############################################################
                        
        t=0
        counter=0
        
        while t<self.iterate and not self.stop_mniwi:
            self.progress(t+1,self.iterate,status="GA is running...")
            #############################################################
            #Sort
            ordered_indexes = pop[:,self.dim].argsort()
            self.write_opt_pkls(ordered_indexes[0]) #Mikaels
            self.clean_up_pkls() #Mikaels
            pop = pop[ordered_indexes]
            times = times[ordered_indexes]
            
            if pop[0,self.dim]<self.best_function:
                counter=0
                self.best_function=pop[0,self.dim].copy()
                self.best_variable=pop[0,: self.dim].copy()
            else:
                counter+=1

            #############################################################
            # Report, write cont file and log
            self.timelapse.append(pop[0,self.dim])
            self.write_to_logfile(t,pop,times)
            self.manage_function_report(report_str="",title_str="Generation " + str(t) + "\n")

            if self.cont_file_writing_frequency:
                #self.pops[t] = pop[:, :self.dim]
                #self.fitnesses[t] = pop[:, self.dim]
                #self.all_eval_times[t] = times
                if t % self.cont_file_writing_frequency == 0:
                    gen = 'gen'+str(t)
                    self.output_dict[gen] = {'pop':pop[:,:self.dim],
                                            'fitness':pop[:,self.dim],
                                            'eval_times':times.copy()}
                    self.write_to_cont_file(completed=False)
                    
            #############################################################
            # Cut down initial population
            if len(pop) > self.pop_s:
                pop = pop[:self.pop_s]
                times = times[:self.pop_s]
    
            ##############################################################         
            # Normalizing objective function 
            
            if self.f_cutoff is not None:
                alive_pop_s = np.searchsorted(pop[:,self.dim],self.f_cutoff)
                if alive_pop_s < self.par_s:
                    print('WARNING: Alive pop not enough for parents. Ressurrecting parents.')
                    alive_pop_s = self.par_s
                normobj = pop[:alive_pop_s,self.dim]
            else:
                alive_pop_s = self.pop_s
                normobj = pop[:,self.dim]

            minobj=pop[0,self.dim]
            if minobj < 0:
                normobj += minobj
    
            maxnorm=np.amax(normobj)
            normobj=maxnorm-normobj+1

            #############################################################        
            # Calculate probability
            
            sum_normobj=np.sum(normobj)
            prob=np.zeros(self.pop_s)
            prob=normobj/sum_normobj
            cumprob=np.cumsum(prob)
  
            #############################################################        
            # Select parents
            par=np.array([np.zeros(self.dim+1)]*self.par_s)

            if self.par_to_new_gen:
                num_to_new_gen=self.par_s
            else:
                num_to_new_gen=self.num_elit

            num_best_par = self.par_s-self.num_stoch
            
            #picked_indices = []
            unpicked_indices = list(range(alive_pop_s))
            #print(alive_pop_s)
            
            for k in range(0,num_best_par):
                par[k]=pop[k].copy()
                #picked_indices.append(k)
                unpicked_indices.remove(k)
            for k in range(num_best_par,self.par_s):
                if self.stoch_method=='normalized_obj_func':
                    index=np.searchsorted(cumprob,np.random.random())
                    par[k]=pop[index].copy()
                elif self.stoch_method=='uniform':
                    index=np.random.randint(0,alive_pop_s)
                    par[k]=pop[index].copy()
                elif 'unique' in self.stoch_method:
                    if alive_pop_s==self.par_s:
                        par[k]=pop[k].copy()
                    else:
                        if self.stoch_method=='unique_uniform':
                            index_index=np.random.randint(len(unpicked_indices))
                        elif self.stoch_method=='unique_triangular':
                            tri_float=np.random.triangular(0,0,len(unpicked_indices))
                            index_index=int(np.floor(tri_float))
                        else:
                            raise Exception('Stochastic method not recognized')
                        index = unpicked_indices[index_index]
                        par[k]=pop[index].copy()
                        unpicked_indices.remove(index)
                else:
                    raise Exception('Stochastic method not recognized')

            rands = np.random.random(self.par_s)
            sorted_inds = rands.argsort()
            #Pick at least two, then according to crossover prob
            k = 2
            while k < self.par_s:
                if rands[sorted_inds[k]] > self.prob_cross:
                    raise Exception()
                    break
                k += 1

            ef_par=par[sorted_inds[:k]]
            par_count=len(ef_par)
            
            #############################################################  
            #New generation
            self.manage_function_report(report_str="",
                                        title_str="\nGeneration " + str(t+1) + ":\n",nb_hash=45)
            
            pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
            
            for k in range(0,num_to_new_gen):
                pop[k]=par[k].copy()
                times[k]=0.0 #Reset runtimes

            processes=[]
            if self.func_exec_report_file != None:
                f_reports = ""

            k = num_to_new_gen
            while k < self.pop_s:
                r1=np.random.randint(0,par_count)
                r2=np.random.randint(0,par_count)
                while r1==r2:
                    r2=np.random.randint(0,par_count)
                pvar1=ef_par[r1,: self.dim].copy()
                pvar2=ef_par[r2,: self.dim].copy()

                ch=self.cross(pvar1,pvar2,self.c_type)
                
                for m, mutalgo in enumerate(self.mutalgos):
                    ran=np.random.random()
                    if ran < self.individual_prob_mut:
                        ch[m] = self.mutate(ch[m],algo=mutalgo,p1=pvar1,p2=pvar2)
                    if k >= self.pop_s: #Will happen at the end if pop_s is odd
                        continue
                    solo[: self.dim]=ch[m].copy()
                    if self.parallelization:
                        solo[self.dim]=1.     #Temporary
                        processes.append(PEX.submit(self.sim,ch[m],k))
                    else:
                        value,time,f_report=self.sim(ch[m],k)
                        solo[self.dim]=value
                        times[index]=time
                        self.manage_function_report(f_report,k)
                    pop[k] = solo.copy()
                    k += 1

            if self.parallelization:
                results=[pr.result() for pr in processes]
                for i,result in enumerate(results):
                    index=i+num_to_new_gen
                    value,time,f_report=result
                    pop[index][self.dim]=value
                    times[index]=time
                    self.manage_function_report(f_report,index)


        #############################################################       
            t+=1
            if counter > self.mniwi:
                pop = pop[pop[:,self.dim].argsort()]
                if pop[0,self.dim]>=self.best_function:
                    #t=self.iterate
                    self.progress(self.iterate,self.iterate,status="GA is running...")
                    #t+=1
                    self.stop_mniwi=True
        #############################################################
        #Sort
        ordered_indexes = pop[:,self.dim].argsort()
        self.write_opt_pkls(ordered_indexes[0]) #Mikaels
        self.clean_up_pkls() #Mikaels
        pop = pop[ordered_indexes]
        times = times[ordered_indexes]

        #mean = np.mean(self.time_list)
        #print('Mean time: ' + str(round(mean,2)) + ' seconds.')
        
        if pop[0,self.dim]<self.best_function:
                
            self.best_function=pop[0,self.dim].copy()
            self.best_variable=pop[0,: self.dim].copy()

        #############################################################
        # Report, log and write cont file

        self.timelapse.append(pop[0,self.dim])
        self.write_to_logfile(t,pop,times)
        #self.manage_function_report(report_str="",title_str="\nGeneration 0\n")

        if self.cont_file_writing_frequency:
            gen = 'gen'+str(t)
            self.output_dict[gen] = {'pop':pop[:,:self.dim],
                                     'fitness':pop[:,self.dim],
                                     'eval_times':times.copy()}
            self.write_to_cont_file(completed=True)
        
        self.output_dict['variable']=self.best_variable
        self.output_dict['function']=self.best_function
        self.output_dict['pop']=pop

        #r = False
        #show=' '*100
        #if r:
        #    sys.stdout.write('\r%s' % (show))
        #    sys.stdout.write('\r The best solution found:\n %s' % (self.best_variable))
        
        sys.stdout.write('\n The best solution found:\n %s' % (self.best_variable))
        sys.stdout.write('\n\n Objective function:\n %s\n' % (self.best_function))
        nb_edge_points = self.boundary_proximity(onesided=0.05)
        if nb_edge_points:
            sys.stdout.write('\n\n Warning! Variable close to edge in %s instances\n'\
                             % (nb_edge_points))
        sys.stdout.flush() 
        re=np.array(self.timelapse)
        if self.save_timelapse:
            plt.clf()
            plt.plot(re)
            plt.xlabel('Iteration')
            plt.ylabel('Objective function')
            plt.title('Genetic Algorithm')
            plt.savefig('timelapse.pdf')
        	#plt.show()
            plt.clf()
        if self.stop_mniwi==True:
            sys.stdout.write('\nWarning: GA is terminated due to the'+\
                             ' maximum number of iterations without improvement was met!\n')
##############################################################################         
##############################################################################         
    def cross(self,x,y,c_type):
         
        ofs1=x.copy()
        ofs2=y.copy()
        

        if c_type=='one_point':
            ran=np.random.randint(0,self.dim)
            for i in range(0,ran):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()
  
        if c_type=='two_point':
                
            ran1=np.random.randint(0,self.dim)
            ran2=np.random.randint(ran1,self.dim)
                
            for i in range(ran1,ran2):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()
            
        if c_type=='uniform':
                
            for i in range(0, self.dim):
                ran=np.random.random()
                if ran <0.5:
                    ofs1[i]=y[i].copy()
                    ofs2[i]=x[i].copy() 
                   
        return np.array([ofs1,ofs2])
###############################################################################  
    def mutate(self,x,algo='uniform',
               p1=None,p2=None,stop_at_boundary=True):
        if self.max_muts != None:
            muts = np.random.randint(1,self.max_muts+1)
            gene_inds = np.random.choice(range(muts),muts)
            integers = [i for i in self.integers[0] if i in gene_inds]
            reals = [i for i in self.reals[0] if i in gene_inds]
        else:
            integers = self.integers[0]
            reals = self.reals[0]
                
        if algo == 'uniform':
            #Mutated genes take uniform value over the interval
            for i in integers:
                ran=np.random.random()
                if ran < self.prob_mut:              
                    x[i]=np.random.randint(self.var_bound[i][0],\
                     self.var_bound[i][1]+1)
                     
            for i in reals:                
                ran=np.random.random()
                if ran < self.prob_mut:   
                   x[i]=self.var_bound[i][0]+np.random.random()*\
                    (self.var_bound[i][1]-self.var_bound[i][0])
        elif algo == 'middle':
            #Mutated gene take uniform value between the parents' genes
            if p1 == None or p2 == None:
                raise Exception('Error with parents in mutate_middle')
                
            for i in integers:
                ran=np.random.random()
                if ran < self.prob_mut:
                    if p1[i]<p2[i]:
                        x[i]=np.random.randint(p1[i],p2[i]+1)
                    elif p1[i]>p2[i]:
                        x[i]=np.random.randint(p2[i],p1[i]+1)
                    else:
                        x[i]=np.random.randint(self.var_bound[i][0],\
                     self.var_bound[i][1]+1)
                            
            for i in reals:                
                ran=np.random.random()
                if ran < self.prob_mut:   
                    if p1[i]<p2[i]:
                        x[i]=p1[i]+np.random.random()*(p2[i]-p1[i])  
                    elif p1[i]>p2[i]:
                        x[i]=p2[i]+np.random.random()*(p1[i]-p2[i])
                    else:
                        x[i]=self.var_bound[i][0]+np.random.random()*\
                    (self.var_bound[i][1]-self.var_bound[i][0])
        elif algo == 'coinflip':
            #50% if change up or down for old gen
            #Then +/- min(factor * interval,distance to boundary)
            for i in integers:
                ran = np.random.random()
                if ran < self.prob_mut:
                    interval = self.var_bound[i][1] - self.var_bound[i][0]
                    max_diff = math.ceil(self.mutation_factor * interval)
                    flip = np.random.random()
                    if flip >= 0.5:
                        max_diff_up = min(max_diff,self.var_bound[i][1]-x[i])
                        x[i] += np.random.randint(0,max_diff_up+1)
                    else:
                        max_diff_down = min(max_diff,x[i]-self.var_bound[i][0])
                        x[i] -= np.random.randint(0,max_diff_down+1)
                        
            for i in reals:
                ran = np.random.random()
                if ran < self.prob_mut:
                    interval = self.var_bound[i][1]-self.var_bound[i][0]
                    max_diff = self.mutation_factor*interval
                    flip = np.random.random()
                    if flip >= 0.5:
                        max_diff_up = min(max_diff,self.var_bound[i][1]-x[i])
                        x[i] += np.random.random() * max_diff_up
                    else:
                        max_diff_down = min(max_diff,x[i]-self.var_bound[i][0])
                        x[i] -= np.random.random() * max_diff_down
        elif algo == 'factor_with_truncation':
            #Mutated gene is old gene +/- factor * interval
            for i in integers:
                ran = np.random.random()
                if ran < self.prob_mut:
                    interval = self.var_bound[i][1] - self.var_bound[i][0]
                    max_diff = math.ceil(self.mutation_factor * interval)
                    x[i] += np.random.randint(-max_diff,max_diff+1)
                    
            for i in reals:
                ran = np.random.random()
                if ran < self.prob_mut:
                    interval = self.var_bound[i][1] - self.var_bound[i][0]
                    max_diff = self.mutation_factor * interval
                    x[i] += (2*np.random.random()-1.0) * max_diff
                    
            if stop_at_boundary:
                for i in range(len(x)):
                    x[i] = min(x[i],self.var_bound[i][1])
                    x[i] = max(x[i],self.var_bound[i][0])
            
        return x
            
###############################################################################

    def mutuniform(self,x):
        #Mutated genes take uniform value over the interval

        for i in self.integers[0]:
            ran=np.random.random()
            if ran < self.prob_mut:              
                x[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1) 
                    
        

        for i in self.reals[0]:                
            ran=np.random.random()
            if ran < self.prob_mut:   
               x[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0])    
            
        return x
###############################################################################
    def mutmidle(self, x, p1, p2):
        #Mutated gene take uniform value between the parents' genes
        for i in self.integers[0]:
            ran=np.random.random()
            if ran < self.prob_mut:
                if p1[i]<p2[i]:
                    x[i]=np.random.randint(p1[i],p2[i])
                elif p1[i]>p2[i]:
                    x[i]=np.random.randint(p2[i],p1[i])
                else:
                    x[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1)
                        
        for i in self.reals[0]:                
            ran=np.random.random()
            if ran < self.prob_mut:   
                if p1[i]<p2[i]:
                    x[i]=p1[i]+np.random.random()*(p2[i]-p1[i])  
                elif p1[i]>p2[i]:
                    x[i]=p2[i]+np.random.random()*(p1[i]-p2[i])
                else:
                    x[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0]) 
        return x
###############################################################################
    def mutfactor(self, x, factor,stop_at_boundary=True):
        #Mutated gene is old gene +/- factor * interval
        for i in self.integers[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                interval = self.var_bound[i][1] - self.var_bound[i][0]
                max_diff = math.ceil(factor * interval)
                x[i] += np.random.randint(-max_diff,max_diff+1)
                
        for i in self.reals[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                interval = self.var_bound[i][1] - self.var_bound[i][0]
                max_diff = factor * interval
                x[i] += (2*np.random.random()-1.0) * max_diff
                
        if stop_at_boundary:
            for i in range(len(x)):
                x[i] = min(x[i],self.var_bound[i][1])
                x[i] = max(x[i],self.var_bound[i][0])
            
        return x
###############################################################################
    def mutcoinflip(self, x, factor):
        #50% if change up or down for old gen
        #Then +/- min(factor * interval,distance to boundary)
        for i in self.integers[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                interval = self.var_bound[i][1] - self.var_bound[i][0]
                max_diff = math.ceil(factor * interval)
                flip = np.random.random()
                if flip >= 0.5:
                    max_diff_up = min(max_diff,self.var_bound[i][1]-x[i])
                    x[i] += np.random.randint(0,max_diff_up+1)
                else:
                    max_diff_down = min(max_diff,x[i]-self.var_bound[i][0])
                    x[i] -= np.random.randint(0,max_diff_down+1)
                    
        for i in self.reals[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                interval = self.var_bound[i][1]-self.var_bound[i][0]
                max_diff = factor*interval
                flip = np.random.random()
                if flip >= 0.5:
                    max_diff_up = min(max_diff,self.var_bound[i][1]-x[i])
                    if max_diff_up < 0:
                        raise Exception()
                    x[i] += np.random.random() * max_diff_up
                else:
                    max_diff_down = min(max_diff,x[i]-self.var_bound[i][0])
                    if max_diff_down < 0:
                        raise Exception()
                    x[i] -= np.random.random() * max_diff_down
        return x
                
###############################################################################
    def evaluate(self):
        return self.f(self.temp,self.index)
###############################################################################    
    def sim(self,X,i):
        self.temp=X.copy()
        self.index=i
        #obj=None
        result = None
        start = time.time()
        try:
            #obj=func_timeout(self.funtimeout,self.evaluate)
            result=func_timeout(self.funtimeout,self.evaluate)
            """
            if not isinstance(obj,float):
                print('Obj is not float. Obj:',obj,' type:',type(obj))
                print('Individual:')
                pprint(X)
                raise Exception('Objective function must be float')
            """
        except FunctionTimedOut:
            #print('Given function has not provided output after ' + \
            #      str(self.funtimeout) + ' seconds delay.' + \
            #      'Obj function set to ' + str(self.timeout_obj_func))
            #obj = self.timeout_obj_func
            result = (self.timeout_obj_func,"Evaluation timed out\n")
        if isinstance(result,tuple):
            obj=result[0]
            report=result[1]
        else:
            obj=result
            report=""
        if not isinstance(obj,float):
            print('Obj is not float. Obj:',obj,' type:',type(obj))
            print('Individual:',i)
            raise Exception('Objective function must be float')
        """
        assert (obj!=None), "After "+str(self.funtimeout)+" seconds delay "+\
                "func_timeout: the given function does not provide any output"
        """
        runtime = time.time() - start
        if self.verbose:
            print('Obj: ' + str(round(obj,2)) + ', Evaluation time: ' + str(round(runtime,1)))
        return obj,runtime,report

###############################################################################
    def progress(self, count, total, status=''):
        percents = round(100.0 * count / float(total), 1)
        if self.progress_bar:
            bar_len = 50
            filled_len = int(round(bar_len * count / float(total)))
            bar = '|' * filled_len + '_' * (bar_len - filled_len)
            sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        else:
            sys.stdout.write('%s%s %s\n' % (percents, '%', status))
        sys.stdout.flush()
###############################################################################
    def write_to_logfile(self,gen,pop,times):
        if self.logfile != None:
            l = open(self.logfile,'a')
            l.write('Generation ' + str(gen) + ':\n')

            int_len = 1
            frac_len = 1
            for p in range(len(pop)):
                obj_str = f"{pop[p][self.dim]:.3}"
                if 'e' in obj_str:
                    integer = str(pop[p][self.dim]).split('.')[0]
                    fraction = '0'
                else:
                    integer,fraction = obj_str.split('.')
                int_len = max(int_len,len(integer))
                frac_len = max(frac_len,len(fraction))
            for p in range(len(pop)):
                l.write('    Individual: ' + '{:>3}'.format(p) + ', Obj: ' + \
                        f"{pop[p][self.dim]:{int_len+frac_len+1}.{frac_len}f}" + \
                        ', Time: ' + f"{times[p]:.3}" + '\n')
        return

###############################################################################
    def manage_function_report(self,report_str,index="",nb_hash=35,title_str=None):
        if title_str == None:
            title_str = "Individual " + str(index) + ":\n"
        title_str += nb_hash * "#" + "\n"
        report_str = title_str + report_str
        if self.func_exec_report_file == None:
            print(report_str)
        else:
            fd = open(self.func_exec_report_file,'a')
            fd.write(report_str + '\n')
            fd.close()
        return
            
###############################################################################
    def clean_up_pkls(self):
        for filename in self.pkl_list:
            if '*' in filename:
                name = filename.split('*')
                for i in range(self.pop_s):
                    f = name[0] + str(i)  + name[1]
                    if os.path.isfile(f):
                        os.remove(f)
            else:
                raise Exception('Data file name must contain * to be replaced \
                                with index')
        return
###############################################################################  
    def write_opt_pkls(self,index):
        for filename in self.pkl_list:
            name = filename.split('*')
            f = name[0] + str(index) + name[1]
            opt_f = name[0] + name[1]
            if os.path.isfile(f):
                shutil.move(f,opt_f)
        return
###############################################################################   
    def write_to_cont_file(self,completed=False):
        if self.cont_file is not None:
            cfile = open(self.cont_file,'wb')

            self.output_dict['best_function_report'] = np.array(self.timelapse)
            self.output_dict['variable'] = self.best_variable #Not nec. from last gen if non-elitist
            self.output_dict['function'] = self.best_function #Not nec. from last gen if non-elitist
            self.output_dict['completed'] = completed

            pickle.dump(self.output_dict,cfile)

            cfile.close()
        return
        
###############################################################################
    def boundary_proximity(self,onesided):
        # onesided is the ratio considered to be close to a boundary, say
        # onesided=0.1 means that being within 10% of the boundary raises
        # an 'edge point'
        nb_edge_points = 0
        for i in range(self.dim):
            var_range = self.var_bound[i][1] - self.var_bound[i][0]
            low_threshold = self.var_bound[i][0] + onesided * var_range
            high_threshold = self.var_bound[i][1] - onesided * var_range
            if self.best_variable[i] < low_threshold or \
               self.best_variable[i] > high_threshold:
                nb_edge_points += 1
        return nb_edge_points
###############################################################################
