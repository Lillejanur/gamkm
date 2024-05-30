GA_MKMV, GENETIC ALGORITHM WITH MIKROKINETIC MODEL
Written mostly by Mikael Valter-Lithander and partly by Minttu Kauppinen

The program runs a genetic algorithm, whose primarily adjustable parameters are energies, adsorbate-adsorbate interaction parameters
for a microkinetic model of a chemical reaction. The intention is to match the model output, coverages of adsorbates, to experimental
XPS data. More parameters for the genetic algorithm are changes to the XPS peaks and broadenings.

The purpose of the program is to introduce a new method for comparison between theory and experiment in catalysis, to assess DFT errors,
and to pinpoint the problems with a model. If, e.g., very large energy differences are required for agreement, a conclusion could be
that more complexity in the form of difference surface facets or more intermediary steps are needed. If only a few adsorbates stick out,
it could be interesting to study them with higher-level electronice structure calculations than DFT.

A manuscript using the program can be found here: 10.26434/chemrxiv-2024-5c9p3

To install, run `pip install --editable .` (--editable is optional)

Files in gamkm:
      
ga_mkmv[xx].py - The main program.
geneticalgorithm[xx].py - The genetic algorithm. Intended to work independently of the mikrokinetics.
mikrokinetics[xx].py - Calculate mikrokinetics
energy_calc[xx].py - Calculate thermodynamics
thermodynamics.py - Called by energy_calc[xx].py

Please see a separate README in examples
