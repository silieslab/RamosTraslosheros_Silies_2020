## Packages
To run the model code the following packages are used. Software was tested with the indicated versions.
- Python 3.5
- NEURON 7.4 (1370:16a7055d4a86) 2015-11-09

Additionally python libraries `numpy`, `scypi` and `matplotlib` are required for numerical analysis and plotting, respectively. And helper libraries `os`, `re` and `pickle` are required for operative system instructions, regular expressions and data storage, respectively.

Model was run on a linux computer with Ubuntu 18.04.

The model could also run on a Windows 10 PC, with NEURON 7.8.1 and python 3.5.

## Installation
### Python
We used the anaconda software to manage the Python software.
Installation instructions are given in https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

An example environment (here named myenv) with python version 3.5 can be created by running the command on the terminal
```
conda create -n myenv python=3.5
```
For the installing python packages run the folloiwing
```
conda install numpy matplotlib scipy
```
### NEURON
Installation instructions from source code can be found here https://neuron.yale.edu/neuron/getstd

Alternatively, installation can be done from precompiled files (https://neuron.yale.edu/neuron/download), although this was not the procedure followed by this paper.

A set of instructions for installing NEURON with Python in Linux can also be found in https://neurojustas.com/2018/03/27/tutorial-installing-neuron-simulator-with-python-on-ubuntu-linux/

And instructions for Windows 10 used to test our model can be found in https://www.neuron.yale.edu/ftp/neuron/nrn_mswin_install.pdf

Installation time may vary depending on the operative system. We anticipate 30 min without exceptional troubleshooting.


## Running the scripts
### Generating output data
We provide a script to generate the output data from the model to sine gratings moving in different directions called `simulate_direction_tuning.py`
The script can be run from its home directory by typing the following in the command line
```
python simulate_direction_tuning.py
```
It can be timed with the command
```
time python simulate_direction_tuning.py
```

In our linux (32 GB RAM) laptop the result is:
```
Runtime:
real	7m40,067s
user	7m17,399s
sys	0m19,770s
```
While the Windows 10 computer (8GB RAM) took about 30 min.

After the simulation is done, the data will be stored in the output_data folder.

### Plotting output data
We also provide a script to plot the output data called `plot_direction_tuning.py`
Run it as
```
python plot_direction_tuning.py
```

The script will collect data from all simulations into a `outputdata/summary_dict.pkl` file that will be used to generate two figures, one for the direction tuning curve in polar coordinates and a bar plot with the direction selectivity index for each of the three model variants,
and a second one with the voltage response changes for gratings moving in the preferred and null directions. Figures will be saved as PDF files to the figures folder.


## Code organization

### To generate simulation data
#### simulate_direction_tuning.py
script to simulate the response of T5 dendrites to a moving grating across different directions to calculate direction selectivity according to the properties of the input neurons. This is also used to simulate direction selectivity for other temporal frequencies.

#### simulate_direction_tuning_norm_eiratio.py
script to simulate the response of T5 dendrites to a moving grating across different directions to calculate direction selectivity according to the properties of the input neurons, for a range of inhibition to excitation ratios (I/E). 

#### neuronsimulation.py
simulates T5 dendrite response to moving sinewave gratings.
- local dependencies: simulatesinewave
- python packages: neuron, numpy

#### simulatesinewave.py
Simulate responses to a sinewave of fixed paramaters besides the motion direction, output the series resistance to be used in neuron simulations.
- local dependencies: visualstimuli, receptivefields
- python packages: numpy

#### visualstimuli.py
Generates visual stimuli, so far only sinewaves are implemented.
- local dependencies: none
- python packages: numpy

#### receptivefields.py
transforms visual stimulus into a series resistance to be injected into the NEURON simulation via SEVC (single-electrode voltage-clamp) process.
uses a single hexagonal array to place the receptive fields of T5 input neurons
- local dependencies: timefilters
- python packages: numpy, matplotlib, scypi

#### timefilters.py
provides the temporal filters of the LN model used in receptivefields
- dependencies: none

### To analyze simulation data

#### analyzesimulation.py
This is contained in a module with a class SimData and two functions
 get_freq_amp and get_pref_dir


### To plot simulation data
#### plot_direction_tuning.py
Plots tuning curves over motion directions for the different models using different readouts: max voltage, voltage A1 Fourier component, and mean calcium response.
- local dependencies: collect_simulation_data, analyzesimulation, plotsimulation
- python packages: matplotlib, numpy, os, re, pickle

#### plot_direction_tuning_over_tf.py
Plots tuning curves over motion directions for the different models over different temporal frequencies and different readouts
- local dependencies: collect_simulation_data, analyzesimulation, plotsimulation
- python packages: matplotlib, numpy, os, re, pickle

#### plot_direction_tuning_ei.py
Plots tuning curves over motion directions for the different models over different EI ratios and different readouts
- local dependencies: collect_simulation_data, analyzesimulation, plotsimulation
- python packages: matplotlib, numpy, os, re, pickle

