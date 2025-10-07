''' 

Configuration file for the simulation:

Here you can set the parameters used in the simulation.

'''

N_EVGEN = 1000          # Events generated in the simulation
N_TRK = 1               # Number of simultaneous tracks per event
mu = 0.                 # Alignment precision precisione
sigma = 0.005			# ALPIDE spatial resolution 4-5 micron or 28/sqrt(12) 0.005
ProbNoise = 1e-6        # ALPIDE fake-rate
ProbMiss = 1e-6         # Dead pixels or threshold related inefficiencies