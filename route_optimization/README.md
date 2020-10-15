This module contains a set of functions in the opt_lib.py file. The code in these functions completes the following tasks:

1. Reads in data files and reformats them for writing out pyomo .dat files
2. Constructs pyomo .dat files using Jinja2 templates
3. Constructs the abstract pyomo model
4. Creates instances of the abstract pyomo model using a .dat file
5. Solves instances of the pyomo model and saves the solution results
6. Reads pickled solution files and provides some utility functions for inspecting the routes computed 
