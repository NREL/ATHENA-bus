""" Author: Devon Sigler 
    Purpose: This file contains tests that verify the installation of the software dependecies and the software in this repo. This is done by verifying that the sample data provided in this repo can be read in and used to construct an instance of the optimization model, and the open source solver GLPK can being the process of solving that model. We terminate the solution process prematurely because we only need to see that the alogithms from glpk can be executed on the model constructed. """
# test file
#### importing packages ################################################################
import pandas as pd
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import jinja2 as jin
import time
import datetime
from datetime import timedelta
import math
import scipy
import glob
import pickle
import cloudpickle
import os
import sys
from pyomo.environ import *
from pyomo.pysp.ef import create_ef_instance
from pyomo.opt import SolverFactory
from pyutilib.misc.config import ConfigBlock
import route_optimization.opt_lib as RO


def test_solve():
    pyomo_data_dir_name = './VR_data'
    pyomo_data_file_name = 'Test_Scenario.dat'
    path_to_travel_time_csv = "./example_data/travel_time_example.csv"
    path_to_bus_type_csv = "./example_data/bus_types_example.csv"
    path_to_requests_csv = "./example_data/requests_example.csv"
    path_to_dwell_time_csv = "./example_data/dwell_time_example.csv"
    path_to_parameters_csv = "./example_data/parameters_example.csv"


    RO.generate_data_file(pyomo_data_dir_name,pyomo_data_file_name,path_to_travel_time_csv, 
                                          path_to_bus_type_csv,path_to_requests_csv,
                                          path_to_dwell_time_csv,path_to_parameters_csv)

    solution = RO.run_VR_model(pyomo_data_dir_name,pyomo_data_file_name,'glpk',mip_gap = 0.001, max_time = 20, verbose_output = True)

    RO.save_solution('Test.pkl',"./tests",solution)
 
    os.remove('./tests/Test.pkl')
