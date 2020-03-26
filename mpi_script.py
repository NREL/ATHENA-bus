from mpi4py import MPI
import pandas as pd
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
#import cufflinks as cl
#cl.go_offline()
import jinja2 as jin

#import cytoolz00
import time
import datetime
from datetime import timedelta
import math
import scipy
import glob
import scipy

import os
import sys
from pyomo.environ import *
#from pyomo.pysp.scenariotree.scenariotreemanager import ScenarioTreeManagerSerial
from pyomo.pysp.ef import create_ef_instance
from pyomo.opt import SolverFactory
from pyutilib.misc.config import ConfigBlock
from pyomo.opt import SolverStatus, TerminationCondition

import multiprocessing
import subprocess
import numpy as np
import os
import time
import cloudpickle

from utils import opt_lib as UT

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(size)
if rank == 0:
    solution_dir = '/Users/dsigler/ATHENA-bus/VR_solutions'
    
    with open("models.txt") as file: # Use file to refer to the file object

       names = file.read()
    
    name_list = names.split('\n')
    if name_list[-1] == '':
       name_list = name_list[0:-1]
    
    assign_tasks = {}
    for i in range(1,size) :
        assign_tasks[i] = []
        
    for i in range(len(name_list)):
        loc = (i % (size-1)) + 1
        data = name_list[i]
        assign_tasks[loc].append((solution_dir,data))
        
    for i in range(1,size):
        data = assign_tasks[i]
        comm.send(data, dest=i, tag=11)

elif rank != 0:

    data = comm.recv(source=0, tag=11)
    print(data)
    
    for example in data:
        model_instance,results = UT.run_VR_model(example,600)
