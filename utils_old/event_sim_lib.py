

#### importing packages ################################################################

#import Pyro
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

import utils


import multiprocessing
import subprocess
import numpy as np
import os
import time
import cloudpickle

import pickle


class Bus(object):
    def __init__(self,s,loc,time,name,route):
        # init should be used to do assignment of attributes
        self.current_state=s
        self.current_location=loc
        self.current_time = time
        self.name = name
        self.current_route = route




def update_active_bus(bus,routes_dict,day,time_window,tt_dict,dwell_times_dict ):
    
    route_name = bus.current_route
    if route_name in routes_dict.keys():
        
        route = routes_dict[route_name]
        dwell_means = route[0]
        tt_means = route[1]
        mapping = route[2]
    
        n = len(dwell_means.keys())

        assert bus.current_state == 'in route'
            
            # update time
        if bus.current_location < n-1:
            pair = (bus.current_location,bus.current_location + 1)
        else:
            pair = (bus.current_location,0)

        #tt_mean = tt_means[pair]
        #dwell_mean = dwell_means[pair[1]]
        
        start = pair[0]
        end = pair[1]
        geo_end = mapping[end]
        #trip_name = mapping[start] + mapping[end]
        time_sample = sample_tt(pair,day,time_window,tt_dict,route)
        dwell_sample = sample_dwell(geo_end,day,time_window,dwell_times_dict)
        
        #bus.current_time = bus.current_time + np.random.normal(tt_mean,5.0) + np.random.normal(dwell_mean,2.0)
        bus.current_time = bus.current_time + time_sample + dwell_sample
        # update loc
        if bus.current_location < n-1:
            bus.current_location = bus.current_location + 1
        else:
            bus.current_location = 0
        # update state and assign no route
        if  bus.current_location == 0:
            bus.current_state ='parked'
            bus.current_route = 'none'
    return bus




def activate_bus(bus,time,route):
    bus.current_state = 'in route'
    bus.current_time = time + 1
    bus.current_location = 0
    bus.current_route = route
    return 
    


def check_for_bus(bus_dict, time):
    avail_buses = []
    for key in bus_dict:
        bus  = bus_dict[key]
        if (bus.current_time <= time) and (bus.current_location == 0) and         (bus.current_state == 'parked') and (bus.current_route == 'none') :
            avail_buses.append(key)
    return avail_buses




def compute_route_length(route_name,routes_dict):
    
    route = routes_dict[route_name]
    dwell_means = route[0]
    tt_means = route[1]
    
    dwell_sum = 0
    for key in dwell_means.keys():
        dwell_sum  += dwell_means[key]#[0]
        
    tt_sum = 0
    for key in tt_means.keys():
        tt_sum  += tt_means[key]
    route_length = tt_sum + dwell_sum
    return route_length




def compute_theoretical_bus_number(route_name,routes_dict, head_way):
    route_length = compute_route_length(route_name,routes_dict)
    num_buses = ceil(route_length/head_way)
    return num_buses



def evolve_fleet_in_time(bus_dict,time,routes_dict,day,time_window,tt_dict,dwell_times_dict):  
    need_update_list = bus_dict.keys()
    while bool(need_update_list) == True :
        for bus_name in need_update_list :
            bus = bus_dict[bus_name]

            if bus.current_time >= time:
                need_update_list.remove(bus_name)
            elif (bus.current_state =='parked') and (bus.current_route == 'none') :
                need_update_list.remove(bus_name)
            else:
                bus = update_active_bus(bus,routes_dict,day,time_window,tt_dict,dwell_times_dict)
                bus_dict[bus_name] = bus
    return bus_dict



def run_sim(stagger,flex_allowed,num_buses,routes_dict,routes,times,day,time_window,tt_dict,dwell_times_dict):
    
    b = num_buses
    buses = range(b)

    bus_dict ={}

    for bus in buses:
        name = 'bus_' + str(bus) 
        route = 'none'
        bus_dict[name] = Bus('parked',0,0,name,route)
    
    
    fail = 0
    stagger = 0
    flex_allowed = 5.0
    for time in times:
        #print time ,time/1425.

        #routes =[1,2,3,4,5]
        for route in routes:
            fail = 0


            #check available buses
            avail_buses = check_for_bus(bus_dict, time)

            try:
                free_bus = avail_buses.pop()
                 # if available bus activate it
                bus = bus_dict[free_bus]
                activate_bus(bus,time,route)
            except:
                #print 'FAILURE no buses within hard constraints'
                fail = 1

            if fail == 1:
                try:

                    bus_dict=evolve_fleet_in_time(bus_dict,time + flex_allowed,routes_dict,day,time_window,tt_dict,dwell_times_dict)
                    avail_buses = check_for_bus(bus_dict, time + flex_allowed)
                    free_bus = avail_buses.pop()

                    # if available bus activate it
                    bus = bus_dict[free_bus]
                    soonest_time = bus.current_time
                    activate_bus(bus,soonest_time ,route)


                except:
                    #print 'FAILURE no buses'
                    fail = 2
                    break

            # update time 
            time = time + stagger
            # now we update all active buses until there time is past the new time or they
            # are not active
            bus_dict=evolve_fleet_in_time(bus_dict,time,routes_dict,day,time_window,tt_dict,dwell_times_dict)

        if fail == 2:
#             print time 
#             for bus in bus_dict.values():
#                 print float(bus.current_time), bus.current_location
            break


    return fail




# In[17]:


def create_route_data(route,route_name,tt_dict,dwell_times_dict,routes_dict,day,time_window,df_term):
    dwell_means = {}
    mapping_dict = {}
    tt_means = {}
    for i in range(len(route)-1):
        
        mapping_dict[i] = route[i]
        geo_loc = route[i]
        
        if route[i] == 'R':
            geo_loc = 'A' 
        
        y = dwell_times_dict[geo_loc][day][time_window]
        dwell_mean = np.array(y).mean()
        dwell_means[i] = dwell_mean/60.0
    
    for i in range(len(route) - 1): 

        if i+1 == len(route) - 1:
            num_pair = (i,0)
        else:
            num_pair = (i, i+1)
        pair = str(route[i]) + str(route[i +1])
        if (pair[0] !='R') and (pair[1] != 'R'):
            x = df_term.loc[pair]
        else:
            x = tt_dict[pair][day][time_window]
        tt_mean = np.array(x).mean()
        tt_means[num_pair] = tt_mean/60.0
        
    routes_dict[route_name] = [dwell_means,tt_means,mapping_dict]
    return routes_dict


def sample_tt(pair,day,time_window,tt_dict,route):
            
    tt_means = route[1]
    mapping = route[2]
    
    start = pair[0]
    end = pair[1]
    geo_start = mapping[start]
    geo_end = mapping[end]
    trip_name = geo_start + geo_end
    
    if (geo_start !='R') and (geo_end != 'R'):
        x = tt_means[pair]/60.0
        #x = df_term.loc[trip_name]/60.0
        sample = np.random.normal(x,1.0)
    else:
        A = np.array(tt_dict[trip_name][day][time_window])
        sample = np.random.choice(A)/60.0
        
    return sample


def sample_dwell(geo_stop,day,time_window,dwell_times_dict):
    if geo_stop == 'R':
        geo_stop = 'A'
        A = np.array(dwell_times_dict[geo_stop][day][time_window])
        sample = np.random.choice(A)
        sample = sample/2 
        sample = sample/60.0
    
    else:
        A = np.array(dwell_times_dict[geo_stop][day][time_window])
        sample = np.random.choice(A)/60.0
    return sample



def determine_fleet_size(stagger,flex_allowed,num_buses,routes_dict,routes,
                       times,day,time_window,tt_dict,dwell_times_dict):
    fail = 2
    num_buses = 1
    while fail !=0: 
        fail = run_sim(stagger,flex_allowed,num_buses,routes_dict,routes,
                       times,day,time_window,tt_dict,dwell_times_dict)

        num_buses += 1
        
    return num_buses



def compute_sim_bus_number(day,time_window,stagger,flex_allowed,head_way,routes_as_dict):
    
    # read in data that must be present 
    df_term = pd.read_csv('terminal_tt.csv', index_col = ['combos'])

    pickle_off = open("dwell_dict.pkl","rb")
    dwell_times_dict = pickle.load(pickle_off)

    pickle_off = open("tt_dict.pkl","rb")
    tt_dict = pickle.load(pickle_off)
    
    # create simulation duration
    div = int(60.0/head_way)
    times = [ head_way*i for i in range(div*4*52) ]
    
    
    # construct route dictionary as needed for simulations
    routes_dict = {}
    for key,val in routes_as_dict.items():
        #print key, val
        routes_dict = create_route_data(val,key,tt_dict,dwell_times_dict,
                                    routes_dict,day,time_window, df_term)
    
    # compute theoretical number of bused need with perfect spacing
    routes = routes_as_dict.keys()
    total =0 
    theo_list = []
    for name in routes:
        num_buses = compute_theoretical_bus_number(name,routes_dict, head_way)
        total = total + num_buses
        theo_list.append((name,num_buses))
    theo_num = total
    
    # compute similated number
    sim_num = determine_fleet_size(stagger,flex_allowed,num_buses,routes_dict,routes,
                       times,day,time_window,tt_dict,dwell_times_dict)
    
    return theo_num, sim_num , theo_list
