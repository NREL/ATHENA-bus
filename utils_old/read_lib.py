# packages

# coding: utf-8

# In[1]:


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
import utils
import cloudpickle



def check_status(path,case_name):
    name = path + case_name + '_Scenario_solution_status.csv'
    df = pd.read_csv(name)
    return df.columns[1], df.iloc[0,1]





def read_pickle(path,case_name):
    #print path + case_name + '_Scenario.pkl'
    with open(path + case_name + '_Scenario.pkl', mode='rb') as file:
        test = cloudpickle.load(file)
    return test



def get_route(df_nodes, instance):
    list_num =[]
    list_geo = []

    k=1
    start = 0
    list_num.append(start)
    node = 0
    while start < 21:
        for j in range(0,22):
            if instance.x[start,j,k].value > .5:
                #print start,j

                list_geo.append(df_nodes.loc[start,'geo node'])
                start = j
                list_num.append(j)

    list_geo.append(df_nodes.loc[start,'geo node'])
    return list_num, list_geo




def clean_route(list_geo):    
    clean_list = ['R']
    for i in list_geo:
        if i != clean_list[-1]:
            clean_list.append(i)

    return clean_list
    



def route_cap(list_num,instance):
    cap_list = []
    k=1
    for node in list_num:
        #print instance.Q_track[node,k].value
        cap_list.append(ceil(instance.Q_track[node,k].value))
        
    return cap_list



def loop_length(instance):
    distance_list = []
    for i in range(len(list_num) - 1):
        start = list_num[i]
        end = list_num[i+1]
        d = instance.dist[start,end]
        distance_list.append(d)


    total_distance = np.sum(np.array(distance_list))
    return total_distance
    


def compute_loop_time(instance,df_nodes): 
    list_num, list_geo = get_route(df_nodes, instance)
    total_time_list = []
    for i in range(len(list_num)):
        node = list_num[i]
        s = instance.s[node]
        total_time_list.append(s)
        if i < len(list_num) - 1:
            start = list_num[i]
            end = list_num[i+1]
            tt = instance.tt[start,end]
            total_time_list.append(tt)


    total_time = np.sum(np.array(total_time_list))
    return total_time



def cost_per_hour(instance,df_nodes):
    
    total_time = compute_loop_time(instance,df_nodes)
    obj = get_objective(instance)
    x = float(obj) / float(total_time)
    return x     




def get_max_ride(instance):
    x = instance.max_ride_time.value
    return x




def get_HW(instance):
    x = instance.HW.value
    return x



def get_objective(instance):
    x = instance.Total_Cost()
    return x




def get_vehicles(instance):
    x = instance.nv[1].value
    return x






def create_look_up_dictionary():
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_windows = [[0,3],[4,7],[8,11],[12,15],[16,19],[20,23]]
    max_rides = [15,20,25]
    head_ways = [10,15,20]
    dwell_reductions =[0.8,1.0,1.2]
    look_up_dictionary ={}
    for day in days:
        look_up_dictionary[day] = {}
        for time_window in time_windows:
            tw = 'tw=' + str(time_window)
            look_up_dictionary[day][tw] ={}
            for max_ride in max_rides:
                mr = 'mr=' + str(max_ride)
                look_up_dictionary[day][tw][mr] ={}
                for head_way in head_ways:
                    hw = 'hw=' + str(head_way)
                    look_up_dictionary[day][tw][mr][hw] ={}
                    for dwell_reduction in dwell_reductions:
                        dr = 'dr=' + str(dwell_reduction)
                        look_up_dictionary[day][tw][mr][hw][dr] ={}

    return look_up_dictionary



def create_case_list(days,time_windows,max_rides,head_ways,dwell_reductions):
#     days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
#     time_windows = [[0,3],[4,7],[8,11],[12,15],[16,19],[20,23]]
#     max_rides = [15,18,25]
#     head_ways = [10,15,20]
#     dwell_reductions =[0.8,1.0,1.2]
    case_list = []
    solution_dict ={}
    for day in days:
        for time_window in time_windows:
            for max_ride in max_rides:
                for head_way in head_ways:
                    for dwell_reduction in dwell_reductions:
                        file_name = str(day) + '_tw=' + str(time_window) + '_mr=' + str(max_ride) + '_hw=' + str(head_way) + '_dr=' + str(dwell_reduction)
                        case_list.append(file_name)
                        
    return case_list

def partition_route(route):
    I =[]
    for i in range(len(route)):
        if route[i] == 'R':
            I.append(i)
    
    route_partition ={}
    for j in range(len(I)-1):
        route_partition[j] = route[I[j]:I[j+1] + 1]

    return route_partition   


