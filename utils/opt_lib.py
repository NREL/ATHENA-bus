# place functions here
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

import pickle
import cloudpickle







##################################################################

def read_data():
    pickle_tt = open("/Users/dsigler/grid/notebooks/raw_tt_dict.pkl","rb")
    dict_tt = pickle.load(pickle_tt)
    
    pickle_dwell = open("/Users/dsigler/grid/notebooks/raw_dwell_dict.pkl","rb")
    dict_dwell = pickle.load(pickle_dwell)
    
    pickle_arrival = open("/Users/dsigler/grid/notebooks/raw_arrival_rates_dict.pkl","rb")
    dict_arrival = pickle.load(pickle_arrival)
    
           

    return dict_tt, dict_dwell , dict_arrival
##################################################################
##################################################################
def set_parameters(time_window,max_ride,head_way,dwell_reduction, routes ):
    df_parameters = pd.DataFrame(index = [1],columns = ['big m'])
    df_parameters.loc[1,'big m']= 50000
    df_parameters.loc[1,'max ride time']= max_ride
    #df_parameters.loc[1,'dwell constant']= 1/12.0
    #df_parameters.loc[1,'long ride/min'] = 5
    #df_parameters.loc[1,'long wait/min'] = 30
    #df_parameters.loc[1,'leave person behind'] =500
    df_parameters.loc[1,'energy cost'] = .1275
    df_parameters.loc[1,'max bus number'] = 50
    #df_parameters.loc[1,'number of types'] = 5
    df_parameters.loc[1,'headway'] = head_way
    df_parameters.loc[1,'number of routes'] = routes
    df_parameters.loc[1,'start hour'] = int(time_window[0])
    df_parameters.loc[1,'end hour'] = int(time_window[1])
    df_parameters.loc[1,'dwell_reduction'] = dwell_reduction
    df_parameters['start hour'] = df_parameters['start hour'].astype('int')
    df_parameters['end hour'] = df_parameters['end hour'].astype('int')
    return df_parameters


##################################################################
def clean_arrival_data(dict_arrival,df_parameters,day):
    start = df_parameters.loc[1,'start hour']
    end = df_parameters.loc[1,'end hour']
    
    for terminal in dict_arrival.keys():
        for hour in range(start,end + 1):
            d = dict_arrival[terminal][day][hour]
            A = np.array(d)
            B = np.where(A>43, 43, A) 
            dict_arrival[terminal][day][hour] = list(B)

    return dict_arrival
##################################################################
def clean_dwell_data(dict_dwell,df_parameters,day, cuttoff):
    
    start = df_parameters.loc[1,'start hour']
    end = df_parameters.loc[1,'end hour']
    
    for terminal in dict_dwell.keys():
        for hour in range(start,end + 1):
            d = dict_dwell[terminal][day][hour]
            A = [i for i in d if i < cuttoff]
            dict_dwell[terminal][day][hour] = A

    return dict_dwell
##################################################################
def clean_tt_data(dict_tt,df_parameters,day,cuttoff):
    
    start = df_parameters.loc[1,'start hour']
    end = df_parameters.loc[1,'end hour']
    
    for terminal in dict_tt.keys():
        for hour in range(start,end + 1):
            d = dict_tt[terminal][day][hour]
            A = [i for i in d if i < cuttoff]
            dict_tt[terminal][day][hour] = A

    return dict_tt
##################################################################
def create_requests(df_parameters,dict_arrival,day,standard_dev):

    geo_nodes = ['A','B','C','D','E','R']
    number_of_time_slots = 1
    number_of_nodes = (len(geo_nodes) - 1) *  number_of_time_slots *2 
    columns = ['start node','end node','demand']
    #columns = ['start node','end node','pick up start','pick up end','demand','dwell time','time slot']
    df_req = pd.DataFrame(index = range(1,number_of_nodes +1),columns = columns)
    #df_req['dwell time'] = 1

    geo_nodes = ['A','B','C','D','E','R']

    #w = windows[0:number_of_time_slots]
    count = 1
    time_slot = 0
    #for window in w:
        #time_slot = time_slot + 1
    for n1 in geo_nodes:
        n2 = 'R'
        if n1 == n2 :
            pass
        else:

            df_req.loc[count, 'start node'] = n1
            df_req.loc[count, 'end node'] = n2


            df_req.loc[count + 1, 'start node'] = n2
            df_req.loc[count  + 1, 'end node'] = n1
            count = count + 2


    start = df_parameters.loc[1,'start hour']
    end = df_parameters.loc[1,'end hour']
    for i in df_req.index:
        start_node = str(df_req.loc[i,'start node'])
        end_node = str(df_req.loc[i,'end node'])
        name = start_node + end_node
#         s = '('+ "'" + str(df_req.loc[i,'start node']) + "'" + ',' +' ' + "'" + str(df_req.loc[i,'end node']) + "'"  + ')'
#         df_req.loc[i,'demand'] = df_demand_ave.loc[start:end,s].mean()
        d = []
        for hour in range(start,end + 1):
            d += dict_arrival[name][day][hour]

        A = np.array(d)
        val = np.mean(A) + standard_dev*np.std(A)
        
        df_req.loc[i,'demand'] =  val  
    
    return df_req


################################################################## got here
def reformat_requests_data(df_req,dict_dwell,df_parameters,day):
    number_of_time_slots = 1
    geo_nodes = ['A','B','C','D','E','R']
    number_of_nodes = (len(geo_nodes) - 1) *  number_of_time_slots *2 
    #hour = df_parameters.loc[1,'hour']
    #columns = ['node type','geo node','pick up start','pick up end','demand','dwell time', 'time slot']
    columns = ['node type','geo node','demand','dwell time']

    df_nodes = pd.DataFrame(index = range(0,2*number_of_nodes + 2),columns = columns)

    df_nodes.loc[0,'node type'] = 'O'
    df_nodes.loc[range(1,number_of_nodes +1) ,'node type'] = 'P'
    df_nodes.loc[range(number_of_nodes +1,2*number_of_nodes +1) ,'node type'] = 'P'
    df_nodes.loc[range(number_of_nodes +1,2*number_of_nodes +1) ,'node type'] = 'D'
    df_nodes.loc[2*number_of_nodes +1 ,'node type'] = 'O'

    for i in df_req.index:
        df_nodes.loc[i,'geo node'] = df_req.loc[i,'start node']
        df_nodes.loc[i + number_of_nodes ,'geo node'] = df_req.loc[i,'end node']
    df_nodes.loc[0 ,'geo node'] = 'R'#'depot'
    df_nodes.loc[2*number_of_nodes +1 ,'geo node'] = 'R' #'depot'



    for i in df_req.index:
        df_nodes.loc[i,'demand'] = df_req.loc[i,'demand']
        df_nodes.loc[i + number_of_nodes ,'demand'] = -df_req.loc[i,'demand']
    df_nodes.loc[0 ,'demand'] = 0
    df_nodes.loc[2*number_of_nodes +1 ,'demand'] = 0
    
    
    
    start = df_parameters.loc[1,'start hour']
    end = df_parameters.loc[1,'end hour']
    
    dr = df_parameters.loc[1,'dwell_reduction']
    
    for i in df_nodes.index:
        geo_node = df_nodes.loc[i,'geo node']
        d = []
        for hour in range(start,end + 1):
            d += dict_dwell[geo_node][day][hour]

        A = np.array(d)
        val = np.median(A)
        
        df_nodes.loc[i,'dwell time'] = val/2.0 * dr
        #df_nodes.loc[i,'dwell time'] = (df_dwell.loc[start:end,geo_node].mean())/2.0 * dr
    df_nodes.loc[0 ,'dwell time'] = 0
    df_nodes.loc[2*number_of_nodes +1 ,'dwell time'] = 0   
        


    return df_nodes


##################################################################
def create_tt_matrix(df_nodes,df_parameters,dict_tt,day):

    geo_nodes = ['A','B','C','D','E','R']
    start = df_parameters.loc[1,'start hour']
    end = df_parameters.loc[1,'end hour']
    df_time = pd.DataFrame(columns=geo_nodes, index=geo_nodes,data = 0)
    for n1 in geo_nodes:
        for n2 in geo_nodes:
            if n1 == n2:
                pass
            else:
                name = n1 + n2
                d = []
                for hour in range(start,end + 1):
                    d += dict_tt[name][day][hour]

                A = np.array(d)
                val = np.median(A)

                df_time.loc[n1,n2] =  val   



    
    df_tt = pd.DataFrame(index = df_nodes.index, columns = df_nodes.index)
    for i in df_tt.index:
        for j in df_tt.index:

            start_node = df_nodes.loc[i,'geo node']
            end_node = df_nodes.loc[j,'geo node']

            df_tt.loc[i,j] = df_time.loc[start_node,end_node]  # time_dict[time_slot].loc[start_node,end_node]

            if i == j:
                df_tt.loc[i,j] = 10000

    return df_tt


##################################################################
# NOT NEEDED IN THE NEW MODEL
# def create_distance_matrix(df_sumo_dist,df_nodes):
#     geo_nodes = ['A','B','C','D','E','R']
#     data = df_sumo_dist.set_index('Unnamed: 0',drop=True).values

#     df_distance = pd.DataFrame(columns=geo_nodes, index=geo_nodes,data = data)
#     df_dist = pd.DataFrame(index = df_nodes.index, columns = df_nodes.index)
#     for i in df_dist.index:
#         for j in df_dist.index:


#             start_node = df_nodes.loc[i,'geo node']
#             end_node = df_nodes.loc[j,'geo node']

#             df_dist.loc[i,j] = df_distance.loc[start_node,end_node]

#             if i == j:
#                 df_dist.loc[i,j] = 10000

#     return df_dist 


##################################################################
my_template = jin.Template("""
{% for key, values in sets.iteritems() %}
set {{key}} :=
{% for x in values %}{{x}}
{% endfor %};
{% endfor %}
{% for key, value in values.iteritems() %}
param {{key}} := {{ value }};
{% endfor %}
{% for key, values in lists.iteritems() %}
param {{key}} :=
{% for x,y in values %}{{x}}    {{y}}
{% endfor %};
{% endfor %}
""")


##################################################################

scenario_template = jin.Template("""
set Stages := FirstStage SecondStage ;

set Nodes := RootNode 
             {% for x in nodes %}{{x}}Node
             {% endfor %};

param NodeStage := RootNode FirstStage
                   {% for x in nodes %}{{x}}Node SecondStage
                   {% endfor %};
                   
set Children[RootNode] :=  {% for x in nodes %}{{x}}Node
                          {% endfor %};       
                         
param ConditionalProbability := RootNode 1.0
                                {% for key, value in probs.iteritems() %}{{key}}Node {{value}}
                                {% endfor %};
                               
set Scenarios := {% for x in nodes %}{{x}}Scenario
                 {% endfor %};
                 
param ScenarioLeafNode := {% for x in nodes %}{{x}}Scenario {{x}}Node
                 {% endfor %};
                 
set StageVariables[FirstStage] := {% for x in varsFirstStage %}{{x}}
                                  {% endfor %};
                                  
set StageVariables[SecondStage] := {% for x in varsSecondStage %}{{x}}
                                   {% endfor %};
                                  
param StageCost := FirstStage  FirstStageCost
                   SecondStage SecondStageCost ;

""")



##################################################################
##################################################################
def create_dfw_bus_data():
    index = range(1,2)

    E_list = [  2+ i * .2 for i in range(1,6)]
    size_list = [ 10 + i * 10 for i in range(1,6)]

    df_buses = pd.DataFrame(index = index)
    df_buses['capacity'] = 43
    df_buses['gallons/mile'] = .2
    df_buses['$/gallon'] = 2.20
    df_buses['O&M in $/mile'] = 0.88
    return df_buses

##################################################################
def create_bus_types_data():
    index = range(1,6)
    E_list = [  2+ i * .2 for i in range(1,6)]
    size_list = [ 10 + i * 10 for i in range(1,6)]

    df_buses = pd.DataFrame(index = index)
    df_buses['capacity'] = size_list
    df_buses['kwhr/mile'] = E_list

    return df_buses

##################################################################
def bus_types_data():
    index = range(1,3)
    E_list = [ 45.3, 32.34]
    size_list = [ 43, 14]

    df_buses = pd.DataFrame(index = index)
    df_buses['capacity'] = size_list
    df_buses['kwhr/hr'] = E_list

    return df_buses

##################################################################
# def generate_pyomo_data(network_dict,df_net,df_capacities,df_OD,df_FTT,scenario_tuple,scenario_number):
def generate_pyomo_data(df_parameters,df_nodes,df_buses,df_tt,dir_name,file_name):    
    # sets
    sets={}
    sets['P'] = df_nodes[df_nodes['node type'] == 'P'].index # set of pick up nodes
    sets['D'] = df_nodes[df_nodes['node type'] == 'D'].index # set of drop off nodes
    sets['R'] = range(1,int(df_parameters.loc[1,'number of routes']) +1 )#  this has become the set of routes instead of busess
    sets['N'] = df_nodes.index # set of total nodes
    sets['S'] = range(1,int(df_parameters.loc[1,'max bus number']) + 1) # set of different numbers of buses you can have on routes
    sets['Ty'] = df_buses.index # set of different type/sizes of buses we are considering
 
    # values
    values={}
    values['M'] = df_parameters.loc[1,'big m'] # param for big M constraints 
    values['max_ride_time'] = df_parameters.loc[1,'max ride time'] # max ride time for a passengers
    #values['alpha'] = df_parameters.loc[1,'dwell constant'] # load time per passenger
    #values['LR'] = df_parameters.loc[1,'long ride/min'] # long ride penalty / not currently used
    #values['LW'] = df_parameters.loc[1,'long wait/min'] # long wait penalty / not currently used
    #values['LP'] = df_parameters.loc[1,'leave person behind'] # leave person behind penalty / not currently used
    #values['c'] = df_parameters.loc[1,'energy cost'] # energy cost per kw/hr 
   # values['c'] = df_buses.loc[1,'$/gallon'] # cost per mile 
    #values['o_and_m'] = df_buses.loc[1,'O&M in $/mile']
    
   
    values['r'] = int(len(sets['P'])) # number of pickup nodes
    values['HW'] = df_parameters.loc[1,'headway'] # allowed max headway
    
    # lists
    lists = {}

    demand_rate = values['HW']/60.0
    lists['Q'] = zip(sets['Ty'],list(df_buses.loc[sets['Ty'],'capacity'])) # capacity of different bus types
    lists['q'] = zip(sets['N'],list(df_nodes.loc[sets['N'],'demand']))#*demand_rate)) # demand at different nodes
    #lists['a'] = zip(sets['P'],list(df_nodes.loc[sets['P'],'pick up start'])) # not relevant in current model
    #lists['b'] = zip(sets['P'],list(df_nodes.loc[sets['P'],'pick up end'])) # not relevant in current model
    
    

    #lists['E'] = zip(sets['Ty'],df_buses.loc[sets['Ty'],'kwhr/mile']) # energy consumption per mile for each bus type (kwhr/per mile)
    lists['E'] = zip(sets['Ty'],df_buses.loc[sets['Ty'],'kwhr/hr']) # energy consumption per hour for each bus type
    #lists['size'] = zip(sets['T'],size_list)
    
    lists['s'] = zip(sets['N'],list(df_nodes.loc[sets['N'],'dwell time']/60.0)) # dwell time at nodes
    lists['n'] = zip(sets['S'],sets['S']) # number of buses param indexed by integers

                                  

    
    tt_array = (df_tt/60.0).values
    tt_list=[]
    for i in tt_array:
        tmp = ' '.join(map(str, i))
        tt_list.append(tmp)

    name = 'tt: ' + ' '.join(str(x) for x in sets['N']) # travel times on links
    tt_list = zip(sets['N'],tt_list)
    lists[name] = tt_list
    
#     dist_array = df_dist.values
#     dist_list=[]
#     for i in dist_array:
#         tmp = ' '.join(map(str, i))
#         dist_list.append(tmp)

#     name = 'dist: ' + ' '.join(str(x) for x in sets['N']) # distances on links
#     dist_list = zip(sets['N'],dist_list)
#     lists[name] = dist_list





    string = file_name
    # render the jinja template
    with open(os.path.join(dir_name, string + '_Scenario.dat'), 'w') as outfile:
        outfile.write(my_template.render(lists=lists, values=values, sets=sets))


    
    return


##################################################################
def generate_data_file(time_window,day,max_ride,head_way,dir_name,file_name,
                       dwell_reduction,cuttoff_dwell,cuttoff_tt,standard_dev,routes):
    # read in data
    dict_tt, dict_dwell , dict_arrival = read_data()
    
    # set parameters 
    df_parameters = set_parameters(time_window,max_ride,head_way,dwell_reduction, routes ) # should be ok
    
    
    #clean outliers from the data
    dict_tt = clean_tt_data(dict_tt,df_parameters,day,cuttoff_tt)
    dict_dwell = clean_dwell_data(dict_dwell,df_parameters,day, cuttoff_dwell)
    #dict_arrival =  clean_arrival_data(dict_arrival,df_parameters,day)
 
    #create requests
    df_req = create_requests(df_parameters,dict_arrival,day,standard_dev)
    
    # turn requests into a network of nodes to be served
    df_nodes = reformat_requests_data(df_req,dict_dwell,df_parameters,day)
    
    # create travel time matrix for the set of  request nodes 
    df_tt = create_tt_matrix(df_nodes, df_parameters,dict_tt,day) # this should be a much simplier function
    
    # create distance matrix for the set of request notes
    #df_dist = create_distance_matrix(df_sumo_dist,df_nodes)
    
    # create a set of buses to choose from
    #df_buses = create_bus_types_data()
    #df_buses = create_dfw_bus_data()
    df_buses = bus_types_data()
    
    # create data file
    generate_pyomo_data(df_parameters,df_nodes,df_buses,df_tt,dir_name,file_name) # remove df_dist
    
    return df_nodes
    


##################################################################


##################################################################



##################################################################
def make_VR_model():

    model= AbstractModel()


    # model sets:
    model.P = Set () # set of pick ups
    model.D = Set () # set of drop offs
    model.R = Set () # set of routees ( really set of route routes)
    model.N = Set () # set of network nodes
    model.S = Set () # possible numbers of busees on a route
    model.Ty = Set () # type of buses
    model.P_union_D = model.P | model.D
#model.W_union_W_new = model.W | model.W_new

    # model parameters that depend on sets
    model.Q = Param(model.Ty) # route cap
    model.q = Param(model.N) # demand at a node
    #model.a = Param(model.P) # start of pick up window at a node
    #model.b = Param(model.P) # end of pick up window at node
    model.E = Param(model.Ty) # Energy consumption per mile 
    model.s = Param(model.N) # dwell time at a node
    model.tt = Param(model.N,model.N) # matrix of travel times between nodes 
   # model.dist = Param(model.N,model.N) # matrix of travel times between nodes
    model.n = Param(model.S)
   
    
    # model parameters that DON'T depend on sets:
    model.M = Param() # big M constant
    model.max_ride_time = Param() # max time a passaenger can be on a route               
    model.alpha = Param() # loading time constant
    #model.LR = Param() # long ride penalty 
    #model.LW = Param() # long wait penalty 
    #model.LP = Param() # leave people behind
    #model.c = Param() # cost of energy
    model.r = Param() # number of requests
    model.HW = Param()
    #model.o_and_m = Param() # outage and maintaince cost  per mile


    # variables (continuous):
    model.z = Var(model.N,model.N,model.R, domain=Binary)
    model.x = Var(model.N,model.N,model.R, domain=Binary)
    model.y = Var(model.S,model.R, domain=Binary)
    model.w = Var(model.Ty,model.R, domain=Binary)
    model.T = Var(model.N,model.R, domain=NonNegativeReals)
    model.Q_track = Var(model.N,model.R, domain=NonNegativeReals)
    #model.S_R = Var(model.N, domain=NonNegativeReals)
    #model.S_W = Var(model.N, domain=NonNegativeReals)
    #model.S_Q = Var(model.N, domain=NonNegativeReals)
    model.t = Var(model.Ty, model.S,model.R, domain=NonNegativeReals) # auxilary variable 
    model.nv = Var(model.R,domain = NonNegativeIntegers)
    model.size = Var(model.R,domain= NonNegativeIntegers)
    model.dynamic_q = Var(model.N,model.R)
    
    #model.Day_Cost = Var()


    # objective function ###########################################################
    

    def total_cost(model):
        obj_1 = 0.0

        # expansion sum
        for route in model.R:
            for i in model.S:
                for type_ in model.Ty:
                    obj_1 += model.t[type_,i,route] 

        return  obj_1 #+ obj_2 + obj_3 + obj_4 

    model.Total_Cost = Objective(rule=total_cost, sense=minimize)

    # first and second stage costs ################################################
    
    
    # Model Constraints ############################################################
        
    def t_aux(model,type_,s,route):
        obj_1 = 0.0
        obj_2 = 0.0
        obj_3 = 0.0
        obj_4 = 0.0
        # expansion sum
        obj_1 = model.E[type_] * model.n[s]
                
        
#         for i in model.N:
#             for j in model.N:
#                 obj_1 += model.c * model.E[type_]*model.dist[i,j] * model.x[i,j,route] * model.n[s]
#                 obj_1 += model.o_and_m * model.dist[i,j] * model.x[i,j,route] * model.n[s]


        #for node in model.P:
            #obj_2 += model.LW * model.q[node] * model.S_W[node]
            
        #for node in model.P:
         #   obj_3 += model.LP * model.S_Q[node]
            
        #for node in model.P:
            #obj_4 += model.LR * model.q[node] * model.S_R[node]


        return - model.M*(2 - model.y[s,route] - model.w[type_,route]) + obj_1 <= model.t[type_,s,route]
        #return -100 * (1 - model.y[s,route]) + obj_1 <= model.t[s,route]
    
    model.con_t_aux = Constraint(model.Ty,model.S,model.R,rule=t_aux)
    
    ########################
    def y_sum(model,route):
        obj = 0.0
        for s in model.S:
            obj += model.y[s,route]
        return obj <= 1
    
    model.con_y_sum = Constraint(model.R, rule = y_sum) 
    #######################
    def y_encode(model,route):
        obj = 0.0
        for s in model.S:
            obj += model.y[s,route] * model.n[s]
        return obj == model.nv[route]
    
    model.con_y_encode= Constraint(model.R, rule = y_encode) 
    #######################
    def test_con(model):
        return model.nv[2] == 0
        #return model.x[0,21,1] == 1
    #model.con_test_con = Constraint(rule = test_con)
    #######################
    def route_number(model,route):
        k= 2*model.r + 1
        return (model.T[k,route]/model.HW) - .01 <= model.nv[route]
    
    model.con_route_number = Constraint(model.R, rule = route_number) 
    #######################
    def w_sum(model,route):
        obj = 0.0
        for type_ in model.Ty:
            obj += model.w[type_,route]
        return obj == 1
    
    model.con_w_sum = Constraint(model.R, rule = w_sum) 
    #######################    
    def route_size(model,route):
        obj = 0.0
        for type_ in model.Ty:
            obj += model.w[type_,route] * model.Q[type_]
        return obj == model.size[route]
    model.con_route_size= Constraint(model.R, rule = route_size)    
###################################


    def no_loops_z(model,i,route):

        return model.z[i,i,route] == 0
    
    model.con_no_loops_z = Constraint(model.N,model.R, rule = no_loops_z) 
    
    #######################
    def x_and_z(model,i,j,route):

        return model.x[i,j,route] <= model.z[i,j,route] 
    
    model.con_x_and_z = Constraint(model.N, model.N,model.R, rule = x_and_z ) 

    #######################
    def sub_tour_1(model,i,j,route):

        
        if i != j:
            
            obj_1 = 0
            obj_2 = 0

            for ii in model.N:
                obj_1  += model.x[j,ii,route]
            for jj in model.N:
                obj_2  += model.x[i,jj,route]
            return model.z[i,j,route] + model.z[j,i,route]  >= obj_1 + obj_2 - 1
        
        else:
            return Constraint.Skip
    
    model.con_sub_tour_1 = Constraint(model.N, model.N,model.R, rule = sub_tour_1 ) 
    
    def sub_tour_1_prime(model,i,j,route):

        
        if i != j:
            
            return model.z[i,j,route] + model.z[j,i,route]  <= 1
        
        else:
            return Constraint.Skip
    
    model.con_sub_tour_1_prime = Constraint(model.N, model.N,model.R, rule = sub_tour_1_prime ) 

    ########################
    def sub_tour_2(model,i,j,k,route):

        return model.z[i,j,route] + model.z[j,k,route] + model.z[k,i,route] <= 2
    
    model.con_sub_tour_2 = Constraint(model.N, model.N,model.N,model.R, rule = sub_tour_2 ) 
    ###########################################################
    
    def one_pickup_and_only_one(model,i):
        obj = 0
        for route in model.R:
            for j in model.N:
                obj += model.x[i,j,route]
        return obj == 1
    model.con_one_pickup_and_only_one = Constraint(model.P, rule = one_pickup_and_only_one )         
    ########################
    def pick_up_drop_off_same_vehicle(model,i,route):
        obj = 0
        k = i + model.r
        for j in model.N:
                obj +=  model.x[i,j,route] - model.x[k,j,route]
        return obj == 0
    model.con_pick_up_drop_off_same_vehicle = Constraint(model.P,model.R, rule = pick_up_drop_off_same_vehicle )     
    
    ########################
    def all_leave_depo(model,route):
        obj = 0
        for j in model.N:
                obj +=  model.x[0,j,route] 
        return obj == 1
    model.con_all_leave_depo = Constraint(model.R, rule = all_leave_depo )     
        
    ########################
    def if_enter_must_leave(model,i,route):
        obj_1 = 0
        obj_2 = 0
        for j in model.N:
            obj_1 +=  model.x[j,i,route] 
            
        for j in model.N:
            obj_2 +=  model.x[i,j,route] 
        
        return obj_1 - obj_2 == 0
    model.con_if_enter_must_leave = Constraint(model.P_union_D, model.R, rule = if_enter_must_leave )    
    
    ########################
    def all_return_depo(model,route):
        obj = 0
        k= 2*model.r + 1
        for i in model.N:
                obj +=  model.x[i,k,route] 
        return obj == 1
    model.con_all_return_depo = Constraint(model.R, rule = all_return_depo )  
    
    #######################
    def service_times(model,i,j,route):
        
        return model.T[j,route] >= (model.T[i,route] + model.s[i] + model.tt[i,j]) - model.M*(1 - model.z[i,j,route])
    
    model.con_service_times = Constraint(model.N,model.N,model.R, rule = service_times )  
        
    #######################
    def precedence(model,i,route):

        return model.T[i + model.r,route] - (model.T[i,route] + model.s[i] + model.tt[i,i + model.r]) >= 0
    
    model.con_precedence = Constraint(model.P,model.R, rule = precedence )  
    
    
#     #######################
#     def time_window_lower(model,i,route):

#         return model.a[i] <= model.T[i,route]
    
#     model.con_time_window_lower = Constraint(model.P,model.R, rule = time_window_lower) 
        
#     ########################
#     def time_window_upper(model,i,route):

#         return  model.T[i,route] <= model.b[i] #+ model.S_W[i]
    
#     model.con_time_window_upper = Constraint(model.P,model.R, rule = time_window_upper) 
    
    #######################
    def dynamic_demand(model,i,route,s):
        k= 2*model.r + 1
        return - model.M*(1 - model.y[s,route]) + ((model.T[k,route]*model.q[i])/(model.n[s] * 60.0)) <= model.dynamic_q[i,route]
        
    model.con_dynamic_demand = Constraint(model.N, model.R, model.S, rule = dynamic_demand)
        
#         if s!=0:
        
#             return - model.M*(1 - model.y[s,route]) + ((model.T[k,route]*model.q[i])/(model.n[s] * 60.0)) <= model.dynamic_q[i,route]
#         else:
#             return - model.M*(1 - model.y[s,route]) + model.q[i]*100 <= model.dynamic_q[i,route]
            
#     model.con_dynamic_demand = Constraint(model.N, model.R, model.S, rule = dynamic_demand)
    
#     #######################
    def dynamic_vehicle_capacity(model,i,j,route):
        
        return model.Q_track[j,route] >= (model.Q_track[i,route]+ model.dynamic_q[j,route]) - model.M*(1 - model.x[i,j,route])
    
    model.con_dynamic_vehicle_capacity = Constraint(model.N,model.N,model.R, rule = dynamic_vehicle_capacity )     

    #######################
    def conservation_of_mass(model,i,route):
    
        return model.dynamic_q[i,route] + model.dynamic_q[model.r + i,route] == 0
    
    model.conservation_of_mass = Constraint(model.P,model.R, rule = conservation_of_mass) 
    
    #######################
    def capacity_1_lower(model,i,route):

        return 0 <= model.Q_track[i,route]
    
    model.con_capacity_1_lower = Constraint(model.N,model.R, rule = capacity_1_lower) 
        
    ########################
    def capacity_1_upper(model,i,route):

        return  model.Q_track[i,route] <= model.size[route] + model.dynamic_q[i,route] #+ model.S_Q[i]
    
    model.con_capacity_1_upper = Constraint(model.N,model.R, rule = capacity_1_upper)    
    
    #######################
    def capacity_2_lower(model,i,route):

        return model.dynamic_q[i,route] <= model.Q_track[i,route]
    
    model.con_capacity_2_lower = Constraint(model.N,model.R, rule = capacity_2_lower) 
        
    ########################
    def capacity_2_upper(model,i,route):

        return  model.Q_track[i,route] <= model.size[route] #+ model.S_Q[i]
    
    model.con_capacity_2_upper = Constraint(model.N,model.R, rule = capacity_2_upper)    
    
    #######################
    def trip_time(model, i, route):
        obj = 0
        for j in model.N:
            obj += model.x[i,j,route]
            
        return model.T[model.r + i,route] - (model.T[i,route] + model.s[i]) <= model.max_ride_time + (1 - obj)*model.M #+ model.S_R[i]
    model.con_trip_time = Constraint(model.P,model.R, rule = trip_time) 
    ########################
    def cant_leave_depo(model,i,route):
        k = 2*model.r + 1
        
        return model.x[k,i,route] == 0
    
    model.con_cant_leave_depo = Constraint(model.N,model.R, rule = cant_leave_depo) 
    
    
    ########################
    def cant_return_to_origin(model,i,route):

        return model.x[i,0,route] == 0
    
    model.con_cant_return_to_origin = Constraint(model.N,model.R, rule = cant_return_to_origin) 

#     ########################
    def no_loops(model,i,route):

        return model.x[i,i,route] == 0
    
    model.con_no_loops = Constraint(model.N,model.R, rule = no_loops) 
    
    #     ########################
    def no_origin_to_dropoff(model,i,route):

        return model.x[0,i,route] == 0
    
    #model.con_no_origin_to_dropoff = Constraint(model.D,model.R, rule = no_origin_to_dropoff) 
    
    #     ########################
    
    def no_pickup_to_depo(model,i,route):
        m = max(model.N)
        return model.x[i,m,route] == 0
    
    #model.con_no_pickup_to_depo = Constraint(model.P,model.R, rule = no_pickup_to_depo)
    def drop_then_pick_up(model,i,route):
        obj = 0
        k = 1 + model.r
        if i % 2 == 0:
            for j in model.N:
                obj +=  model.x[j,i,route]
            return obj == model.x[i,i - k,route]
        else:
            return Constraint.Skip
    
    model.con_drop_then_pick_up = Constraint(model.D,model.R, rule = drop_then_pick_up)
    
    #################################
    def upper_bound(model):
        obj_1 = 0.0

        for route in model.R:
            for i in model.S:
                for type_ in model.Ty:
                    obj_1 += model.t[type_,i,route] 

        return obj_1 <= 800
    #model.con_upper_bound = Constraint(rule = upper_bound)
    
    return model 





##################################################################
def run_VR_model(model_data_tuple,time_limit):
    solution_dir , model_data = model_data_tuple
    #solution_dir = '/Users/dsigler/ATHENA-siem-sumo/siem/VR_solutions'
    #solver = SolverFactory("gurobi")
    solver = SolverFactory("xpress")
    model = make_VR_model()
    model_instance = model.create_instance(model_data)
    solver.options['mipgap'] = '.001'
    #solver.options['bestobjstop'] = 38
    solver.options['MAXTIME'] = str(time_limit)# xpress
    #solver.options['MIPFocus'] = '3'
    #solver.options['TimeLimit'] = '3600' 
    #solver.options['ConcurrentMIP'] = '3'
    model_instance.preprocess()
    results = solver.solve(model_instance)#,tee=True)
    #instance.solutions.load_from(results)
    #instance.display() # shows everything from solution
    save_solution(model_data,solution_dir,model_instance)
    save_solver_status(model_data,solution_dir,results)
    print model_data
    return model_instance,results

##################################################################
def save_solver_status(model_data,solution_dir,results):
    name = os.path.split(model_data)[1].strip('.dat') + '_solution_status.csv'
    loc = os.path.join(solution_dir, name)
    #print loc
    
    
    df = pd.Series(index = [1,2])
    status = (results.solver.termination_condition == TerminationCondition.optimal)
    df[1] = status
    df[2] = results.solver.termination_condition
    df.to_csv(loc)
    return

##################################################################

def save_solution(model_data,solution_dir,model_instance):
    name = os.path.split(model_data)[1].strip('.dat') + '.pkl'
    loc = os.path.join(solution_dir, name)
    #print name
    #print loc
    with open(loc, mode='wb') as file:
       cloudpickle.dump(model_instance, file)

    return
##################################################################


##################################################################


##################################################################





    