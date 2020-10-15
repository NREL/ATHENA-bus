""" Author: Devon Sigler
    Purpose: Provides tools for constructing the vehicle routing optimization model, processing data and constructing data files for the optimization model, solving an instance of the optimization model, saving and postprocessing optimzation model solutions.
"""

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
import os
import sys
from pyomo.environ import *
from pyomo.pysp.ef import create_ef_instance
from pyomo.opt import SolverFactory
from pyutilib.misc.config import ConfigBlock
from pyomo.opt import SolverStatus, TerminationCondition
import pickle
import cloudpickle


# defining jinja2 template we use to write .dat files out for pyomo
#################################################################
my_template = jin.Template("""
{% for key, values in sets.items() %}
set {{key}} :=
{% for x in values %}{{x}}
{% endfor %};
{% endfor %}
{% for key, value in values.items() %}
param {{key}} := {{ value }};
{% endfor %}
{% for key, values in lists.items() %}
param {{key}} :=
{% for x,y in values %}{{x}}    {{y}}
{% endfor %};
{% endfor %}
""")

##################################################################

# defining functions
#############################################################
def read_parameters(path_to_parameters_csv):
    """ This function reads in the model parameter csv provided by the user so it can be used to construct at data file for the optimization model. """
    df_parameters = pd.read_csv(path_to_parameters_csv,index_col=0)
    return df_parameters
#############################################################
def read_dwell_times(path_to_dwell_time_csv):
    """ This function reads in the dwell times csv provided by the user so it can be used to construct at data file for the optimization model """
    dwell_df = pd.read_csv(path_to_dwell_time_csv,index_col=0)
    return dwell_df
###########################################################
def read_requests(path_to_requests_csv):
    """ This function reads in the ride requests csv provided by the user so it can be used to construct at data file for the optimization model """
    df_req = pd.read_csv(path_to_requests_csv,index_col=0)
    return df_req
###############################################################
def read_bus_types_data(path_to_bus_type_csv):
    """ This function reads in the bus types csv provided by the user  so it can be used to construct at data file for the optimization model """
    df_buses = pd.read_csv(path_to_bus_type_csv, index_col=0)

    return df_buses

################################################################
def reformat_requests_data(df_req,df_dwell_times,df_parameters):
    """ This function produces a dataframe called df_nodes that is used to construct the data file for an instance of the optimization model. It uses the requests, dwell times, and parameters dataframes to do so. This function is for internal use and not meant to be called by the user. """
    number_of_time_slots = 1
    geo_nodes = list(df_dwell_times.index)
    number_of_nodes = (len(geo_nodes) - 1) *  number_of_time_slots *2

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


    dr = df_parameters.loc[1,'dwell_reduction']

    for i in df_nodes.index:
        geo_node = df_nodes.loc[i,'geo node']
        val = df_dwell_times.loc[geo_node,'dwell time']
        df_nodes.loc[i,'dwell time'] = val/2.0 * dr

    df_nodes.loc[0 ,'dwell time'] = 0
    df_nodes.loc[2*number_of_nodes +1 ,'dwell time'] = 0

    return df_nodes



##################################################################
def create_tt_matrix(df_nodes,df_parameters,path_to_travel_time_csv):
    """ This function reads in the travel time csv provided by the user and reformats it so it can be used to construct at data file for the optimization model """

    df_time = pd.read_csv(path_to_travel_time_csv,index_col=0)

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
def generate_pyomo_data(df_parameters,df_nodes,df_buses,df_tt,data_dir_name,data_file_name):
    """ This function takes in a set of dataframes and uses them to construct the pyomo .dat file for to construct an instance of the optimization model. It takes in a path to the directory where the .dat file is written to, and the name the .dat file should be saved under. """
    # sets for template
    sets={}
    sets['P'] = df_nodes[df_nodes['node type'] == 'P'].index # set of pick up nodes
    sets['D'] = df_nodes[df_nodes['node type'] == 'D'].index # set of drop off nodes
    sets['R'] = range(1,int(df_parameters.loc[1,'number of routes']) +1 )#  this has become the set of routes instead of busess
    sets['N'] = df_nodes.index # set of total nodes
    sets['S'] = range(1,int(df_parameters.loc[1,'max bus number']) + 1) # set of different numbers of buses you can have on routes
    sets['Ty'] = df_buses.index # set of different type/sizes of buses we are considering

    # values for template
    values={}
    values['M'] = df_parameters.loc[1,'big m'] # param for big M constraints
    values['max_ride_time'] = df_parameters.loc[1,'max ride time'] # max ride time for a passengers
    values['r'] = int(len(sets['P'])) # number of pickup nodes
    values['HW'] = df_parameters.loc[1,'headway'] # allowed max headway

    # lists for template
    lists = {}

    demand_rate = values['HW']/60.0
    lists['Q'] = zip(sets['Ty'],list(df_buses.loc[sets['Ty'],'capacity'])) # capacity of different bus types
    lists['q'] = zip(sets['N'],list(df_nodes.loc[sets['N'],'demand']))#*demand_rate)) # demand at different nodes
    lists['E'] = zip(sets['Ty'],df_buses.loc[sets['Ty'],'kwhr/hr']) # energy consumption per hour for each bus type
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

    #string = data_file_name
    # render the jinja template
    with open(os.path.join(data_dir_name,data_file_name), 'w') as outfile:
        outfile.write(my_template.render(lists=lists, values=values, sets=sets))

    return


##################################################################
def generate_data_file(pyomo_data_dir_name,pyomo_data_file_name,path_to_travel_time_csv, path_to_bus_type_csv,path_to_requests_csv,path_to_dwell_time_csv,path_to_parameters_csv):
    """ This function reads in paths to all the data needed to construct a specific instance of the pyomo model. It uses these paths to read in the data, reformat it, and construct and save a .dat file for the pyomo model. This function takes in a path to the directory where the .dat file is written to, and the name the .dat file should be saved under. A copy of the df_nodes csv is also saved in this location so it can be uses in the post processing of solution results."""


    # read in data from csv files
    df_parameters = read_parameters(path_to_parameters_csv) # should be ok
    df_dwell_times = read_dwell_times(path_to_dwell_time_csv)
    df_req = read_requests(path_to_requests_csv)
    df_buses = read_bus_types_data(path_to_bus_type_csv)

    # turn requests into a network of nodes to be served
    df_nodes = reformat_requests_data(df_req,df_dwell_times,df_parameters)

    # create travel time matrix for the set of  request nodes
    df_tt = create_tt_matrix(df_nodes, df_parameters,path_to_travel_time_csv) # this should be a much simplier function

    # create data file for pyomo
    generate_pyomo_data(df_parameters,df_nodes,df_buses,df_tt,pyomo_data_dir_name,pyomo_data_file_name) # remove df_dist
    df_nodes.to_csv(pyomo_data_dir_name + "/" + 'df_nodes.csv')

    return


##################################################################
def make_VR_model():
    """ This function constructs and returns a pyomo model for the vehicle routing problem this repo is focuses on solving """

    model= AbstractModel()


    # model sets:
    model.P = Set () # set of pick ups
    model.D = Set () # set of drop offs
    model.R = Set () # set of routees ( really set of route routes)
    model.N = Set () # set of network nodes
    model.S = Set () # possible numbers of busees on a route
    model.Ty = Set () # type of buses
    model.P_union_D = model.P | model.D


    # model parameters that depend on sets
    model.Q = Param(model.Ty) # route cap
    model.q = Param(model.N) # demand at a node
    model.E = Param(model.Ty) # Energy consumption per mile
    model.s = Param(model.N) # dwell time at a node
    model.tt = Param(model.N,model.N) # matrix of travel times between nodes
    model.n = Param(model.S)


    # model parameters that DON'T depend on sets:
    model.M = Param() # big M constant
    model.max_ride_time = Param() # max time a passaenger can be on a route
    model.alpha = Param() # loading time constant
    model.r = Param() # number of requests
    model.HW = Param()

    # variables (continuous):
    model.z = Var(model.N,model.N,model.R, domain=Binary)
    model.x = Var(model.N,model.N,model.R, domain=Binary)
    model.y = Var(model.S,model.R, domain=Binary)
    model.w = Var(model.Ty,model.R, domain=Binary)
    model.T = Var(model.N,model.R, domain=NonNegativeReals)
    model.Q_track = Var(model.N,model.R, domain=NonNegativeReals)
    model.t = Var(model.Ty, model.S,model.R, domain=NonNegativeReals) # auxilary variable
    model.nv = Var(model.R,domain = NonNegativeIntegers)
    model.size = Var(model.R,domain= NonNegativeIntegers)
    model.dynamic_q = Var(model.N,model.R)


    # objective function ###########################################################

    def total_cost(model):
        obj_1 = 0.0

        # expansion sum
        for route in model.R:
            for i in model.S:
                for type_ in model.Ty:
                    obj_1 += model.t[type_,i,route]

        return  obj_1

    model.Total_Cost = Objective(rule=total_cost, sense=minimize)

    # Model Constraints ############################################################

    def t_aux(model,type_,s,route):
        obj_1 = 0.0
        obj_2 = 0.0
        obj_3 = 0.0
        obj_4 = 0.0
        # expansion sum
        obj_1 = model.E[type_] * model.n[s]

        return - model.M*(2 - model.y[s,route] - model.w[type_,route]) + obj_1 <= model.t[type_,s,route]


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

    #######################
    def dynamic_demand(model,i,route,s):
        k= 2*model.r + 1
        return - model.M*(1 - model.y[s,route]) + ((model.T[k,route]*model.q[i])/(model.n[s] * 60.0)) <= model.dynamic_q[i,route]

    model.con_dynamic_demand = Constraint(model.N, model.R, model.S, rule = dynamic_demand)

    #######################
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

        return model.T[model.r + i,route] - (model.T[i,route] + model.s[i]) <= model.max_ride_time + (1 - obj)*model.M

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
def run_VR_model(pyomo_data_dir_name,pyomo_data_file_name,solver_name,mip_gap = 0.001, max_time = 60,verbose_output = True):
    """ This function solve the vehicle routing pyomo model for a given .dat file, and saves the results. It reads in the location and name of the .dat file, and can take in some basic solver keyword arguements. The function returns a dictionary with the pyomo model solution, and the df_nodes dataframe, which is used in some of the postprocessing functions."""

    model_data_file_path  = os.path.join(pyomo_data_dir_name,pyomo_data_file_name)
    solver = SolverFactory(solver_name)
    model = make_VR_model()
    instance = model.create_instance(model_data_file_path)
    if solver_name == 'xpress':
        solver.options['mipgap'] = mip_gap
        solver.options['MAXTIME'] = str(max_time)# xpress
    if solver_name == 'glpk':
        solver.options["tmlim"] = max_time
        solver.options['mipgap'] = mip_gap
    instance.preprocess()
    if verbose_output == True:
        results = solver.solve(instance,tee=True)
    else:
        results = solver.solve(instance,tee=False)
    solution = {}
    solution['pyomo model object'] = instance
    solution['geographic mapping'] = pd.read_csv(os.path.join(pyomo_data_dir_name,"df_nodes.csv"),index_col=0)

    return solution

##################################################################
def save_solver_status(model_data,solution_dir,results):
    """ This function allows the user to report the status of the solution from the solver. In particular it lets one know if their model was solved to optimality."""
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

def save_solution(solution_name,solution_directory_path,model_instance):
    """ This function allows the user to pickle the solution to a solved pyomo model so it can be analyzed in a seperate context."""
    loc = os.path.join(solution_directory_path,solution_name)

    with open(loc, mode='wb') as file:
       cloudpickle.dump(model_instance, file)

    return
##################################################################

def read_pickle(path,pickled_file_name):
    """ This function allows the user to unpickle a saved pyomo model so analysis can be conducted."""
    #print path + case_name + '_Scenario.pkl'
    with open(os.path.join(path,pickled_file_name), mode='rb') as file:
        test = cloudpickle.load(file)
    return test

##################################################################
def get_routes(df_nodes, instance):
    """ This function takes in a pyomo solution object and constructs dictionary with information about each route selected in the solution of the vehichle routing optimization problem. """
    routes = instance.R.ordered_data() 
    routes_dict = {}

    for k in routes:
        list_num =[]
        list_geo = []
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
        cleaned_geo_list = clean_route(list_geo)
        number_of_vehicles = get_vehicles_on_route(instance,k)
        bus_size = instance.size[k].value
        #routes_dict[k] = [list_num, cleaned_geo_list,number_of_vehicles,bus_size]
        routes_dict['route ' + str(k)] = {"route computed" : list_num, "geographic route" : cleaned_geo_list,"number of vehicles" : number_of_vehicles,"size of bus used on route" : bus_size}

    return routes_dict

##################################################################
def get_vehicles_on_route(instance,route):
    """ utility function for getting the number of vehicles on a route from the pyomo solution object."""
    x = instance.nv[route].value
    return x

##################################################################
def clean_route(list_geo):
    """ utility function for processing a route from the pyomo solution object."""
    clean_list = ['R']
    for i in list_geo:
        if i != clean_list[-1]:
            clean_list.append(i)

    return clean_list

##################################################################

def get_objective(instance):
    """ utility function for getting the objective function value from the pyomo solution object."""
    x = instance.Total_Cost()
    return x

##################################################################
