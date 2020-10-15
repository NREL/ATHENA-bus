This directory provides sample set of data for creating and solving an instance of the optimization model this repo is focused on solving. The sample set of data consists of 5 csv files in the correct format. Below we provide a desciption of the data contained in each.

1. [bus_types_example.csv](bus_types_example.csv) This csv specifies the types of bus available to the optimization model. Each bus type is described by its maximum passenger capacity and its average hourly energy consumption when in service. 
2. [dwell_time_example.csv](dwell_time_example.csv) This csv specifies in seconds the dwell time by a bus at each terminal and the rental car center. In particular the dwell time is considered to be the time to unload passengers and load new passengers at the stop. 
3. [parameters_example.csv](parameters_example.csv) This csv defines a set of paramters the model needs. 
   * big m is a large constant needed for big m constraints in the model
   * max ride time is the time maximum time passengers are allowed to be on a bus before they get dropped off at their stop. This constrains routes from having passengers visit many stops before being dropped off for the sake of saving energy
   * max bus number is the number of buses available to be on each route
   * headway determines in minutes how often each stop should be visited by a bus
   * number of routes specifies the maximum number of routes that can be used to service all terminals and the rental car center
   * dwell reduction allows the user to see how routes would change if dwell times were reduced at each stop by some percentage. If dwell reduction is set to 1.0 then no reduction is considered. If it is set to 0.7 then the dwell times at each stop will be reduced by 30%.
4. [requests_example.csv](requests_example.csv) this csv provides the number of passengers per hour wanting to take each of the 10 possible trips on the airport road network between the five terminals and the rental car center. 
5. [travel_time_example.csv](travel_time_example.csv) this csv provides the travel times in seconds between each pair of locations (i.e. the five terminals and the rental car center). Note there is no symmetry requirement, meaning going from A to B might have a different travel time then going from B to A. 
