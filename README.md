# ATHENA-bus
This is a code repository that uses python code, mixed integer programming in the Modeling language Pyomo, and optimization solvers to construct solutions to vehicle routing problems at Airports. 

## Authors

- Sigler, Devon <Devon.Sigler@nrel.gov>, Qichao Wang <Qichao.Wang@nrel.gov>, Zhaocai Liu <Zhaocai.Liu@nrel.gov>


## Purpose

The primary purpose of the repository is to optimize shuttle routes to and from terminals, given some airport campus location like the rental car center. This code was developed at part of the DOE funded Athena project. One aspect of the Athena project was to investigate if the optimization of shuttle routes that move passengers to and from the five DFW terminals and the rental car center could result in lower energy consumption from the shuttle fleet while still maintaining an acceptable level of service to passengers. Hence, the optimization done by the code in this repository determines a set of shuttle routes, the type of shuttle on each route, and the number of shuttles servicing each route. The optimization aims to make these choices in a way that the energy consumption per hour by the fleet is minimized. 

## Solvers

The model constructed and solved by the code in this repo needs powerful commercial solvers to provide good solutions. We have seen that often even commercial solvers fail to prove optimality after an hour. They provide good incumbet solutions but they struggle to find strong lower bounds needed to prove the optimality of  incumbent solutions. Hence to use this code effectively the user should have access to a high quality commerical solver license like Gurobi or CPLEX. Additionlly, the user should expect to find good solutions but not ones that are provably optimal. 

## Environment Installation

You will need to build the conda environment defined in the [environment.yml](environment.yml) file. This can be done with the following command:

                conda env create

This will create the `bus-optimization` environment. If you don't have Anaconda installed you can install it here: [Anaconda](https://docs.anaconda.com/anaconda/install/)

Finally, you will need to install the athena module by activating the environment and then running the [setup.py](setup.py) command.

                conda activate bus-optimization
                python setup.py develop
                pip install -U Pyomo (only if you have choosen to use python 2 for some reason)

## Testing

We use a limited set of tests to verify the installation.  Run `pytest -s -v` to test. One can find more information about the tests at [repo tests README](tests/README.md)

## Getting Started

The best way to understand how to use this code is through our [notebook](notebooks/README.md) examples.

## Sample Data

We have provided a set of sample [data](example_data/) for running an instance of the optimization model the code in this repo constructs and solves. For a description of the sample data see [sample data README.md](example_data/README.md)

## Code

The bulk of the code in this repository is contained within the module route_optimization. See [route_optimization README](route_optimization/README.md) for more information on the contents of this module.



