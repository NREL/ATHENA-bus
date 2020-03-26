
#!/bin/bash

#SBATCH --ntasks=2 # Tasks to be run
#SBATCH --nodes=2  # Run the tasks on the same node
#SBATCH --time=04:00:00   # Required, estimate 5 minutes
#SBATCH --account=athena # Required Talk to Wes about what you should use
#SBATCH --mail-user=devon.sigler@nrel.gov
#SBATCH --job-name=test_job
#SBATCH --output=job_monitor.out
#SBATCH --error=job_monitor.err


module purge
module load conda
source activate speed_env

sleep 5

cd /scratch/$USER/DATA

mpirun -n 2 python script.py                  # where script.py reads files from DATA to run
