#!/bin/bash

#BATCH -J run_lm           # Job name
#SBATCH -o run_lm.%j.out    # Name of stdout output file (%j expands to jobId)
#SBATCH -e run_lm.%j.err    # Name of stderr output file
#SBATCH -p rtx            # Queue name
#SBATCH -N 1                 # Total number of nodes requested
#SBATCH -n 4                 # Total number of tasks
#SBATCH -t 06:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-user=j.mahowald@utexas.edu
#SBATCH --mail-type=all      # Send email at begin and end of job

module load cuda/12.2

# Set up your Python environment if needed
source /work2/09989/jmahowald/frontera/envs/icon-tacc/bin/activate

# Run the datagen.sh script
bash run_lm.sh
