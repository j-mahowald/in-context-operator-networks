#!/bin/bash

#BATCH -J datagen           # Job name
#SBATCH -o datagen.%j.out    # Name of stdout output file (%j expands to jobId)
#SBATCH -e datagen.%j.err    # Name of stderr output file
#SBATCH -p flex            # Queue name
#SBATCH -N 1                 # Total number of nodes requested
#SBATCH -n 1                 # Total number of tasks
#SBATCH -t 6:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-user=j.mahowald@utexas.edu
#SBATCH --mail-type=all      # Send email at begin and end of job

module load cuda/12.2

# Set up your Python environment if needed
source /work2/09989/jmahowald/frontera/envs/icon-tacc/bin/activate

# Run the datagen.sh script
bash data_preparation/datagen.sh
