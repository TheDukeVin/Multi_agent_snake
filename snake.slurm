#!/bin/bash
#SBATCH -c 20 # Number of cores requested
#SBATCH -t 4000 # Runtime in minutes
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=10000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -o snake_std.out # Standard out goes to this file
#SBATCH -e snake_std.err # Standard err goes to this filehostname
srun -l a.out