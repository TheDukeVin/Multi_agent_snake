#!/bin/bash
#SBATCH -c 30 # Number of cores requested
#SBATCH -t 3000 # Runtime in minutes
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=30000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -o lstm_std.out # Standard out goes to this file
#SBATCH -e lstm_std.err # Standard err goes to this filehostname
srun -l a.out