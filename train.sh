#!/bin/sh
#SBATCH --job-name=train_tangent_net   # Job name
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=josebouza@ufl.edu # Where to send mail  
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4gb           # Memory per processor
#SBATCH --partition=gpu
#SBATCH --gpus=quadro:1
#SBATCH --time=64:00:00             # Time limit hrs:min:sec
#SBATCH --output=array_%A-%a.out    # Standard output and error log
# This is an example script that combines array tasks with
# bash loops to process many short runs. Array jobs are convenient
# for running lots of tasks, but if each task is short, they
# quickly become inefficient, taking more time to schedule than
# they spend doing any work and bogging down the scheduler for
# all users. 
pwd; hostname; date
PYTHON=~/anaconda3/envs/registration_network/bin/python3
$PYTHON train_synthetic_affine.py
date
