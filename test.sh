#!/bin/bash
#SBATCH --job-name=toodeep
#SBATCH --partition=sandyb
#SBATCH --output=batch.out
#SBATCH --error=batch.err
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

module load python/3.6.1+intel-16.0
cd /home/anisfeld/scratch-midway/nets
source dlproj/bin/activate


python main.py --bptt 8 --nhid 128 --emsize 32 --epoch 1 >> test.out
