#!/bin/bash  
# Job name:
#SBATCH --job-name=v2g_rl_final_case
#
# Account:
#SBATCH --account=fc_mixedav
#
# Partition:
#SBATCH --partition=savio3
#
#Number of nodes used for job (defaults to 1):
#SBATCH --nodes=1
#
# Quality of service i.e. what kind of resources your job can use (defaults to savio_normal):
#SBATCH --qos=savio_normal
#
# Wall clock limit i.e. maximum time your job will be run (max time limit is 3 days, example shows 30min):
#SBATCH --time=24:00:00
#
#Get email notifications of process. Also has options for BEGIN, REQUEUE, and ALL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jarviskroos7@berkeley.edu

## Command(s) to run:
# Move to root folder (this resolves the path binding issue)

# Load environment
module load python
source activate bistro
# pip install -r requirements.txt
# pip install --user multiprocessing

# Config
export MAXRAM=96g
cd rl/dynamic_td/final_case/
python3 policy_eval.py

