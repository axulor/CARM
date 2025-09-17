#!/bin/sh
# ------------------------------------------------------------------
# PJM Directives for Figure 8 Micro-simulation MPI Job
# ------------------------------------------------------------------

#PJM -L rscgrp=a-batch
#PJM -L node=1                
#PJM -L elapse=01:00:00        
#PJM -j
#PJM -N fig8_mpi_30core
#PJM -o ./logs/fig8_mpi_30core.log

# ------------------------------------------------------------------
# Shell Script Section
# ------------------------------------------------------------------

set -eu

echo "================================================="
echo "MPI Job for Figure 8 Started on Genkai System"
echo "Job ID: $PJM_JOBID"
echo "Requested Nodes: $PJM_NODE" 
echo "================================================="

source ~/MPGG/myenv/bin/activate
echo "Python virtual environment activated."

cd ~/MPGG/CARM
echo "Changed directory to: $(pwd)"

mkdir -p ./logs
mkdir -p ./data/fig8

module load intel impi
echo "Intel MPI module loaded."

NUM_PROCS=30

echo "Starting MPI job with ${NUM_PROCS} processes on ${PJM_NODE} node(s)..."

mpiexec -n ${NUM_PROCS} python src/run_fig8_simulations.py

echo "================================================="
echo "MPI Job for Figure 8 Finished."
echo "Final data saved in data/fig8/"
echo "================================================="