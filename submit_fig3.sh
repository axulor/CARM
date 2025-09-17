#!/bin/sh
# ------------------------------------------------------------------
# PJM Directives for a 10-node MPI job (based on Genkai User Guide)
# ------------------------------------------------------------------

#PJM -L rscgrp=a-batch

#PJM -L node=10

#PJM -L elapse=02:00:00

#PJM -j

#PJM -N fig3_mpi_10node

#PJM -o ./logs/fig3_mpi_10node.log


# ------------------------------------------------------------------
# Shell Script Section (The part that actually runs)
# ------------------------------------------------------------------

set -eu

echo "================================================="
echo "Multi-Node MPI Job Started on Genkai System"
echo "Job ID: $PJM_JOBID"
echo "Requested Nodes: $PJM_NODE" 
echo "================================================="

source ~/MPGG/myenv/bin/activate
echo "Python virtual environment activated."

cd ~/MPGG/CARM
echo "Changed directory to: $(pwd)"

mkdir -p ./logs

module load intel impi
echo "Intel MPI module loaded."

TOTAL_PROCS=$(($PJM_NODE * 120))
echo "Starting MPI job with ${TOTAL_PROCS} processes across ${PJM_NODE} nodes..."

mpiexec -n ${TOTAL_PROCS} python src/run_fig3_simulations.py


echo "================================================="
echo "MPI Job Finished."
echo "================================================="