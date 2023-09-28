#!/bin/bash
#SBATCH --job-name=model_CL
#SBATCH --time=2:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=2393033@students.wits.ac.za
RUN_DIR="${SLURM_JOB_NAME}_${SLURM_JOB_ID}_$DATE_STR"
mkdir -p $RUN_DIR/

source /home-mscluster/${USER}/.bashrc
conda activate research

# sbatch python train_mmt.py --gpu 0 --exp model_CL > $RUN_DIR/output.txt 2> $RUN_DIR/errors.txt

sbatch --partition="bigbatch" --nodes=1 --job-name="not head node" --output="${RUN_DIR}/output.txt" --error="${RUN_DIR}/errors.txt" --open-mode=append python train_mmt.py --gpu 0 --exp model_CL