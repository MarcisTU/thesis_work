#!/bin/sh -v
#PBS -e /mnt/home/abstrac01/logs
#PBS -o /mnt/home/abstrac01/logs
#PBS -q batch
#PBS -l nodes=1:ppn=32:gpus=4:shared,feature=v100
#PBS -l mem=160gb
#PBS -l walltime=48:00:00
#PBS -N emo_transfer
#PBS -m bea
#PBS -M marcisreb@gmail.com

eval "$(conda shell.bash hook)"
conda activate conda_env
export LD_LIBRARY_PATH=/mnt/home/abstrac01/.conda/envs/conda_env/lib:$LD_LIBRARY_PATH

cd /mnt/beegfs2/home/abstrac01/marcis_upenieks/emo_transfer
CUDA_VISIBLE_DEVICES=0 python emo_cyclegan.py -batch_size=1 -d_iter=2 -run_path=bsize_1_diter_2 &
CUDA_VISIBLE_DEVICES=1 python emo_cyclegan.py -batch_size=1 -d_iter=3 -run_path=bsize_1_diter_3 &
CUDA_VISIBLE_DEVICES=2 python emo_cyclegan.py -batch_size=4 -d_iter=2 -run_path=bsize_4_diter_2 &
CUDA_VISIBLE_DEVICES=3 python emo_cyclegan.py -batch_size=4 -d_iter=3 -run_path=bsize_4_diter_3
wait
