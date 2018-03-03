#!/bin/bash
#SBATCH --job-name=toodeep
#SBATCH --partition=sandyb
#SBATCH --output=batch.out
#SBATCH --error=batch.err
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

module load python/3.6.1+intel-16.0
# for some reason virtualenv wouldn't work in  scratch
cd /home/anisfeld/dl2018_final_project
source dlproj/bin/activate
# run from nets
cd /home/anisfeld/scratch-midway/nets
echo "3/1 group 1" > dlstm_02.out
python main.py --bptt 8 --bptt_multiplier 2 --nhid 64 --emsize 64 --tied --save m_lstm_20.pt >> dlstm_00.out
python main.py --bptt 8 --bptt_multiplier 2 --nhid 64  --save m_lstm_21.pt >> dlstm_00.out
python main.py --bptt 8 --bptt_multiplier 2 --emsize 64  --save m_lstm_22.pt >> dlstm_00.out
echo "dropout test" >> dlstm_02.out
python main.py --bptt 8 --bptt_multiplier 2 --dropout 0  --epoch 4 >> dlstm_02.out
python main.py --bptt 8 --bptt_multiplier 2 --dropout .4 --epoch 4 >> dlstm_02.out
python main.py --bptt 8 --bptt_multiplier 2 --dropout .6 --epoch 4 >> dlstm_02.out
echo "2 rounders" >> dlstm_02.out
python main.py --bptt 8 --nhid 256 --emsize 256 --tied  --epoch 2  >> dlstm_02.out
python main.py --bptt 8 --bptt_multiplier 2 --nhid 256 --emsize 256 --tied  --epoch 2  >> dlstm_02.out
python main.py --bptt 8 --nhid 256 --emsize 256 --tied --dropout 0 --epoch 2  >> dlstm_02.out
python main.py --bptt 8 --bptt_multiplier 2 --nhid 256 --emsize 256 --tied --dropout 0 --epoch 2  >> dlstm_02.out
python main.py --bptt 8 --nhid 64   --epoch 2  >> dlstm_02.out
python main.py --bptt 8 --emsize 256  --epoch 2  >> dlstm_02.out
