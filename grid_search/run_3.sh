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
echo "this on?"
python main.py --bptt 8 --bptt_multiplier 1.5 --save m_lstm_03.pt >> dlstm_00.out
python main.py --bptt 8 --bptt_multiplier 2 --nhid 64 --emsize 64 --tied --save m_lstm_17.pt >> dlstm_00.out
python main.py --bptt 8 --bptt_multiplier 2 --nhid 64  --save m_lstm_18.pt >> dlstm_00.out
python main.py --bptt 8 --bptt_multiplier 2 --emsize 64  --save m_lstm_19.pt >> dlstm_00.out
echo "1 rounders"
python main.py --bptt 8 --nhid 512 --emsize 512  --epoch 1  >> dlstm_01.out
python main.py --bptt 8 --nhid 256 --emsize 256  --epoch 1  >> dlstm_01.out
python main.py --bptt 8 --nhid 256 --emsize 256 --tied  --epoch 1  >> dlstm_01.out
python main.py --bptt 8 --nhid 256   --epoch 1  >> dlstm_01.out
python main.py --bptt 8 --emsize 256  --epoch 1  >> dlstm_01.out
echo "back at it"
python main.py --bptt 8 --bptt_multiplier 2 --lr 22 --save m_lstm_18.pt >> dlstm_00.out
python main.py --bptt 8 --bptt_multiplier 2 --lr 18 --save m_lstm_18.pt >> dlstm_00.out
python main.py --bptt 8 --bptt_multiplier 2 --emsize 64 --save m_lstm_18.pt >> dlstm_00.out
echo
python main.py --bptt 8 --bptt_multiplier 2 --dropout .15 --save m_lstm_18.pt >> dlstm_00.out
python main.py --bptt 8 --bptt_multiplier 2 --dropout .35 --save m_lstm_18.pt >> dlstm_00.out
python main.py --bptt 8 --bptt_multiplier 2 --dropout .5 --save m_lstm_18.pt >> dlstm_00.out
