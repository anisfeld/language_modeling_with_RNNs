#!/bin/bash
#SBATCH --job-name=toodeep
#SBATCH --partition=sandyb
#SBATCH --output=batch.out
#SBATCH --error=batch.err
#SBATCH --time=03:30:03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

module load python/3.6.1+intel-16.0
cd /home/anisfeld/scratch-midway/nets
source dlproj/bin/activate




echo "dropout test" >> dlstm_03.out
python main.py --bptt 8 --bptt_multiplier 2 --dropout 0.1  --epoch 5 --save m_lstm_30.pt >> dlstm_03.out
python main.py --bptt 8 --bptt_multiplier 2 --dropout 0 --change_dropout 2  --epoch 5  --save m_lstm_31.pt >> dlstm_03.out
python main.py --bptt 8 --bptt_multiplier 2 --dropout 0 --change_dropout 1  --epoch 5  --save m_lstm_32.pt >> dlstm_03.out


echo "3 rounders" >> dlstm_03.out
python main.py --bptt 8 --bptt_multiplier 2 --nhid 400 --emsize 400 --tied  --epoch 3  >> dlstm_03.out
python main.py --bptt 8 --bptt_multiplier 2 --nhid 200 --emsize 200 --tied  --epoch 3  >> dlstm_03.out

