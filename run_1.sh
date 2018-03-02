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

python3 main.py --save m_lstm_00.pt > dlstm_00.out
python3 main.py --bptt 8 --save m_lstm_01.pt >> dlstm_00.out
python3 main.py --bptt 8 --bptt_multiplier 2 --save m_lstm_02.pt >> dlstm_00.out
python3 main.py --bptt 8 --bptt_multiplier 1.5 --save m_lstm_03.pt >> dlstm_00.out
python3 main.py --bptt 8 --nhid 100 --save m_lstm_04.pt >> dlstm_00.out
python3 main.py --bptt 8 --nhid 100 --tied --save m_lstm_05.pt >> dlstm_00.out
python3 main.py --bptt 8 --emsize 128 --save m_lstm_06.pt >> dlstm_00.out
python3 main.py --bptt 8 --emsize 128 --tied --save m_lstm_07.pt >> dlstm_00.out
python3 main.py --bptt 8 --nhid 64 --emsize 64 --tied --save m_lstm_08.pt >> dlstm_00.out
python3 main.py --bptt 8 --nhid 256 --emsize 256 --tied --save m_lstm_09.pt >> dlstm_00.out
python3 main.py --bptt 8 --bptt_multiplier 2 --nhid 100 --tied --save m_lstm_10.pt >> dlstm_00.out
python3 main.py --bptt 8 --bptt_multiplier 1.5 --nhid 100 --tied --save m_lstm_11.pt >> dlstm_00.out
python3 main.py --bptt 8 --bptt_multiplier 2 --nhid 256 --emsize 256 --tied --save m_lstm_12.pt >> dlstm_00.out

python3 main.py --bptt 16  --save m_lstm_13.pt >> dlstm_00.out

python3 main.py --bptt 8 --batch_size 64 --save m_lstm_14.pt >> dlstm_00.out
python3 main.py --bptt 16 --batch_size 64 --save m_lstm_15.pt >> dlstm_00.out
python3 main.py --bptt 8 --batch_size 64 --save m_lstm_16.pt >> dlstm_00.out
python3 main.py --bptt 16 --batch_size 64 --save m_lstm_17.pt >> dlstm_00.out
python3 main.py --bptt 8 --batch_size 64 --save m_lstm_18.pt >> dlstm_00.out
