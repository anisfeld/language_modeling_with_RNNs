#!/bin/bash

source activate pytorch_p36

#echo 'testing cutoffs' >> results.out
#python3 main.py --epochs 5 --cutoff 176,698,4753 --tied --nhid 200 --emsize 200 --save ./checkpoint/aws1.pt --adasoft --bptt 8 --bptt_multiplier 2 >> results.out
#python3 main.py --epochs 5 --cutoff 454,2711,5400 --tied --nhid 200 --emsize 200 --save ./checkpoint/aws2.pt --adasoft --bptt 8 --bptt_multiplier 2 >> results.out
#python3 main.py --epochs 5 --cutoff 454,1071,5400 --tied --nhid 200 --emsize 200 --save ./checkpoint/aws3.pt --adasoft --bptt 8 --bptt_multiplier 2 >> results.out
#python3 main.py --epochs 5 --cutoff 454,1071,2711 --tied --nhid 200 --emsize 200 --save ./checkpoint/aws4.pt --adasoft --bptt 8 --bptt_multiplier 2 >> results.out
python3 main.py --epochs 5 --cutoff 500,2000 --tied --nhid 200 --emsize 200 --save ./checkpoint/aws1-2.pt --adasoft --bptt 8 --bptt_multiplier 2 >> results.out



