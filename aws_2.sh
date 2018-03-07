#!/bin/bash

source activate pytorch_p36

echo 'around the 200/200 edges' >> results2.out
python3 main.py --epochs 5 --cutoff 454,1071,2711 --nhid 200 --emsize 200 --save ./checkpoint/aws5.pt--adasoft --bptt 8 --bptt_multiplier 2 >> results2.out
python3 main.py --epochs 5 --cutoff 454,1071,2711 --tied --nhid 200 --emsize 200 --adam --lr .01 --save ./checkpoint/aws6.pt--adasoft --bptt 8 --bptt_multiplier 2 >> results2.out
python3 main.py --epochs 3 --cutoff 454,1071,2711 --nhid 200 --emsize 200 --save ./checkpoint/aws7.pt--adasoft --bptt 8 --bptt_multiplier 2 >> results2.out
python3 main.py --epochs 2 --cutoff 454,1071,2711 --nhid 200 --emsize 200 --save ./checkpoint/aws8.pt--adasoft --bptt 8 --bptt_multiplier 2 >> results2.out