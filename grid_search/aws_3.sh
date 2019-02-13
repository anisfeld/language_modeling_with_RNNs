#!/bin/bash

source activate pytorch_p36

echo 'around the 200/200 edges' >> results3.out
python3 main.py --epochs 5 --cutoff 454,1071,2711 --nhid 256 --emsize 256 --save ./checkpoint/aws9.pt--adasoft --bptt 8 --bptt_multiplier 2 >> results3.out
python3 main.py --epochs 5 --cutoff 454,1071,2711 --tied --nhid 256 --emsize 256 --adam --lr .01 --save ./checkpoint/aws10.pt--adasoft --bptt 8 --bptt_multiplier 2 >> results3.out


echo 'ogs' >>results3
python3 main.py --epochs 4 --nhid 200 --emsize 200 --save ./checkpoint/aws11.pt --bptt 8 --bptt_multiplier 2 >> results3.out
python3 main.py --epochs 4 --nhid 220 --emsize 220 --save ./checkpoint/aws12.pt--adasoft --bptt 8 --bptt_multiplier 2 >> results3.out