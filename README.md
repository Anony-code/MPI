# MPI

In submission.

## Requirements

+ Python 3.10
+ PyTorch 1.1.3
+ DGL 1.0.1
+ OGB 1.3

## Run

+ python mplKD.py --rank  --encoder adj2gnn  --msg_edge ed_kd --seed 1  --ratio 0.3  --msg_edge_other '' --patience 40  --metric Hits@10 --log_steps 1 --lr 0.001  --margin 3  --device 1 --fre_filter 1500
   
## Update

The code will be further organized and refactored upon acceptance.
