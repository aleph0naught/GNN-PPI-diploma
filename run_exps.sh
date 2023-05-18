for Model in GCNTorch SAGE GATTorch GATv2Torch GATv2 GAT GCN MLP;
do
    for Dataset in SHS148 SHS27 SHS148all;
    do
        for Ppimode in all;
        do 
	    if [[ $Model -eq "SAGE" || $Model -eq "GCNTorch" ]]; then
		    python3 train.py --save 1 --task lp --dataset ppi --lr 0.005 --n-heads 4 --dim 16 --num-layers 3 --act relu --bias 1 --dropout 0.4 --weight-decay 0.0001 --manifold Euclidean --log-freq 4 --ppitype $Dataset --ppimode $Ppimode --cuda -1 --model $Model --min-epochs 1000000 --epochs 3500 >> log.txt 2>&1
	    else
		    python3 train.py --save 1 --task lp --dataset ppi --lr 0.001 --n-heads 4 --dim 16 --num-layers 3 --act relu --bias 1 --dropout 0.4 --weight-decay 0.0001 --manifold Euclidean --log-freq 4 --ppitype $Dataset --ppimode $Ppimode --cuda -1 --model $Model --lr 0.0001 --min-epochs 1000000 --epochs 3500 >> log.txt 2>&1
	    fi
        done 
    done
done
