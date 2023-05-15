for Model in GAT GATv2 GCN SAGE;
do
    for Dataset in SHS148 SHS27;
    do
        for Ppimode in activation binding catalysis expression inhibition ptmod reaction;
        do 
            python3 train.py --task lp --dataset ppi --lr 0.01 --n-heads 4 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.2 --weight-decay 0.0001 --manifold Euclidean --log-freq 4 --ppitype $Dataset --ppimode $Ppimode --cuda 0 --model $Model --min-epochs 100000 --epochs 5000
        done 
    done
done
