from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics


def test(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)

    data = load_data(args, os.path.join(os.environ['DATAPATH'], "ppi_test"))
    args.n_nodes, args.feat_dim = data['features'].shape
    args.nb_false_edges = len(data['train_edges_false'])
    args.nb_edges = len(data['train_edges'])
    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs
    model = LPModel(args)
    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
    path = input("input model path: ")
    model.load_state_dict(torch.load(path))
    model.eval()

    embeddings = model.encode(data['features'], data['adj_train_norm'])
    pos_scores = model.decode(embeddings, data['train_edges'])
    print('Embeddings: ', embeddings)
    print(data['train_edges'])
    print('Score: ', pos_scores)

if __name__ == '__main__':
    args = parser.parse_args()
    test(args)
