from copy import copy
import random
import numpy as np
import pandas as pd
import argparse
import os
import warnings
from datetime import datetime
from time import time
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from configs import get_cfg_defaults
from dataloader import DTIDataset
from models import DOGMA
from trainer import Trainer
from utils import set_seed, graph_collate_func, mkdir,graph_collate_func_2
from sklearn.utils import shuffle

cuda_id=0

def init_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    print(cfg)
    set_seed(cfg.SOLVER.SEED)
    mkdir(cfg.RESULT.OUTPUT_DIR)
    print(f"Running on: {device}", end="\n\n")
    dataFolder = f'../datasets/sample/'

    torch.cuda.empty_cache()

    """load data"""
    train_set = pd.read_csv(dataFolder + "traindata.csv")
    val_set = pd.read_csv(dataFolder + "valdata.csv")
    test_set = pd.read_csv(dataFolder + "testdata.csv")

    print("data preprocess end !!!")
    print(f"train:{len(train_set)}")
    print(f"dev:{len(val_set)}")
    print(f"test:{len(test_set)}")
    train_dataset = DTIDataset(train_set.index.values, train_set)
    val_dataset = DTIDataset(val_set.index.values, val_set)
    test_dataset = DTIDataset(test_set.index.values, test_set)

    model = DOGMA(**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    torch.backends.cudnn.benchmark = True

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func_2}

    train_generator = DataLoader(train_dataset, **params)

    params['shuffle'] = False
    params['drop_last'] = False

    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    trainer = Trainer(model, opt, device, train_generator, val_generator, test_generator, **cfg)
    result = trainer.train()
    return result

if __name__ == '__main__':

    device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
    name="DO-GMA"
    result=main()