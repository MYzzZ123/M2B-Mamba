import os
import torch
import yaml
import warnings
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader

from train import train_model_process
from test import test_model_process
from model import M2B_Mamba
from dataloader import get_networks
from utils import fix_seeds

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    fix_seeds(config['seed'])

    fc, sc, y = get_networks()

    kf = KFold(n_splits=config['split_num'], shuffle=True, random_state=config['seed'])

    fold = 0
    Fold_metrics = []

    os.makedirs("results", exist_ok=True)

    for train_idx, test_idx in kf.split(fc):
        fold += 1
        print(f'--------------------Fold {fold}/{config["split_num"]}-------------------------')

        fc_train, sc_train, y_train = fc[train_idx], sc[train_idx], y[train_idx]
        fc_test, sc_test, y_test = fc[test_idx], sc[test_idx], y[test_idx]

        train_set = TensorDataset(torch.tensor(fc_train, dtype=torch.float),
                                  torch.tensor(sc_train, dtype=torch.float),
                                  torch.tensor(y_train, dtype=torch.long))
        test_set = TensorDataset(torch.tensor(fc_test, dtype=torch.float),
                                 torch.tensor(sc_test, dtype=torch.float),
                                 torch.tensor(y_test, dtype=torch.long))

        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

        model = M2B_Mamba().to(device)
        store_path = f"results/model{fold}.pt"

        train_model_process(model, train_loader, config, store_path)

        acc, sen, spe, pre, f1, auc = test_model_process(model, test_loader, config, store_path)

        print(f'Fold {fold} | ACC: {acc:.4f}, SEN: {sen:.4f}, SPE: {spe:.4f}, PRE: {pre:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')
        Fold_metrics.append([acc, sen, spe, pre, f1, auc])

    metrics_array = np.array(Fold_metrics)
    mean_metrics = np.mean(metrics_array, axis=0)
    std_metrics = np.std(metrics_array, axis=0)

    print("----------------------------------------------------------")
    print(f"Mean ACC: {mean_metrics[0] * 100:.2f} ± {std_metrics[0] * 100:.2f}")
    print(f"Mean SEN: {mean_metrics[1] * 100:.2f} ± {std_metrics[1] * 100:.2f}")
    print(f"Mean SPE: {mean_metrics[2] * 100:.2f} ± {std_metrics[2] * 100:.2f}")
    print(f"Mean PRE: {mean_metrics[3] * 100:.2f} ± {std_metrics[3] * 100:.2f}")
    print(f"Mean F1:  {mean_metrics[4] * 100:.2f} ± {std_metrics[4] * 100:.2f}")
    print(f"Mean AUC: {mean_metrics[5] * 100:.2f} ± {std_metrics[5] * 100:.2f}")
    print("----------------------------------------------------------")