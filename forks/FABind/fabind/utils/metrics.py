import torchmetrics
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd

def myMetric(y_pred, y, threshold=0.5):
    y = y.float()
    criterion = nn.BCELoss()
    with torch.no_grad():
        loss = criterion(y_pred, y)

    # y = y.long()
    y = y.bool()
    # y = y.int()
    acc = torchmetrics.functional.accuracy(y_pred, y, threshold=threshold)
    auroc = torchmetrics.functional.auroc(y_pred, y)
    precision_0, precision_1 = torchmetrics.functional.precision(y_pred, y, 
                                      num_classes=2, multiclass=True,
                                      average='none', threshold=threshold)
    recall_0, recall_1 = torchmetrics.functional.recall(y_pred, y, 
                                      num_classes=2, multiclass=True,
                                      average='none', threshold=threshold)
    f1_0, f1_1 = torchmetrics.functional.f1_score(y_pred, y, 
                                      num_classes=2, multiclass=True,
                                      average='none', threshold=threshold)
    return {"BCEloss":loss.item(),
            "acc":acc, "auroc":auroc, "precision_1":precision_1,
           "recall_1":recall_1, "f1_1":f1_1,"precision_0":precision_0,
           "recall_0":recall_0, "f1_0":f1_0}

def cls_metric(y_pred, y, threshold=0.5):
    y = y.float()
    criterion = nn.BCELoss()
    with torch.no_grad():
        loss = criterion(y_pred, y)

    # y = y.long()
    y = y.bool()
    # y = y.int()
    acc = torchmetrics.functional.accuracy(y_pred, y, threshold=threshold)
    auroc = torchmetrics.functional.auroc(y_pred, y)
    precision_0, precision_1 = torchmetrics.functional.precision(y_pred, y, 
                                      num_classes=2, multiclass=True,
                                      average='none', threshold=threshold)
    recall_0, recall_1 = torchmetrics.functional.recall(y_pred, y, 
                                      num_classes=2, multiclass=True,
                                      average='none', threshold=threshold)
    f1_0, f1_1 = torchmetrics.functional.f1_score(y_pred, y, 
                                      num_classes=2, multiclass=True,
                                      average='none', threshold=threshold)
    return {"pocket_BCEloss":loss.item(),
            "pocket_acc":acc, "pocket_auroc":auroc, "pocket_precision_1":precision_1,
           "pocket_recall_1":recall_1, "pocket_f1_1":f1_1,"pocket_precision_0":precision_0,
           "pocket_recall_0":recall_0, "pocket_f1_0":f1_0}

def affinity_metrics(affinity_pred, affinity):
    pearson = torchmetrics.functional.pearson_corrcoef(affinity_pred, affinity)
    rmse = torchmetrics.functional.mean_squared_error(affinity_pred, affinity, squared=False)
    return {"pearson":pearson, "rmse":rmse}

def pocket_metrics(pocket_coord_pred, pocket_coord):
    pearson_x = torchmetrics.functional.pearson_corrcoef(pocket_coord_pred[:, 0], pocket_coord[:, 0])
    rmse_x = torchmetrics.functional.mean_squared_error(pocket_coord_pred[:, 0], pocket_coord[:, 0], squared=False)
    mae_x = torchmetrics.functional.mean_absolute_error(pocket_coord_pred[:, 0], pocket_coord[:, 0])
    pearson_y = torchmetrics.functional.pearson_corrcoef(pocket_coord_pred[:, 1], pocket_coord[:, 1])
    rmse_y = torchmetrics.functional.mean_squared_error(pocket_coord_pred[:, 1], pocket_coord[:, 1], squared=False)
    mae_y = torchmetrics.functional.mean_absolute_error(pocket_coord_pred[:, 1], pocket_coord[:, 1])
    pearson_z = torchmetrics.functional.pearson_corrcoef(pocket_coord_pred[:, 2], pocket_coord[:, 2])
    rmse_z = torchmetrics.functional.mean_squared_error(pocket_coord_pred[:, 2], pocket_coord[:, 2], squared=False)
    mae_z = torchmetrics.functional.mean_absolute_error(pocket_coord_pred[:, 2], pocket_coord[:, 2])
    pearson = (pearson_x + pearson_y + pearson_z) / 3
    rmse = (rmse_x + rmse_y + rmse_z) / 3
    mae = (mae_x + mae_y + mae_z) / 3
    pocket_pairwise_dist = F.pairwise_distance(pocket_coord_pred, pocket_coord, p=2)
    DCC = (pocket_pairwise_dist < 4).sum().item() / len(pocket_pairwise_dist)
    return {"pocket_pearson":pearson, "pocket_rmse":rmse, "pocket_mae":mae, "pocket_center_avg_dist": pocket_pairwise_dist.mean().item(), "pocket_center_DCC": DCC * 100}
    
def pocket_direct_metrics(pocket_coord_pred, pocket_coord):
    pearson_x = torchmetrics.functional.pearson_corrcoef(pocket_coord_pred[:, 0], pocket_coord[:, 0])
    rmse_x = torchmetrics.functional.mean_squared_error(pocket_coord_pred[:, 0], pocket_coord[:, 0], squared=False)
    mae_x = torchmetrics.functional.mean_absolute_error(pocket_coord_pred[:, 0], pocket_coord[:, 0])
    pearson_y = torchmetrics.functional.pearson_corrcoef(pocket_coord_pred[:, 1], pocket_coord[:, 1])
    rmse_y = torchmetrics.functional.mean_squared_error(pocket_coord_pred[:, 1], pocket_coord[:, 1], squared=False)
    mae_y = torchmetrics.functional.mean_absolute_error(pocket_coord_pred[:, 1], pocket_coord[:, 1])
    pearson_z = torchmetrics.functional.pearson_corrcoef(pocket_coord_pred[:, 2], pocket_coord[:, 2])
    rmse_z = torchmetrics.functional.mean_squared_error(pocket_coord_pred[:, 2], pocket_coord[:, 2], squared=False)
    mae_z = torchmetrics.functional.mean_absolute_error(pocket_coord_pred[:, 2], pocket_coord[:, 2])
    pearson = (pearson_x + pearson_y + pearson_z) / 3
    rmse = (rmse_x + rmse_y + rmse_z) / 3
    mae = (mae_x + mae_y + mae_z) / 3
    return {"pocket_direct_pearson":pearson, "pocket_direct_rmse":rmse, "pocket_direct_mae":mae}

def print_metrics(metrics):
    out_list = []
    for key in metrics:
        try:
            out_list.append(f"{key}:{metrics[key]:6.3f}")
        except:
            out_list.append(f"\n{key}:\n{metrics[key]}")
    out = ", ".join(out_list)
    return out


def compute_individual_metrics(pdb_list, inputFile_list, y_list):
    r_ = []
    for i in range(len(pdb_list)):
        pdb = pdb_list[i]
        # inputFile = f"{pre}/input/{pdb}.pt"
        inputFile = inputFile_list[i]
        y = y_list[i]
        (coords, y_pred, protein_nodes_xyz, 
         compound_pair_dis_constraint, pdb, sdf_fileName, mol2_fileName, pre) = torch.load(inputFile)
        result = myMetric(torch.tensor(y_pred).reshape(-1), y.reshape(-1))
        for key in result:
            result[key] = float(result[key])
        result['idx'] = i
        result['pdb'] = pdb
        result['p_length'] = protein_nodes_xyz.shape[0]
        result['c_length'] = coords.shape[0]
        result['y_length'] = y.reshape(-1).shape[0]
        result['num_contact'] = int(y.sum())
        r_.append(result)
    result = pd.DataFrame(r_)
    return result

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report