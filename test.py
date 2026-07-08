import torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

def test_model_process(model, test_loader, config, store_path_model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(store_path_model))
    model.eval()

    correct_test = 0
    total_test = 0
    test_preds, test_labels, test_preds_probs = [], [], []

    with torch.no_grad():
        for sc, fc, labels in test_loader:
            sc = sc.to(device)
            fc = fc.to(device)
            sc = torch.nan_to_num(sc, nan=0.0)
            fc = torch.nan_to_num(fc, nan=0.0)
            labels = labels.to(device)

            pm, ps, pf, ff, sf, ff_, sf_, output = model(sc, fc, 'test')
            _, predicted = torch.max(pm, 1)

            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_preds_probs.extend(pm[:, 1].cpu().numpy())

    cm = confusion_matrix(test_labels, test_preds)
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    acc = correct_test / total_test if total_test > 0 else 0
    sen = TP / (TP + FN) if (TP + FN) > 0 else 0
    spe = TN / (TN + FP) if (TN + FP) > 0 else 0
    pre = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    auc = roc_auc_score(test_labels, test_preds_probs) if len(set(test_labels)) > 1 else 0

    return acc, sen, spe, pre, f1, auc