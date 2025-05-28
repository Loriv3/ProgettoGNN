# coteaching_gnn_trainer.py

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from copy import deepcopy


def label_smoothing_loss(logits, targets, smoothing=0.6):
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(logits, dim=-1)
    n_classes = logits.size(1)

    targets_one_hot = F.one_hot(targets, num_classes=n_classes).float()
    smooth_targets = targets_one_hot * confidence + (1.0 - confidence) / n_classes

    return -(smooth_targets * log_probs).sum(dim=-1)


def select_small_loss_samples(losses, forget_rate):
    keep_num = int((1 - forget_rate) * len(losses))
    _, idx = torch.topk(-losses, keep_num)
    return idx


def train_coteaching(model_A, model_B, train_loader, optimizer_A, optimizer_B,
                     device, epoch, forget_rate=0.5):
    model_A.train()
    model_B.train()

    total_loss_A = 0
    total_loss_B = 0
    total_correct_A = 0
    total_correct_B = 0
    total_samples = 0

    for batch in train_loader:
        batch = batch.to(device)

        logits_A = model_A(batch)
        logits_B = model_B(batch)

        loss_A = label_smoothing_loss(logits_A, batch.y)
        loss_B = label_smoothing_loss(logits_B, batch.y)

        idx_A = select_small_loss_samples(loss_B.detach(), forget_rate)
        idx_B = select_small_loss_samples(loss_A.detach(), forget_rate)

        final_loss_A = loss_A[idx_A].mean()
        final_loss_B = loss_B[idx_B].mean()

        optimizer_A.zero_grad()
        final_loss_A.backward()
        optimizer_A.step()

        optimizer_B.zero_grad()
        final_loss_B.backward()
        optimizer_B.step()

        total_loss_A += final_loss_A.item() * len(idx_A)
        total_loss_B += final_loss_B.item() * len(idx_B)

        pred_A = logits_A.argmax(dim=1)
        pred_B = logits_B.argmax(dim=1)
        total_correct_A += (pred_A[idx_A] == batch.y[idx_A]).sum().item()
        total_correct_B += (pred_B[idx_B] == batch.y[idx_B]).sum().item()
        total_samples += len(idx_A)  # uguale a len(idx_B)

    avg_loss_A = total_loss_A / total_samples
    avg_loss_B = total_loss_B / total_samples
    acc_A = total_correct_A / total_samples
    acc_B = total_correct_B / total_samples

    print(f"Epoch {epoch}: [A] Loss: {avg_loss_A:.4f}, Acc: {acc_A:.4f} | [B] Loss: {avg_loss_B:.4f}, Acc: {acc_B:.4f}")
    return avg_loss_A, acc_A, avg_loss_B, acc_B


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = F.cross_entropy(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs

    return total_loss / total, correct / total


# USAGE (outside this file):
# from coteaching_gnn_trainer import train_coteaching, evaluate
