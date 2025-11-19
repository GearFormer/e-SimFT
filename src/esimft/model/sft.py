import torch
import torch.nn as nn
import numpy as np
from esimft.model.functions import compute_cross_entropy_loss
import torch.optim as optim


class SFT:
    def __init__(self, model):
        self.model = model

    def training_step(self, inputs, labels, ignore_index=-1):

        self.model.optimizer.zero_grad()

        logits = self.model(inputs, labels)

        labels = labels.long().to(logits.device)
        
        loss = compute_cross_entropy_loss(logits, labels[:, 1:], ignore_index=ignore_index)
        
        loss.backward()

        self.model.optimizer.step()

        return loss.item()

    def validation_step(self, inputs, labels, ignore_index=-1):

        self.model.eval()

        with torch.no_grad():
            logits = self.model(inputs, labels)

            labels = labels.long().to(logits.device)
        
            loss = compute_cross_entropy_loss(logits, labels[:, 1:], ignore_index=ignore_index)
        
        return loss.item()
