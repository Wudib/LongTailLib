"""
FedIC Client - Direct copy from FEDIC-main/main.py Local class

Reference:
    FEDIC-main/main.py line 265-303 - Local class
    
Key features:
- Simple local training with SGD and CrossEntropyLoss
- Model returns (feature, logit) tuple, only logit used for loss
- No client-side knowledge distillation
"""

import copy
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data


class ClientFEDIC(Client):
    """
    FedIC Client - Direct copy from FEDIC-main/main.py Local class (line 265-303)
    
    Simple local training:
    - SGD optimizer (no momentum in source)
    - CrossEntropyLoss
    - Model returns (feature, logit), only logit used for CE loss
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # FedIC specific parameters - source: options.py
        # lr_local_training = 0.1 (source line 28)
        self.lr = getattr(args, 'fedic_lr_local', 0.1)
        
        # Override optimizer: SGD without momentum (source: main.py line 283)
        self.optimizer = SGD(self.model.parameters(), lr=self.lr)
        
        # Loss function
        self.ce_loss = CrossEntropyLoss()
    
    def set_parameters_dict(self, dict_params):
        """
        Set model parameters from dict.
        Source: main.py line 278 - self.model.load_state_dict(global_params)
        """
        self.model.load_state_dict(dict_params)
    
    def train(self):
        """
        Local training - Direct copy from source: main.py line 287-300
        
        Simple training loop:
        - For each epoch, iterate through local data
        - Forward pass returns (feature, logit)
        - Compute CE loss on logit
        - Backward and update
        """
        self.model.train()
        
        for epoch in range(self.local_epochs):
            # Create dataloader for each epoch (like source)
            train_data = read_client_data(self.dataset, self.id, is_train=True)
            data_loader = DataLoader(
                dataset=train_data,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True
            )
            
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward - model returns (feature, logit)
                # Source: main.py line 296
                _, outputs = self.model(images)
                
                # Loss - source: main.py line 297
                loss = self.ce_loss(outputs, labels)
                
                # Backward - source: main.py line 298-300
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def upload_params(self):
        """
        Upload model parameters.
        Source: main.py line 302-303
        """
        return copy.deepcopy(self.model.state_dict())
    
    def test_metrics_local(self):
        """
        Compute test metrics on local test data.
        Returns dict with num_samples, num_correct, loss.
        """
        self.model.eval()
        
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        
        num_correct = 0
        num_samples = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.model(images)
                
                loss = self.ce_loss(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                
                _, predicted = torch.max(outputs, 1)
                num_correct += (predicted == labels).sum().item()
                num_samples += labels.size(0)
        
        return {
            'num_samples': num_samples,
            'num_correct': num_correct,
            'loss': total_loss / num_samples if num_samples > 0 else 0.0
        }
