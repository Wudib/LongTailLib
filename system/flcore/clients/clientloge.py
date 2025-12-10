"""
FedLoGe: Federated Long-Tailed Learning with Global and Local Classifiers
Direct copy from FedLoGe-master source code

Reference:
    FedLoGe-master/util/update_baseline.py - LocalUpdate.update_weights_gaux
    FedLoGe-master/fedloge.py - main training loop
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from flcore.clients.clientbase import Client


class ClientLOGE(Client):
    """
    FedLoGe Client - copied from FedLoGe-master
    
    Key features:
    - Three-head training: g_head (fixed ETF), g_aux (aggregated), l_head (local)
    - Backbone trained with g_head, then g_aux and l_head trained with detached features
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # FedLoGe specific parameters - source: options.py
        self.loge_lr = getattr(args, 'loge_lr', 0.03)
        self.loge_momentum = getattr(args, 'loge_momentum', 0.5)
        
        # Three heads (set by server)
        self.g_head = None  # Fixed sparse ETF classifier
        self.g_aux = None   # Aggregated auxiliary classifier
        self.l_head = None  # Local personalized classifier
        
        # Override optimizer with FedLoGe defaults
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.loge_lr,
            momentum=self.loge_momentum
        )
        
        # Loss function
        self.loss = nn.CrossEntropyLoss()
    
    def set_heads(self, g_head, g_aux, l_head):
        """
        Set three heads from server
        Source: fedloge.py line 373-376
        """
        self.g_head = copy.deepcopy(g_head).to(self.device)
        self.g_aux = copy.deepcopy(g_aux).to(self.device)
        self.l_head = l_head  # Reference, updated in-place
    
    def extract_features(self, x):
        """
        Extract features before final fc layer
        Source: update_baseline.py line 425: features = net(images, latent_output=True)
        """
        # Use FedLoGe-style model's latent_output parameter
        return self.model(x, latent_output=True)
    
    def train(self):
        """
        Local training - update_weights_gaux
        Source: update_baseline.py line 334-458
        
        Three-step training per batch:
        1. Train backbone with g_head (fixed ETF)
        2. Train g_aux with detached features
        3. Train l_head with detached features
        """
        trainloader = self.load_train_data()
        
        self.model.train()
        self.g_head.train()  # Source: line 372 g_head.train()
        self.g_aux.train()
        self.l_head.train()
        
        # Three optimizers - source: line 337-343
        optimizer_g_backbone = torch.optim.SGD(
            self.model.parameters(),
            lr=self.loge_lr,
            momentum=self.loge_momentum
        )
        optimizer_g_aux = torch.optim.SGD(
            self.g_aux.parameters(),
            lr=self.loge_lr,
            momentum=self.loge_momentum
        )
        optimizer_l_head = torch.optim.SGD(
            self.l_head.parameters(),
            lr=self.loge_lr,
            momentum=self.loge_momentum
        )
        
        # Loss functions - source: line 350-351
        criterion_l = nn.CrossEntropyLoss()
        criterion_g = nn.CrossEntropyLoss()
        
        epoch_loss = []
        
        # Training loop - source: line 411-458
        for epoch in range(self.local_epochs):
            batch_loss = []
            
            for batch_idx, (images, labels) in enumerate(trainloader):
                if isinstance(images, list):
                    images = images[0]
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                
                # Zero gradients - source: line 419-421
                optimizer_g_backbone.zero_grad()
                optimizer_g_aux.zero_grad()
                optimizer_l_head.zero_grad()
                
                # Extract features - source: line 425
                features = self.extract_features(images)
                
                # Step 1: Train backbone with g_head - source: line 430-440
                output_g_backbone = self.g_head(features)
                loss_g_backbone = criterion_g(output_g_backbone, labels)
                loss_g_backbone.backward()
                optimizer_g_backbone.step()
                
                # Step 2: Train g_aux with detached features - source: line 443-446
                output_g_aux = self.g_aux(features.detach())
                loss_g_aux = criterion_l(output_g_aux, labels)
                loss_g_aux.backward()
                optimizer_g_aux.step()
                
                # Step 3: Train l_head with detached features - source: line 449-452
                output_l_head = self.l_head(features.detach())
                loss_l_head = criterion_l(output_l_head, labels)
                loss_l_head.backward()
                optimizer_l_head.step()
                
                # Total loss - source: line 454
                loss = loss_g_backbone + loss_g_aux + loss_l_head
                batch_loss.append(loss.item())
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        
        self.g_head.eval()  # Source: line 379 g_head.eval()
    
    def test_metrics(self):
        """
        Test using g_aux (aggregated classifier)
        Source: update_baseline.py localtest function
        """
        testloader = self.load_test_data()
        self.model.eval()
        self.g_aux.eval()
        
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloader:
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)
                
                features = self.extract_features(x)
                output = self.g_aux(features)
                
                test_acc += (torch.argmax(output, dim=1) == y).sum().item()
                test_num += y.shape[0]
                
                y_prob.append(torch.softmax(output, dim=1).cpu().numpy())
                
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)
        
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        
        try:
            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        except:
            auc = 0.5
        
        return test_acc, test_num, auc
    
    def train_metrics(self):
        """Training loss using g_aux"""
        trainloader = self.load_train_data()
        self.model.eval()
        self.g_aux.eval()
        
        train_num = 0
        losses = 0
        
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)
                
                features = self.extract_features(x)
                output = self.g_aux(features)
                loss = self.loss(output, y)
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        
        return losses, train_num
    
    def test_metrics_global(self):
        """
        Test using g_head (ETF classifier) for global accuracy
        Source: update_baseline.py globaltest function line 1416-1475
        """
        testloader = self.load_test_data()
        self.model.eval()
        self.g_head.eval()
        
        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in testloader:
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)
                
                features = self.extract_features(x)
                output = self.g_head(features)
                
                test_acc += (torch.argmax(output, dim=1) == y).sum().item()
                test_num += y.shape[0]
        
        return test_acc, test_num
