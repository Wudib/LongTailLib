"""
FedETF: Federated Learning with ETF Classifier
for Class-Imbalanced Learning

Reference:
    FedETF official implementation (FedETF-main/client_funct.py)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from flcore.clients.clientbase import Client


def balanced_softmax_loss(logits, labels, sample_per_class, reduction="mean"):
    """
    Compute the Balanced Softmax Loss - source code line 237-251
    
    Args:
        logits: A float tensor of size [batch, no_of_classes]
        labels: A int tensor of size [batch]
        sample_per_class: A int tensor of size [no of classes]
        reduction: string. One of "none", "mean", "sum"
    Returns:
        loss: Balanced Softmax Loss
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


class clientFedETF(Client):
    """
    FedETF client with ETF classifier and balanced softmax loss
    Copied from source code client_funct.py
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # FedETF specific parameters - source code args.py
        self.scaling_train = getattr(args, 'scaling_train', 1.0)
        
        # Override optimizer with FedETF source code defaults
        # Source: args.py: lr=0.04, momentum=0.9, local_wd_rate=5e-4
        fedetf_lr = getattr(args, 'fedetf_lr', 0.04)
        fedetf_momentum = getattr(args, 'fedetf_momentum', 0.9)
        fedetf_wd = getattr(args, 'fedetf_wd', 5e-4)
        
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=fedetf_lr,
            momentum=fedetf_momentum,
            weight_decay=fedetf_wd
        )
        
        # Calculate sample_per_class for balanced softmax loss
        self.sample_per_class = self._compute_sample_per_class()
        
    def _compute_sample_per_class(self):
        """
        Compute the number of samples per class for this client - source code node.py line 37
        """
        train_data = self.load_train_data()
        class_counts = torch.zeros(self.num_classes, device=self.device)
        
        for _, y in train_data:
            for label in y:
                class_counts[int(label.item())] += 1
        
        # Avoid log(0) by adding small epsilon to zero counts
        class_counts = class_counts.clamp(min=1e-8)
        return class_counts
    
    def train(self):
        """
        Local training with ETF classifier and balanced softmax loss
        Source code: client_funct.py line 254-281 (client_fedetf function)
        """
        trainloader = self.load_train_data()
        self.model.train()
        
        for _ in range(self.local_epochs):
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass - model returns (feature, logit, out)
                output = self.model(x)
                
                if isinstance(output, tuple) and len(output) >= 2:
                    feature = output[0]  # Normalized feature for FedETF
                else:
                    # Fallback for models without ETF structure
                    feature = output
                
                # ETF logit: feature @ proto
                # The proto_classifier.proto has shape (feat_in, num_classes)
                # feature has shape (batch, num_classes) after linear_proto + normalization
                # output_local = scaling * (feature @ proto)
                if hasattr(self.model, 'proto_classifier'):
                    proto = self.model.proto_classifier.proto
                    output_local = torch.matmul(feature, proto)
                    output_local = self.model.scaling_train * output_local
                else:
                    output_local = feature
                
                # Balanced softmax loss
                loss = balanced_softmax_loss(output_local, y, self.sample_per_class)
                
                loss.backward()
                self.optimizer.step()
    
    def receive_proto(self, proto):
        """
        Receive and load global prototype from server
        Source code: client_funct.py line 24
        """
        if hasattr(self.model, 'proto_classifier'):
            self.model.proto_classifier.load_proto(proto)
    
    def test_metrics(self):
        """
        Evaluate test metrics
        FedETF uses feature @ proto for prediction
        """
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                
                if isinstance(output, tuple) and len(output) >= 2:
                    feature = output[0]
                    # Use ETF prediction
                    if hasattr(self.model, 'proto_classifier'):
                        proto = self.model.proto_classifier.proto
                        output = torch.matmul(feature, proto)
                        output = self.model.scaling_train * output
                    else:
                        output = output[1]  # Use logit
                
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                
                prob = torch.softmax(output, dim=1)
                y_prob.append(prob.detach().cpu().numpy())
                
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
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
        """
        Compute training loss using balanced softmax loss
        """
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        
        with torch.no_grad():
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                
                if isinstance(output, tuple) and len(output) >= 2:
                    feature = output[0]
                    if hasattr(self.model, 'proto_classifier'):
                        proto = self.model.proto_classifier.proto
                        output = torch.matmul(feature, proto)
                        output = self.model.scaling_train * output
                    else:
                        output = output[1]
                
                loss = balanced_softmax_loss(output, y, self.sample_per_class)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
