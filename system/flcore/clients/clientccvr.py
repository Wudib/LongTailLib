"""
CCVR: Classifier Calibration with Virtual Representations

Reference:
    Luo et al. "No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data"
    https://github.com/Laughing-q/Fed-CCVR
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from flcore.clients.clientbase import Client


class Local(object):
    """Local client for feature distribution computation - copied from source code"""
    
    def __init__(self, data_client, model, device, num_classes, batch_size):
        self.data_client = data_client
        self.local_model = model
        self.device = device
        self.num_classes = num_classes
        self.batch_size = batch_size

    def _cal_mean_cov(self, features):
        """
        Calculate mean and covariance - source code line 86-96
        Note: bias=1 means divide by N instead of N-1 (to avoid NaN when N=1)
        """
        features = np.array(features)
        mean = np.mean(features, axis=0)
        cov = np.cov(features.T, bias=1)
        return mean, cov

    def cal_distributions(self):
        """
        Calculate feature distributions per class - source code line 98-140
        Returns: mean, cov, length for each class
        """
        self.local_model.eval()

        mean = []
        cov = []
        length = []

        # Group data by class
        class_data = {i: [] for i in range(self.num_classes)}
        for idx in range(len(self.data_client)):
            x, y = self.data_client[idx]
            class_data[int(y)].append(x)

        for i in range(self.num_classes):
            features = []
            
            if len(class_data[i]) > 0:
                # Process class i data
                class_tensors = torch.stack(class_data[i])
                class_loader = DataLoader(
                    torch.utils.data.TensorDataset(class_tensors, torch.zeros(len(class_tensors))),
                    batch_size=self.batch_size,
                    shuffle=False
                )
                
                with torch.no_grad():
                    for batch_data, _ in class_loader:
                        batch_data = batch_data.to(self.device)
                        output = self.local_model(batch_data)
                        
                        if isinstance(output, tuple):
                            feature, _ = output
                        else:
                            feature = output
                        
                        features.extend(feature.cpu().tolist())
                
                f_mean, f_cov = self._cal_mean_cov(features)
            else:
                # No data for this class - use zeros
                # Feature dimension determined from model or default 256
                feature_dim = self._get_feature_dim()
                f_mean = np.zeros((feature_dim,))
                f_cov = np.zeros((feature_dim, feature_dim))
            
            mean.append(f_mean)
            cov.append(f_cov)
            length.append(len(class_data[i]))

        return mean, cov, length

    def _get_feature_dim(self):
        """Detect feature dimension from model"""
        self.local_model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            try:
                output = self.local_model(dummy_input)
                if isinstance(output, tuple):
                    feature, _ = output
                    return feature.shape[1]
            except:
                pass
        return 256


class clientCCVR(Client):
    """CCVR client with support for (feature, output) tuple model"""
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        """Local training - handles (feature, output) tuple"""
        trainloader = self.load_train_data()
        self.model.train()
        
        for _ in range(self.local_epochs):
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(x)
                
                if isinstance(output, tuple):
                    feature, output = output
                
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

    def test_metrics(self):
        """Evaluate test metrics, handling (feature, output) tuple"""
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
                
                if isinstance(output, tuple):
                    _, output = output
                
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                # Apply softmax to get probabilities for AUC calculation
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
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        """Compute training loss, handling (feature, output) tuple"""
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        
        with torch.no_grad():
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                
                if isinstance(output, tuple):
                    _, output = output
                
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
