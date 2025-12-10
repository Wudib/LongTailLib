"""
Proto Classifier for FedETF
Generates fixed ETF (Equiangular Tight Frame) structure for neural collapse
"""
import torch
import torch.nn as nn
import numpy as np
import copy


class Proto_Classifier(nn.Module):
    """
    ETF Classifier based on Neural Collapse theory
    
    Generates a simplex ETF structure where all class prototypes 
    are equiangular and form an optimal geometric structure.
    """
    def __init__(self, feat_in, num_classes, device='cuda'):
        super(Proto_Classifier, self).__init__()
        self.feat_in = feat_in
        self.num_classes = num_classes
        self.device = device
        
        # Generate ETF structure
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        
        # Simplex ETF formula: M = sqrt(C/(C-1)) * P * (I - (1/C) * 1)
        M = np.sqrt(num_classes / (num_classes - 1)) * torch.matmul(
            P, I - ((1 / num_classes) * one)
        )
        
        # Register as buffer (not trainable by default)
        self.register_buffer('proto', M.to(device))
    
    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        """
        Generate orthogonal matrix via QR decomposition
        
        Args:
            feat_in: feature dimension
            num_classes: number of classes
            
        Returns:
            P: orthogonal matrix of shape (feat_in, num_classes)
        """
        # Random matrix
        a = np.random.random(size=(feat_in, num_classes))
        
        # QR decomposition to get orthogonal matrix
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        
        # Verify orthogonality: P^T @ P = I
        assert torch.allclose(
            torch.matmul(P.T, P), 
            torch.eye(num_classes), 
            atol=1e-06
        ), f"Orthogonality check failed: {torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))}"
        
        return P
    
    def load_proto(self, proto):
        """Load proto from another classifier"""
        self.proto = copy.deepcopy(proto)
    
    def forward(self, label):
        """
        Get prototypes for given labels
        
        Args:
            label: tensor of shape (B,)
            
        Returns:
            target: prototypes of shape (B, feat_in)
        """
        # Select columns corresponding to labels
        target = self.proto[:, label].T  # (B, feat_in)
        return target
