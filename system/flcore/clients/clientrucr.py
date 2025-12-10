"""
RUCR: Representation Unified Classifier Re-training for Class-Imbalanced Federated Learning

Reference:
    RUCR official implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import SGD
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import copy
import torchvision.transforms as transforms

from flcore.clients.clientbase import Client


class MixupDataset_norm(Dataset):
    """
    Mixup dataset for classifier re-training - copied from source code
    """
    def __init__(self, mean, fs_all, num_classes, device, crt_feat_num, uniform_left=0.2, uniform_right=0.95):
        self.data = []
        self.labels = []
        self.means = mean
        self.num_classes = num_classes
        self.device = device
        self.crt_feat_num = crt_feat_num
        self.fs_all = fs_all
        self.fs_len = len(fs_all)
        self.uniform_left = uniform_left
        self.uniform_right = uniform_right

        self.__mixup_syn_feat_pure_rand_norm__()

    def __mixup_syn_feat_pure_rand_norm__(self):
        num = self.crt_feat_num
        l = self.uniform_left
        r_arg = self.uniform_right - l
        for cls in range(self.num_classes):
            if self.means.get(cls) is None:
                continue
            fs_shuffle_idx = torch.randperm(self.fs_len)
            for i in range(num):
                lam = np.round(l + r_arg * np.random.random(), 2)
                neg_f = self.fs_all[fs_shuffle_idx[i % self.fs_len]]
                mixup_f = lam * self.means[cls] + (1 - lam) * F.normalize(neg_f.view(1, -1), dim=1).view(-1)
                self.data.append(mixup_f)
            self.labels += [cls] * num
        if len(self.data) > 0:
            self.data = torch.stack(self.data).to(self.device)
            self.labels = torch.tensor(self.labels).long().to(self.device)
        else:
            # Empty dataset fallback
            self.data = torch.zeros(1, 256).to(self.device)
            self.labels = torch.zeros(1).long().to(self.device)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]


def get_cls_mean_from_feats(feats_all, labels_all):
    """Calculate class-wise mean features - copied from source code"""
    feats_all = torch.cat(feats_all, dim=0)
    labels_all = torch.cat(labels_all)
    real_cls = labels_all.unique()
    cls_means = dict()
    for cls in real_cls.tolist():
        cls_feat = feats_all[labels_all == cls]
        cls_means[cls] = cls_feat.mean(dim=0)
    return cls_means


class clientRUCR(Client):
    """
    RUCR client with representation learning and classifier re-training
    Copied from source code Local class
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # RUCR specific parameters
        self.criterion = nn.CrossEntropyLoss()
        self.cos_sim = nn.CosineSimilarity(dim=1).to(args.device)
        self.log_max = nn.LogSoftmax(dim=1).to(args.device)
        self.nll = nn.NLLLoss().to(args.device)
        self.soft_max = nn.Softmax(dim=1)
        
        # Pre-model for computing centroids
        self.pre_model = copy.deepcopy(args.model)
        
        # RUCR hyperparameters from args (GitHub recommended settings)
        self.feat_loss_arg = getattr(args, 'feat_loss_arg', 0.15)
        self.t = getattr(args, 't', 0.9)  # Temperature
        self.times_arg = getattr(args, 'times_arg', 1.0)
        self.lr_cls_balance = getattr(args, 'lr_cls_balance', 0.01)
        self.local_bal_ep = getattr(args, 'local_bal_ep', 50)
        self.crt_feat_num = getattr(args, 'crt_feat_num', 100)
        self.crt_batch_size = getattr(args, 'crt_batch_size', 256)
        self.uniform_left = getattr(args, 'uniform_left', 0.35)
        self.uniform_right = getattr(args, 'uniform_right', 0.95)
        
        # Feature dimension from model
        self.feat_dim = self._get_feature_dim()
        
        # Placeholders for global prototypes
        self.cls_syn_c = None
        self.cls_syn_c_norm = None
        self.cls_ratio = None
        self.fake_id_tensor = None
        
        # Data augmentation transform
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])
        
        # Class distribution
        self.class_compose = self._get_class_distribution()
        
    def _get_feature_dim(self):
        """Detect feature dimension from model"""
        self.model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            try:
                output = self.model(dummy_input)
                if isinstance(output, tuple):
                    feature, _ = output
                    return feature.shape[1]
            except:
                pass
        return 256
    
    def _get_class_distribution(self):
        """Get class distribution for this client"""
        train_data = self.load_train_data()
        class_counts = [0] * self.num_classes
        for _, y in train_data:
            for label in y:
                class_counts[int(label.item())] += 1
        return class_counts

    def train(self):
        """
        Local training with balanced contrastive learning - source code line 113-140
        """
        trainloader = self.load_train_data()
        self.model.train()
        
        for _ in range(self.local_epochs):
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Apply data augmentation
                x = self.transform_train(x)
                
                self.optimizer.zero_grad()
                output = self.model(x)
                
                if isinstance(output, tuple):
                    f, output = output
                else:
                    f = output
                
                loss = self.criterion(output, y)
                
                # Add feature loss if configured and prototypes available
                if self.feat_loss_arg > 0 and self.cls_syn_c_norm is not None:
                    feat_loss = self.bal_simclr_imp(f, y)
                    loss += self.feat_loss_arg * feat_loss
                
                loss.backward()
                self.optimizer.step()

    def bal_simclr_imp(self, f, labels):
        """
        Balanced SimCLR loss with class-ratio aware weighting - source code line 142-154
        """
        f_norm = F.normalize(f, dim=1)
        # Cosine similarity
        sim_logit = f_norm.mm(self.cls_syn_c_norm.T)
        sim_logit_real = sim_logit.index_fill(1, self.fake_id_tensor, 0)
        # Temperature scaling
        sim_logit_tau = sim_logit_real.div(self.t)
        # Class ratio weighting
        src_ratio = self.cls_ratio[labels].log() * self.times_arg
        add_src = torch.scatter(torch.zeros_like(sim_logit), 1, labels.unsqueeze(1), src_ratio.view(-1, 1))
        f_out = sim_logit_tau + add_src
        loss = self.criterion(f_out, labels)
        return loss

    def get_local_centroid(self):
        """
        Calculate local class centroids - source code line 156-185
        """
        self.pre_model.eval()
        trainloader = self.load_train_data()
        
        global_feats_all, labels_all = [], []
        with torch.no_grad():
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.pre_model(x)
                
                if isinstance(output, tuple):
                    global_feat, _ = output
                else:
                    global_feat = output
                    
                global_feats_all.append(global_feat.data.clone())
                labels_all.append(y)
        
        cls_means = get_cls_mean_from_feats(global_feats_all, labels_all)
        
        # Build synthetic class prototypes
        syn_c = [torch.zeros(self.feat_dim).to(self.device)] * self.num_classes
        real_id, fake_id = [], []
        for k, v in cls_means.items():
            real_id.append(k)
            syn_c[k] = v
        for i in range(self.num_classes):
            if i not in real_id:
                fake_id.append(i)
        
        syn_c = torch.stack(syn_c).to(self.device)
        self.fake_id_tensor = torch.tensor(fake_id, dtype=torch.int64).to(self.device)
        
        feats_all = torch.cat(global_feats_all, dim=0)
        return cls_means, feats_all

    def set_cls_ratio(self, global_num):
        """Set class ratio from global distribution - source code line 208-210"""
        temp = global_num.clone().float().detach().to(self.device)
        self.cls_ratio = temp / temp.sum()

    def local_crt(self, glo_means, fs_all):
        """
        Local classifier re-training with mixup - source code line 187-206
        """
        # Freeze feature extractor
        for param_name, param in self.model.named_parameters():
            if 'classifier' not in param_name:
                param.requires_grad = False
        
        # Create mixup dataset
        crt_dataset = MixupDataset_norm(
            glo_means, fs_all, self.num_classes, self.device, 
            self.crt_feat_num, self.uniform_left, self.uniform_right
        )
        
        self.model.eval()
        temp_optimizer = SGD(self.model.classifier.parameters(), lr=self.lr_cls_balance)
        
        for _ in range(self.local_bal_ep):
            crt_loader = DataLoader(
                dataset=crt_dataset,
                batch_size=self.crt_batch_size,
                shuffle=True
            )
            for feat, cls in crt_loader:
                feat, cls = feat.to(self.device), cls.to(self.device)
                outputs = self.model.classifier(feat)
                loss = self.criterion(outputs, cls)
                temp_optimizer.zero_grad()
                loss.backward()
                temp_optimizer.step()
        
        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True
            
        return copy.deepcopy(self.model.classifier.state_dict())

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
