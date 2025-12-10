"""
FedLoGe: Federated Long-Tailed Learning with Global and Local Classifiers
Direct copy from FedLoGe-master source code

Reference:
    FedLoGe-master/fedloge.py - main training loop
    FedLoGe-master/util/fedavg.py - aggregation methods
    FedLoGe-master/util/etf_methods.py - ETF classifier
    FedLoGe-master/options.py - default parameters
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from flcore.clients.clientloge import ClientLOGE
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from flcore.trainmodel.resnet_cifar import resnet18_cifar


class ETF_Classifier(nn.Module):
    """
    ETF (Equiangular Tight Frame) Classifier
    Source: FedLoGe-master/util/etf_methods.py line 83-111
    """
    def __init__(self, feat_in, num_classes):
        super(ETF_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes - 1)) * torch.matmul(P, I - ((1 / num_classes) * one))
        self.ori_M = M

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        """Generate orthogonal matrix using QR decomposition"""
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        return P

    def gen_sparse_ETF(self, feat_in, num_classes, beta=0.6):
        """
        Generate sparse ETF matrix
        Source: FedLoGe-master/util/etf_methods.py line 123-177
        """
        etf = copy.deepcopy(self.ori_M)
        
        # Sparsify: randomly set beta% elements to zero
        num_zero_elements = int(beta * feat_in * num_classes)
        zero_indices = np.random.choice(feat_in * num_classes, num_zero_elements, replace=False)
        etf_flatten = etf.flatten()
        etf_flatten[zero_indices] = 0
        sparse_etf = etf_flatten.reshape(feat_in, num_classes)
        
        # Optimize non-zero elements
        sparse_etf = torch.tensor(sparse_etf, requires_grad=True, dtype=torch.float32)
        mask = (sparse_etf != 0).float()
        
        optimizer = optim.Adam([sparse_etf], lr=0.0001)
        
        # Optimization loop - source: line 146-173
        n_steps = 10000
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Constraint 1: L2 norm of each column ≈ 0.1
            col_norms = torch.norm(sparse_etf, p=2, dim=0)
            norm_loss = torch.sum((col_norms - 0.1) ** 2)
            
            # Constraint 2: Maximize angle between vectors
            normalized_etf = sparse_etf / (col_norms + 1e-8)
            cos_sim = torch.mm(normalized_etf.t(), normalized_etf)
            torch.diagonal(cos_sim).fill_(-1)
            angle_loss = -torch.acos(cos_sim.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
            
            loss = norm_loss + angle_loss
            loss.backward()
            
            # Apply mask to keep zero elements as zero
            if sparse_etf.grad is not None:
                sparse_etf.grad *= mask
            
            optimizer.step()
            
            if step % 2000 == 0:
                print(f"[ETF Optimization] Step {step}/{n_steps}, Loss: {loss.item():.6f}")
        
        return sparse_etf.detach()


def FedAvg_noniid(w_locals, dict_len):
    """
    FedAvg aggregation weighted by sample size
    Source: FedLoGe-master/util/fedavg.py line 144-153 (FedAvg_noniid)
    """
    w_avg = copy.deepcopy(w_locals[0])
    total_len = sum(dict_len)
    
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * dict_len[0]
        for i in range(1, len(w_locals)):
            w_avg[k] += w_locals[i][k] * dict_len[i]
        w_avg[k] = w_avg[k] / total_len
    
    return w_avg


def FedAvg_noniid_classifier(classifiers, dict_len):
    """
    FedAvg aggregation for classifiers (nn.Linear)
    Source: FedLoGe-master/util/fedavg.py line 54-68
    """
    model = copy.deepcopy(classifiers[0])
    
    # Convert to state_dict
    w = [c.state_dict() for c in classifiers]
    
    # Weighted average
    w_avg = copy.deepcopy(w[0])
    total_len = sum(dict_len)
    
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * dict_len[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
        w_avg[k] = w_avg[k] / total_len
    
    model.load_state_dict(w_avg)
    return model


def shot_split(class_distribution, threshold_3shot=[75, 95]):
    """
    Split classes into head/middle/tail based on cumulative sample percentage
    Source: FedLoGe-master/util/util.py
    """
    import copy as cp
    class_distribution = cp.deepcopy(class_distribution)
    
    # Create map: [count, classid]
    sorted_classes = [(class_distribution[i], i) for i in range(len(class_distribution))]
    sorted_classes.sort(reverse=True)  # Sort by count descending
    
    # Calculate cumulative percentage
    total = sum(class_distribution)
    cumsum = 0
    cumulative = []
    for count, _ in sorted_classes:
        cumsum += count
        cumulative.append(cumsum / total * 100)
    
    # Find cut points
    cut1, cut2 = 0, 0
    for i, cum_pct in enumerate(cumulative):
        if cum_pct >= threshold_3shot[0] and cut1 == 0:
            cut1 = i + 1
        if cum_pct >= threshold_3shot[1] and cut2 == 0:
            cut2 = i + 1
    
    three_shot_dict = {
        "head": [sorted_classes[i][1] for i in range(cut1)],
        "middle": [sorted_classes[i][1] for i in range(cut1, cut2)],
        "tail": [sorted_classes[i][1] for i in range(cut2, len(sorted_classes))]
    }
    
    return three_shot_dict


class FedLoGe(Server):
    """
    FedLoGe Server - Federated Long-Tailed Learning
    Source: FedLoGe-master/fedloge.py
    
    Key features:
    - SSE-C: Sparse Simplex ETF Classifier (fixed g_head)
    - g_aux: Aggregated auxiliary classifier
    - l_heads: Local personalized classifiers (one per client)
    - FedAvg aggregation for backbone and g_aux
    """
    
    @staticmethod
    def add_args(parser):
        """Register FedLoGe-specific arguments - source: options.py"""
        parser.add_argument('--loge_lr', type=float, default=0.03,
                            help='FedLoGe: learning rate (source: 0.03)')
        parser.add_argument('--loge_momentum', type=float, default=0.5,
                            help='FedLoGe: SGD momentum (source: 0.5)')
        parser.add_argument('--loge_local_epochs', type=int, default=5,
                            help='FedLoGe: local epochs (source: 5)')
        parser.add_argument('--loge_batch_size', type=int, default=8,
                            help='FedLoGe: batch size (source: 8)')
        parser.add_argument('--loge_join_ratio', type=float, default=None,
                            help='FedLoGe: client selection ratio (source: 1.0 for cifar100, 0.2 for cifar10)')
        parser.add_argument('--loge_beta', type=float, default=0.6,
                            help='FedLoGe: ETF sparsity (source: 0.6)')
        parser.add_argument('--loge_num_clients', type=int, default=40,
                            help='FedLoGe: number of clients (source: 40)')
    
    def __init__(self, args, times):
        # Override with FedLoGe source code defaults BEFORE parent init
        # Source: options.py
        
        # Fixed seed from source code - line 30: seed=3407
        args.random_seed = 3407
        
        # num_clients - source: line 14/48: num_users=40
        args.num_clients = getattr(args, 'loge_num_clients', 40)
        
        # local_epochs - source: line 11/45: local_ep=5
        args.local_epochs = getattr(args, 'loge_local_epochs', 5)
        
        # batch_size - source: line 15/49: local_bs=8
        args.batch_size = getattr(args, 'loge_batch_size', 8)
        
        # join_ratio - source: CIFAR-10 frac=0.2 (line 46), CIFAR-100 frac=1.0 (line 12)
        dataset_name = args.dataset.lower()
        if 'cifar10' in dataset_name and 'cifar100' not in dataset_name:
            default_frac = 0.2  # CIFAR-10 default
        else:
            default_frac = 1.0  # CIFAR-100 default
        
        # Use user-provided value or dataset-specific default
        loge_join_ratio = getattr(args, 'loge_join_ratio', None)
        args.join_ratio = loge_join_ratio if loge_join_ratio is not None else default_frac
        
        # CRITICAL: Use FedLoGe-style ResNet18 (exact copy from source)
        # Source: FedLoGe-master/model/model_res.py
        if 'cifar' in dataset_name:
            print(f"\n[FedLoGe] Using FedLoGe-style ResNet18 (from resnet_cifar.py)")
            args.model = resnet18_cifar(num_classes=args.num_classes).to(args.device)
        
        super().__init__(args, times)
        
        # FedLoGe hyperparameters
        self.loge_lr = getattr(args, 'loge_lr', 0.03)
        self.loge_momentum = getattr(args, 'loge_momentum', 0.5)
        self.loge_beta = getattr(args, 'loge_beta', 0.6)
        
        # Get feature dimension from model
        if hasattr(self.global_model, 'fc'):
            self.feature_dim = self.global_model.fc.in_features
        elif hasattr(self.global_model, 'linear'):
            self.feature_dim = self.global_model.linear.in_features
        else:
            self.feature_dim = 512  # Default
        
        # Initialize three classifiers - source: fedloge.py line 233-345
        print(f"\n[FedLoGe] Initializing ETF Classifier...")
        print(f"  Feature dim: {self.feature_dim}, Num classes: {self.num_classes}")
        
        # g_head: Fixed sparse ETF classifier - source: line 233-291
        etf = ETF_Classifier(self.feature_dim, self.num_classes)
        sparse_etf_mat = etf.gen_sparse_ETF(
            feat_in=self.feature_dim,
            num_classes=self.num_classes,
            beta=self.loge_beta
        )
        
        self.g_head = nn.Linear(self.feature_dim, self.num_classes).to(self.device)
        self.g_head.weight.data = sparse_etf_mat.to(self.device).t()
        
        # g_aux: Aggregated auxiliary classifier - source: line 341
        self.g_aux = nn.Linear(self.feature_dim, self.num_classes).to(self.device)
        
        # l_heads: Local personalized classifiers (one per client) - source: line 343-345
        self.l_heads = []
        for i in range(self.num_clients):
            l_head = nn.Linear(self.feature_dim, self.num_classes).to(self.device)
            self.l_heads.append(l_head)
        
        # Create clients
        self.set_clients(ClientLOGE)
        
        # Compute global training distribution for 3-shot split
        self._compute_global_train_distribution()
        
        print(f"\n{'='*60}")
        print(f"FedLoGe Configuration (source code defaults):")
        print(f"  seed: 3407 (source: 3407)")
        print(f"  num_clients: {self.num_clients} (source: 40)")
        print(f"  join_ratio: {self.join_ratio} (source: cifar10=0.2, cifar100=1.0)")
        print(f"  global_rounds: {self.global_rounds}")
        print(f"  local_epochs: {self.local_epochs} (source: 5)")
        print(f"  batch_size: {self.batch_size} (source: 8)")
        print(f"  lr: {self.loge_lr} (source: 0.03)")
        print(f"  momentum: {self.loge_momentum} (source: 0.5)")
        print(f"  ETF sparsity beta: {self.loge_beta} (source: 0.6)")
        print(f"{'='*60}\n")
    
    def set_clients(self, clientObj):
        """Initialize clients"""
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(
                self.args,
                id=i,
                train_samples=len(train_data),
                test_samples=len(test_data),
                train_slow=False,
                send_slow=False
            )
            self.clients.append(client)
    
    def _compute_global_train_distribution(self):
        """Compute global training distribution for 3-shot split"""
        class_counts = [0] * self.num_classes
        
        for c in self.clients:
            train_data = read_client_data(self.dataset, c.id, is_train=True)
            for _, label in train_data:
                if isinstance(label, int):
                    class_counts[label] += 1
                else:
                    class_counts[label.item()] += 1
        
        self.global_train_distribution = class_counts
        self.three_shot_dict = shot_split(class_counts)
        
        print(f"3-Shot Split: head={self.three_shot_dict['head']}, "
              f"middle={self.three_shot_dict['middle']}, tail={self.three_shot_dict['tail']}")
    
    def train(self):
        """
        Main training loop
        Source: FedLoGe-master/fedloge.py line 363-477
        """
        for round_idx in range(self.global_rounds):
            s_t = time.time()
            
            print(f"\n-------------Round {round_idx}-------------")
            
            # Client selection - source: line 369
            self.selected_clients = self.select_clients()
            idxs_users = [c.id for c in self.selected_clients]
            
            # Send global model and heads to clients
            self.send_models()
            
            # Local training - source: line 371-378
            self.g_head.train()  # Source: line 372
            
            w_locals = []
            g_auxs = []
            
            for client in self.selected_clients:
                # Set heads for client - source: line 375-376
                client.set_heads(
                    self.g_head,
                    self.g_aux,
                    self.l_heads[client.id]
                )
                
                # Local update - source: line 376
                client.train()
                
                # Collect results
                w_locals.append(copy.deepcopy(client.model.state_dict()))
                g_auxs.append(copy.deepcopy(client.g_aux))
            
            self.g_head.eval()  # Source: line 379
            
            # Aggregation - source: line 381-387
            dict_len = [c.train_samples for c in self.selected_clients]
            
            # Aggregate backbone - source: line 382
            w_glob = FedAvg_noniid(w_locals, dict_len)
            self.global_model.load_state_dict(w_glob)
            
            # Aggregate g_aux - source: line 385
            self.g_aux = FedAvg_noniid_classifier(g_auxs, dict_len)
            
            # Evaluate
            self.evaluate(round_idx)
            
            # Time cost
            time_cost = time.time() - s_t
            self.Budget.append(time_cost)
            print(f"------------------------- time cost ------------------------- {time_cost:.2f}s")
        
        # Training complete
        print(f"\n{'='*50}")
        print("FedLoGe Training Complete")
        print(f"{'='*50}")
        
        # Save results
        self.save_results()
        self.save_global_model()
    
    def evaluate(self, round_idx=None):
        """
        Evaluate global and local accuracy
        Source: fedloge.py line 394-416 (globaltest) and line 403-416 (localtest)
        """
        # Evaluate on all clients
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        
        test_acc = sum(stats[2]) / sum(stats[1])
        train_loss = sum(stats_train[2]) / sum(stats_train[1])
        
        # Calculate AUC
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        
        self.rs_test_acc.append(test_acc)
        self.rs_train_loss.append(train_loss)
        
        # Global test accuracy (using g_head)
        global_acc, three_shot_acc = self._evaluate_global()
        self.rs_global_acc.append(global_acc)
        
        print(f"Averaged Train Loss: {train_loss:.4f}")
        print(f"Local Averaged Test Accuracy: {test_acc:.4f}")
        print(f"Global Test Accuracy: {global_acc:.4f}")
        print(f"Global 3-Shot Acc: [head: {three_shot_acc['head']:.4f}, "
              f"middle: {three_shot_acc['middle']:.4f}, tail: {three_shot_acc['tail']:.4f}]")
    
    def _evaluate_global(self):
        """
        Evaluate on global test set using g_head with 3-shot accuracy
        Source: update_baseline.py globaltest function line 1416-1475
        """
        if self.global_testloader is None:
            return 0.0, {"head": 0.0, "middle": 0.0, "tail": 0.0}
        
        self.global_model.eval()
        self.g_head.eval()
        correct = 0
        total = 0
        
        # 3-shot tracking
        correct_3shot = {"head": 0, "middle": 0, "tail": 0}
        total_3shot = {"head": 0, "middle": 0, "tail": 0}
        
        with torch.no_grad():
            for x, y in self.global_testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Extract features
                features = self._extract_features(x)
                
                # Use g_head for prediction
                output = self.g_head(features)
                
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
                # 3-shot accuracy calculation
                for i in range(len(y)):
                    label = y[i].item()
                    pred = predicted[i].item()
                    
                    if label in self.three_shot_dict["head"]:
                        total_3shot["head"] += 1
                        if pred == label:
                            correct_3shot["head"] += 1
                    elif label in self.three_shot_dict["middle"]:
                        total_3shot["middle"] += 1
                        if pred == label:
                            correct_3shot["middle"] += 1
                    else:  # tail
                        total_3shot["tail"] += 1
                        if pred == label:
                            correct_3shot["tail"] += 1
        
        # Calculate 3-shot accuracies
        three_shot_acc = {
            "head": correct_3shot["head"] / (total_3shot["head"] + 1e-10),
            "middle": correct_3shot["middle"] / (total_3shot["middle"] + 1e-10),
            "tail": correct_3shot["tail"] / (total_3shot["tail"] + 1e-10)
        }
        
        global_acc = correct / total if total > 0 else 0.0
        return global_acc, three_shot_acc
    
    def _extract_features(self, x):
        """Extract features from global model"""
        # Use FedLoGe-style model's latent_output parameter
        return self.global_model(x, latent_output=True)
    
    def test_metrics(self):
        """Test metrics for all selected clients"""
        num_samples = []
        tot_correct = []
        tot_auc = []
        
        for client in self.clients:
            # Set heads for testing
            client.set_heads(self.g_head, self.g_aux, self.l_heads[client.id])
            ct, ns, auc = client.test_metrics()
            tot_correct.append(ct)
            num_samples.append(ns)
            tot_auc.append(auc * ns)
        
        return [None, num_samples, tot_correct, tot_auc]
    
    def train_metrics(self):
        """Training metrics for all selected clients"""
        num_samples = []
        tot_loss = []
        
        for client in self.clients:
            client.set_heads(self.g_head, self.g_aux, self.l_heads[client.id])
            ls, ns = client.train_metrics()
            tot_loss.append(ls)
            num_samples.append(ns)
        
        return [None, num_samples, tot_loss]
    
    def save_results(self):
        """Save results to file"""
        algo = self.algorithm
        ts = time.strftime("%Y%m%d_%H%M%S")
        result_path = f"../results/{self.dataset}_{algo}_{ts}/"
        
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        # Save metrics
        import h5py
        file_path = result_path + f"{self.dataset}_{algo}_{ts}.h5"
        
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
            hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            hf.create_dataset('rs_global_acc', data=self.rs_global_acc)
        
        print(f"Results saved to: {file_path}")
        
        # Save plots
        self._save_plots(result_path, ts)
    
    def _save_plots(self, result_path, ts):
        """Save accuracy curves"""
        import matplotlib.pyplot as plt
        
        # Local accuracy
        plt.figure()
        plt.plot(range(len(self.rs_test_acc)), self.rs_test_acc)
        plt.xlabel('Rounds')
        plt.ylabel('Local Test Accuracy')
        plt.title(f'{self.algorithm} - Local Accuracy')
        plt.savefig(result_path + f"{self.dataset}_{self.algorithm}_{ts}.svg")
        plt.close()
        
        # Global accuracy
        plt.figure()
        plt.plot(range(len(self.rs_global_acc)), self.rs_global_acc)
        plt.xlabel('Rounds')
        plt.ylabel('Global Test Accuracy')
        plt.title(f'{self.algorithm} - Global Accuracy')
        plt.savefig(result_path + f"{self.dataset}_{self.algorithm}_{ts}_global.svg")
        plt.close()
        
        print(f"Plots saved to: {result_path}")
    
    def save_global_model(self):
        """Save global model"""
        model_path = f"models/{self.algorithm}_server.pt"
        torch.save({
            'model': self.global_model.state_dict(),
            'g_head': self.g_head.state_dict(),
            'g_aux': self.g_aux.state_dict()
        }, model_path)
        print(f"Global model saved to: {model_path}")
