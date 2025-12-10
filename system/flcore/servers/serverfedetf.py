"""
FedETF: Federated Learning with ETF Classifier
for Class-Imbalanced Learning

Reference:
    FedETF official implementation (FedETF-main/server_funct.py, main.py)
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flcore.servers.serverbase import Server
from flcore.clients.clientfedetf import clientFedETF


class FedETF(Server):
    """
    FedETF Server - Federated ETF Classifier
    Copied from source code server_funct.py and main.py
    
    Key features:
    - Uses ETF (Equiangular Tight Frame) classifier
    - Balanced softmax loss for class-imbalanced learning
    - FedAvg aggregation for model parameters
    - Proto sharing across clients
    """
    
    @staticmethod
    def add_args(parser):
        """Register FedETF-specific command line arguments - source code args.py"""
        parser.add_argument('--scaling_train', type=float, default=1.0,
                            help='FedETF: scaling hyperparameter for training (default: 1.0)')
        parser.add_argument('--fedetf_lr', type=float, default=0.04,
                            help='FedETF: learning rate (source default: 0.04)')
        parser.add_argument('--fedetf_momentum', type=float, default=0.9,
                            help='FedETF: SGD momentum (source default: 0.9)')
        parser.add_argument('--fedetf_wd', type=float, default=5e-4,
                            help='FedETF: weight decay (source default: 5e-4)')
        parser.add_argument('--fedetf_local_epochs', type=int, default=3,
                            help='FedETF: local epochs E (source default: 3)')
        parser.add_argument('--fedetf_batch_size', type=int, default=128,
                            help='FedETF: batch size (source default: 128)')
        parser.add_argument('--fedetf_join_ratio', type=float, default=1.0,
                            help='FedETF: client selection ratio (source default: 1.0)')
    
    def __init__(self, args, times):
        # Override with FedETF source code defaults BEFORE parent init
        # Source: args.py: E=3, batchsize=128, select_ratio=1.0
        args.local_epochs = getattr(args, 'fedetf_local_epochs', 3)
        args.batch_size = getattr(args, 'fedetf_batch_size', 128)
        args.join_ratio = getattr(args, 'fedetf_join_ratio', 1.0)
        
        super().__init__(args, times)
        
        # FedETF specific parameters
        self.scaling_train = getattr(args, 'scaling_train', 1.0)
        
        # FedETF hyperparameters from source code args.py
        self.fedetf_lr = getattr(args, 'fedetf_lr', 0.04)
        self.fedetf_momentum = getattr(args, 'fedetf_momentum', 0.9)
        self.fedetf_wd = getattr(args, 'fedetf_wd', 5e-4)
        
        # Create clients (after setting hyperparameters)
        self.set_clients(clientFedETF)
        
        print(f"\n{'='*60}")
        print(f"FedETF Configuration (source code defaults):")
        print(f"  num_clients: {self.num_clients}")
        print(f"  join_ratio: {self.join_ratio} (source: 1.0)")
        print(f"  global_rounds: {self.global_rounds}")
        print(f"  local_epochs: {self.local_epochs} (source: 3)")
        print(f"  batch_size: {self.batch_size} (source: 128)")
        print(f"  lr: {self.fedetf_lr} (source: 0.04)")
        print(f"  momentum: {self.fedetf_momentum} (source: 0.9)")
        print(f"  weight_decay: {self.fedetf_wd} (source: 5e-4)")
        print(f"  scaling_train: {self.scaling_train} (source: 1.0)")
        print(f"{'='*60}\n")
    
    def set_clients(self, clientObj):
        """Initialize clients"""
        from utils.data_utils import read_client_data
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
    
    def train(self):
        """
        Main training loop - source code main.py line 115-149
        """
        for round_idx in range(self.global_rounds):
            s_t = time.time()
            
            # Select clients
            self.selected_clients = self.select_clients()
            
            # Send global model to selected clients (including proto)
            self.send_models()
            
            # Local training on selected clients
            for client in self.selected_clients:
                client.train()
            
            # Receive and aggregate models
            self.receive_models()
            self.aggregate_parameters()
            
            # Evaluate
            if round_idx % self.eval_gap == 0:
                print(f"\n-------------Round {round_idx}-------------")
                self.evaluate()
            
            self.Budget.append(time.time() - s_t)
            print(f"------------------------- time cost ------------------------- {self.Budget[-1]:.2f}s")
        
        # Final output
        print("\n" + "="*50)
        print("FedETF Training Complete")
        print("="*50)
        
        if self.rs_global_acc:
            print(f"\nBest Global Accuracy: {max(self.rs_global_acc):.4f}")
        if self.rs_test_acc:
            print(f"Best Local Accuracy: {max(self.rs_test_acc):.4f}")
        print(f"Average time cost per round: {sum(self.Budget)/len(self.Budget):.2f}s")
        
        self.save_results()
        self.save_global_model()
    
    def send_models(self):
        """
        Send global model and proto to selected clients
        Source code: client_funct.py line 13-29 (receive_server_model)
        """
        for client in self.selected_clients:
            # Send model parameters
            client.set_parameters(self.global_model)
            
            # Send proto (ETF classifier prototypes)
            if hasattr(self.global_model, 'proto_classifier'):
                client.receive_proto(self.global_model.proto_classifier.proto)
    
    def receive_models(self):
        """
        Receive models from clients
        """
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(copy.deepcopy(client.model))
    
    def aggregate_parameters(self):
        """
        FedAvg aggregation - source code server_funct.py line 122-130
        """
        total_samples = sum(self.uploaded_weights)
        
        # Calculate aggregation weights
        agg_weights = [w / total_samples for w in self.uploaded_weights]
        
        # Get model parameters
        client_params = [model.state_dict() for model in self.uploaded_models]
        
        # FedAvg aggregation
        avg_params = copy.deepcopy(client_params[0])
        for name in avg_params:
            avg_params[name] = torch.zeros_like(avg_params[name], dtype=torch.float32)
            for params, weight in zip(client_params, agg_weights):
                avg_params[name] += params[name].float() * weight
            # Convert back to original dtype
            avg_params[name] = avg_params[name].to(client_params[0][name].dtype)
        
        self.global_model.load_state_dict(avg_params)
    
    def evaluate(self, acc=None, loss=None):
        """
        Evaluate and print metrics
        """
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / len(stats[3])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = stats[3]

        self.rs_test_acc.append(test_acc)
        self.rs_train_loss.append(train_loss)
        
        # Global evaluation
        global_acc = self._evaluate_global()
        self.rs_global_acc.append(global_acc)

        print(f"Averaged Train Loss: {train_loss:.4f}")
        print(f"Local Averaged Test Accuracy: {test_acc:.4f}")
        print(f"Averaged Test AUC: {test_auc:.4f}")
        print(f"Std Test Accuracy: {np.std(accs):.4f}")
        print(f"Std Test AUC: {np.std(aucs):.4f}")
        print(f"Global Averaged Test Accuracy: {global_acc:.4f}")
    
    def _evaluate_global(self):
        """
        Evaluate on global test set using ETF classifier
        """
        if self.global_testloader is None:
            return 0.0
        
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in self.global_testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.global_model(x)
                
                if isinstance(output, tuple) and len(output) >= 2:
                    feature = output[0]
                    # Use ETF prediction
                    if hasattr(self.global_model, 'proto_classifier'):
                        proto = self.global_model.proto_classifier.proto
                        output = torch.matmul(feature, proto)
                        output = self.global_model.scaling_train * output
                    else:
                        output = output[1]
                
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def test_metrics(self):
        """
        Gather test metrics from all clients
        """
        num_samples = []
        tot_correct = []
        tot_auc = []
        
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct)
            tot_auc.append(auc)
            num_samples.append(ns)
        
        ids = [c.id for c in self.clients]
        return (ids, num_samples, tot_correct, tot_auc)
    
    def train_metrics(self):
        """
        Gather training metrics from selected clients
        """
        num_samples = []
        losses = []
        
        for c in self.selected_clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl)
        
        ids = [c.id for c in self.selected_clients]
        return (ids, num_samples, losses)
