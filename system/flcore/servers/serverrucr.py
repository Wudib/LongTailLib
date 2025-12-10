"""
RUCR: Representation Unified Classifier Re-training for Class-Imbalanced Federated Learning

Reference:
    RUCR official implementation
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flcore.servers.serverbase import Server
from flcore.clients.clientrucr import clientRUCR


def get_mean(feat_dim, c_infos, nums, num_classes, device):
    """
    Compute global unified prototypes - source code utils.py get_mean()
    """
    real_info = dict()
    syn_info = [torch.zeros(feat_dim).to(device)] * num_classes
    nums = np.array(nums)
    cls_total = np.sum(nums, axis=0)
    
    for cls in range(num_classes):
        for c_idx, c_info in enumerate(c_infos):
            if cls not in c_info.keys():
                continue
            pre = real_info.get(cls, 0)
            real_info[cls] = pre + c_info[cls] * nums[c_idx][cls]
        if real_info.get(cls) is None:
            continue
        temp = real_info.get(cls)
        real_info[cls] = temp / cls_total[cls]

    for k, v in real_info.items():
        syn_info[k] = v
    syn_info = torch.stack(syn_info).to(device)
    return syn_info


def cal_norm_mean(c_means, c_dis, num_classes, device):
    """
    Calculate normalized mean prototypes - source code Global.cal_norm_mean()
    """
    glo_means = dict()
    c_dis = torch.tensor(c_dis).to(device)
    total_num_per_cls = c_dis.sum(dim=0)
    
    for i in range(num_classes):
        for c_idx, c_mean in enumerate(c_means):
            if i not in c_mean.keys():
                continue
            temp = glo_means.get(i, 0)
            glo_means[i] = temp + F.normalize(c_mean[i].view(1, -1), dim=1).view(-1) * c_dis[c_idx][i]
        if glo_means.get(i) is None:
            continue
        t = glo_means[i]
        glo_means[i] = t / total_num_per_cls[i]
    return glo_means


def model_fusion(list_dicts_local_params, list_nums_local_data):
    """FedAvg aggregation - source code utils.py model_fusion()"""
    local_params = copy.deepcopy(list_dicts_local_params[0])
    for name_param in list_dicts_local_params[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        local_params[name_param] = value_global_param
    return local_params


class FedRUCR(Server):
    """
    RUCR Server - Representation Unified Classifier Re-training
    Copied from source code Global class and CReFF() function
    """
    
    @staticmethod
    def register_cli_args(parser):
        """Register RUCR-specific command line arguments (GitHub recommended settings)"""
        parser.add_argument('--crt_ep', type=int, default=20,
                            help='RUCR: Number of rounds for classifier re-training (default: 20)')
        parser.add_argument('--feat_loss_arg', type=float, default=0.15,
                            help='RUCR: Feature contrastive loss weight (default: 0.15)')
        parser.add_argument('--lr_cls_balance', type=float, default=0.01,
                            help='RUCR: Classifier re-training learning rate (default: 0.01)')
        parser.add_argument('--local_bal_ep', type=int, default=50,
                            help='RUCR: Local balance epochs for classifier (default: 50)')
        parser.add_argument('--crt_feat_num', type=int, default=100,
                            help='RUCR: Number of synthetic features per class (default: 100)')
        parser.add_argument('--crt_batch_size', type=int, default=256,
                            help='RUCR: Batch size for classifier re-training (default: 256)')
        parser.add_argument('--uniform_left', type=float, default=0.35,
                            help='RUCR: Mixup left bound (default: 0.35)')
        parser.add_argument('--uniform_right', type=float, default=0.95,
                            help='RUCR: Mixup right bound (default: 0.95)')
        parser.add_argument('--times_arg', type=float, default=1.0,
                            help='RUCR: Class ratio scaling factor (default: 1.0)')
        parser.add_argument('--t', type=float, default=0.9,
                            help='RUCR: Temperature for contrastive loss (default: 0.9)')
    
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # RUCR specific parameters (GitHub recommended settings)
        self.crt_ep = getattr(args, 'crt_ep', 20)  # Number of rounds for classifier re-training
        self.feat_loss_arg = getattr(args, 'feat_loss_arg', 0.15)
        
        # Feature dimension from model
        self.feat_dim = self._get_feature_dim()
        
        # Create clients
        self.set_clients(clientRUCR)
        
        # Track class distribution per client
        self.original_dict_per_client = self._get_clients_class_distribution()
        
        print(f"\n{'='*60}")
        print(f"RUCR Configuration:")
        print(f"  num_clients: {self.num_clients}")
        print(f"  join_ratio: {self.join_ratio}")
        print(f"  global_rounds: {self.global_rounds}")
        print(f"  local_epochs: {self.local_epochs}")
        print(f"  crt_ep: {self.crt_ep}")
        print(f"  feat_loss_arg: {self.feat_loss_arg}")
        print(f"  feature_dim: {self.feat_dim}")
        print(f"{'='*60}\n")

    def _get_feature_dim(self):
        """Detect feature dimension from global model"""
        self.global_model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            try:
                output = self.global_model(dummy_input)
                if isinstance(output, tuple):
                    feature, _ = output
                    return feature.shape[1]
            except:
                pass
        return 256

    def _get_clients_class_distribution(self):
        """Get class distribution for all clients"""
        dict_per_client = []
        for client in self.clients:
            dict_per_client.append(client.class_compose)
        return dict_per_client

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

    def train(self):
        """
        Main training loop - copied from source code CReFF() function
        """
        for round_idx in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            
            # Send global model to selected clients
            self.send_models()
            
            # Determine which clients participate in centroid computation
            # In the last crt_ep rounds, all clients participate
            if self.crt_ep != 0 and self.crt_ep >= self.global_rounds - round_idx:
                real_clients = self.clients
            else:
                real_clients = self.selected_clients
            
            # Phase 1: Compute local centroids
            c_means = []
            c_fs = []
            ccvr_means = []
            c_dis = []
            
            for client in real_clients:
                # Set pre_model to global model
                client.pre_model.load_state_dict(self.global_model.state_dict())
                # Compute local centroids
                real_mean, c_f = client.get_local_centroid()
                ccvr_means.append(real_mean)
                
                if client in self.selected_clients:
                    c_means.append(real_mean)
                    c_fs.append(c_f)
                    c_dis.append(client.class_compose)
            
            # Compute global unified prototypes
            syn_mean = get_mean(self.feat_dim, c_means, c_dis, self.num_classes, self.device)
            global_dis = torch.tensor(c_dis).sum(0).to(self.device)
            
            # Phase 2: Local training with representation learning
            for c_id, client in enumerate(self.selected_clients):
                client.set_cls_ratio(global_dis)
                client.cls_syn_c = syn_mean
                client.cls_syn_c_norm = F.normalize(syn_mean, dim=1)
                client.train()
            
            # Phase 3: Aggregate models
            self.receive_models()
            self.aggregate_parameters()
            
            # Copy aggregated params for evaluation
            eval_params = copy.deepcopy(self.global_model.state_dict())
            
            # Phase 4: Classifier re-training (in last crt_ep rounds)
            if self.crt_ep >= self.global_rounds - round_idx:
                # Compute normalized global prototypes
                norm_means = cal_norm_mean(
                    ccvr_means, self.original_dict_per_client, 
                    self.num_classes, self.device
                )
                
                # Local classifier re-training
                mixup_cls_params = []
                list_nums_local_data = []
                for c_id, client in enumerate(self.selected_clients):
                    if c_id < len(c_fs):
                        mixup_cls_param = client.local_crt(norm_means, c_fs[c_id])
                        mixup_cls_params.append(mixup_cls_param)
                        list_nums_local_data.append(client.train_samples)
                
                # Aggregate classifiers
                if len(mixup_cls_params) > 0:
                    mixup_classifier = model_fusion(mixup_cls_params, list_nums_local_data)
                    # Update eval_params with re-trained classifier
                    for name_param in reversed(eval_params):
                        if name_param == 'classifier.bias':
                            eval_params[name_param] = mixup_classifier['bias']
                        if name_param == 'classifier.weight':
                            eval_params[name_param] = mixup_classifier['weight']
                            break
            
            # Load eval params for evaluation
            self.global_model.load_state_dict(eval_params)
            
            # Evaluate
            if round_idx % self.eval_gap == 0:
                print(f"\n-------------Round {round_idx}-------------")
                self.evaluate()
            
            self.Budget.append(time.time() - s_t)
            print(f"------------------------- time cost ------------------------- {self.Budget[-1]}")
        
        # Final output
        print("\n" + "="*50)
        print("RUCR Training Complete")
        print("="*50)
        
        print(f"\nBest Global Accuracy: {max(self.rs_global_acc) if self.rs_global_acc else 'N/A'}")
        print(f"Best Local Accuracy: {max(self.rs_test_acc) if self.rs_test_acc else 'N/A'}")
        print(f"Average time cost per round: {sum(self.Budget)/len(self.Budget):.2f}s")
        
        self.save_results()
        self.save_global_model()

    def send_models(self):
        """Send global model to selected clients"""
        for client in self.selected_clients:
            client.set_parameters(self.global_model)

    def receive_models(self):
        """Receive models from clients"""
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(copy.deepcopy(client.model))

    def aggregate_parameters(self):
        """FedAvg aggregation"""
        total_samples = sum(self.uploaded_weights)
        
        # Initialize with zeros
        global_params = copy.deepcopy(self.global_model.state_dict())
        for key in global_params:
            global_params[key] = torch.zeros_like(global_params[key], dtype=torch.float32)
        
        # Weighted average
        for w, model in zip(self.uploaded_weights, self.uploaded_models):
            model_params = model.state_dict()
            for key in global_params:
                global_params[key] += model_params[key].float() * (w / total_samples)
        
        # Convert back to original dtype
        original_state = self.global_model.state_dict()
        for key in global_params:
            global_params[key] = global_params[key].to(original_state[key].dtype)
        
        self.global_model.load_state_dict(global_params)

    def evaluate(self, acc=None, loss=None):
        """Evaluate and print metrics"""
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / len(stats[3])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = stats[3]
        
        self.rs_test_acc.append(test_acc)
        self.rs_test_auc.append(test_auc)
        self.rs_train_loss.append(train_loss)

        print(f"Averaged Train Loss: {train_loss:.4f}")
        print(f"Local Averaged Test Accuracy: {test_acc:.4f}")
        print(f"Averaged Test AUC: {test_auc:.4f}")
        print(f"Std Test Accuracy: {np.std(accs):.4f}")
        print(f"Std Test AUC: {np.std(aucs):.4f}")
        
        # Global evaluation
        global_acc = self.global_eval()
        self.rs_global_acc.append(global_acc)
        print(f"Global Averaged Test Accuracy: {global_acc:.4f}")

    def global_eval(self):
        """Evaluate on global test set"""
        if self.global_testloader is None:
            return 0.0
            
        self.global_model.eval()
        num_corrects = 0
        total = 0
        
        with torch.no_grad():
            for x, y in self.global_testloader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.global_model(x)
                
                if isinstance(output, tuple):
                    _, output = output
                    
                _, predicts = torch.max(output, -1)
                num_corrects += (predicts == y).sum().item()
                total += y.size(0)
        
        return num_corrects / total if total > 0 else 0.0

    def test_metrics(self):
        """Collect test metrics from all clients"""
        num_samples = []
        tot_correct = []
        tot_auc = []
        
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct)
            tot_auc.append(auc)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        """Collect training metrics from selected clients"""
        num_samples = []
        losses = []
        
        for c in self.selected_clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl)

        ids = [c.id for c in self.selected_clients]
        return ids, num_samples, losses


# Import needed for set_clients
from utils.data_utils import read_client_data
