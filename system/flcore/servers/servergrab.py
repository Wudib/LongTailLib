"""
FedGraB: Federated Long-Tailed Learning with Self-Adjusting Gradient Balancer
Direct copy from FedGraB-main/fed_grab.py

Reference:
    FedGraB-main/fed_grab.py - main training loop
    FedGraB-main/util/update_baseline.py - LocalUpdate.update_weights_pid
    FedGraB-main/util/fedavg.py - FedAvg_Rod, FedAvg_noniid
    FedGraB-main/util/losses.py - PIDLOSS
    FedGraB-main/options.py - default parameters
"""

import os
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

from flcore.clients.clientgrab import ClientGRAB
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from flcore.trainmodel.resnet_cifar import resnet18_cifar


# ============================================================================
# PIDLOSS - Direct copy from FedGraB-main/util/losses.py line 38-352
# ============================================================================

def random_unit(p: float):
    """Source: losses.py line 576-586"""
    if p == 0:
        return False
    if p == 1:
        return True
    R = random.random()
    if R < p:
        return True
    else:
        return False


class Hook:
    """Source: losses.py line 355-379"""
    def __init__(self):
        self.m_count = 0
        self.input_grad_list = []
        self.output_grad_list = []
        self.gradient = None
        self.gradient_list = []

    def has_gradient(self):
        return self.gradient is not None

    def get_gradient(self):
        return self.gradient

    def hook_func_tensor(self, grad):
        grad = copy.deepcopy(grad)
        self.gradient = grad.cpu().numpy().tolist()
        self.m_count += 1


class PID:
    """Source: losses.py line 382-490"""
    def __init__(self):
        self.mode = "PID_DELTA"
        self.Kp = 10
        self.Ki = 0.01
        self.Kd = 0.1
        self.max_out = 100
        self.max_iout = 100
        self.set = 0
        self.current_value = 0
        self.out = 0
        self.Pout = 0
        self.Iout = 0
        self.Dout = 0
        self.Dbuf = [0, 0, 0]
        self.error = [0, 0, 0]
        self.m_open = False

    def reset(self):
        self.current_value = 0
        self.out = 0
        self.Pout = 0
        self.Iout = 0
        self.Dout = 0
        self.Dbuf = [0, 0, 0]
        self.error = [0, 0, 0]
        self.m_open = False

    def open(self):
        self.m_open = True

    def close(self):
        self.m_open = False

    def is_open(self):
        return self.m_open

    def PID_calc(self, current_value, set_value):
        if self.m_open == False:
            return torch.Tensor([0.])

        self.error[2] = self.error[1]
        self.error[1] = self.error[0]
        self.set_value = set_value
        self.current_value = current_value
        self.error[0] = set_value - current_value

        if self.mode == "PID_DELTA":
            self.Pout = self.Kp * (self.error[0] - self.error[1])
            self.Iout = self.Ki * self.error[0]
            self.Dbuf[2] = self.Dbuf[1]
            self.Dbuf[1] = self.Dbuf[0]
            self.Dbuf[0] = self.error[0] - 2.0 * self.error[1] + self.error[2]
            self.Dout = self.Kd * self.Dbuf[0]
            self.out += self.Pout + self.Iout + self.Dout
            self.LimitMax(self.out, self.max_out)

        return self.out

    def LimitMax(self, input, max):
        if input > max:
            input = max
        elif input < -max:
            input = -max


class PIDLOSS(nn.Module):
    """Source: losses.py line 38-352"""
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=10,
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 pidmask=["head"],
                 vis_grad=False,
                 test_with_obj=True,
                 device='cpu',
                 class_activation=False):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True
        self.hook = Hook()
        self.controllers = [PID() for _ in range(self.num_classes)]
        self.pidmask = pidmask
        self.class_activation = class_activation
        self.class_acti_mask = None

        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        self.register_buffer('pos_neg', torch.ones(self.num_classes) * 100)
        self.register_buffer('pn_diff', torch.zeros(self.num_classes))
        self.pos_grad = self.pos_grad.to(device)
        self.neg_grad = self.neg_grad.to(device)
        self.pos_neg = self.pos_neg.to(device)
        self.pn_diff = self.pn_diff.to(device)

        self.ce_layer = nn.CrossEntropyLoss()
        self.test_with_obj = test_with_obj

        self.head_class = []
        self.middle_class = []
        self.tail_class = []

        def _func(x):
            return (10 / 9) / ((1 / 9) + torch.exp(-0.5 * x))
        self.map_func = partial(_func)

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        self.n_i, self.n_c = cls_score.size()
        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        self.target = expand_label(cls_score, label)
        self.pos_w, self.neg_w = self.get_weight(self.target)
        self.weight = self.pos_w * self.target + self.neg_w * (1 - self.target)

        if self.class_activation:
            if self.class_acti_mask is None:
                self.class_acti_mask = cls_score.new_ones(self.n_i, self.n_c)
                for i in range(self.n_c):
                    if "head" not in self.pidmask and i in self.head_class:
                        self.class_acti_mask[torch.arange(self.n_i), i] = 0
                    if "middle" not in self.pidmask and i in self.middle_class:
                        self.class_acti_mask[torch.arange(self.n_i), i] = 0
                    if "tail" not in self.pidmask and i in self.tail_class:
                        self.class_acti_mask[torch.arange(self.n_i), i] = 0
            else:
                for i in range(label.shape[0]):
                    one_class = label[i]
                    if "head" not in self.pidmask and one_class in self.head_class:
                        self.class_acti_mask[torch.arange(self.n_i), one_class] = 1
                        self.controllers[one_class].open()
                    if "middle" not in self.pidmask and one_class in self.middle_class:
                        self.class_acti_mask[torch.arange(self.n_i), one_class] = 1
                        self.controllers[one_class].open()
                    if "tail" not in self.pidmask and one_class in self.tail_class:
                        self.class_acti_mask[torch.arange(self.n_i), one_class] = 1
                        self.controllers[one_class].open()

            self.weight *= self.class_acti_mask[0:self.n_i, :]

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, self.target, reduction='none')
        cls_loss = torch.sum(cls_loss) / self.n_i
        hook_handle = cls_score.register_hook(self.hook_func_tensor)

        return self.loss_weight * cls_loss

    def hook_func_tensor(self, grad):
        batchsize = grad.shape[0]
        classes_num = grad.shape[1]

        tail_length = len(self.tail_class)
        img_max = 1
        prob_dist = []
        for cls_idx in range(tail_length):
            prob = img_max * (0.1**(cls_idx / (tail_length - 1.0 + 1e-10)))
            prob_dist.append(prob)

        select_record = []
        tail_id = 0
        for c_id in range(classes_num):
            if c_id in self.tail_class:
                if tail_id < len(prob_dist) and random_unit(prob_dist[tail_id]):
                    self.weight[torch.arange(batchsize), c_id] = 1
                    tail_id += 1
                    select_record.append(c_id)

        grad *= self.weight

        target_temp = self.target.detach()
        grad_temp = grad.detach()
        grad_temp = torch.abs(grad_temp)

        for c_id in range(classes_num):
            if c_id in select_record:
                grad_temp[torch.arange(batchsize), c_id] = 0

        pos_grad = torch.sum(grad_temp * target_temp, dim=0)
        neg_grad = torch.sum(grad_temp * (1 - target_temp), dim=0)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-20)
        self.pn_diff = self.pos_grad - self.neg_grad

    def get_3shotclass(self, head_class, middle_class, tail_class):
        self.head_class = head_class
        self.middle_class = middle_class
        self.tail_class = tail_class

    def apply_3shot_mask(self):
        if "head" in self.pidmask:
            for i in self.head_class:
                self.controllers[i].reset()
                self.controllers[i].close()
        else:
            for i in self.head_class:
                self.controllers[i].reset()
                self.controllers[i].open()

        if "middle" in self.pidmask:
            for i in self.middle_class:
                self.controllers[i].reset()
                self.controllers[i].close()
        else:
            for i in self.middle_class:
                self.controllers[i].reset()
                self.controllers[i].open()

        if "tail" in self.pidmask:
            for i in self.tail_class:
                self.controllers[i].reset()
                self.controllers[i].close()
        else:
            for i in self.tail_class:
                self.controllers[i].reset()
                self.controllers[i].open()

    def apply_class_activation(self):
        if self.class_activation:
            if "head" not in self.pidmask:
                for i in self.head_class:
                    self.controllers[i].reset()
                    self.controllers[i].close()
            if "middle" not in self.pidmask:
                for i in self.middle_class:
                    self.controllers[i].reset()
                    self.controllers[i].close()
            if "tail" not in self.pidmask:
                for i in self.tail_class:
                    self.controllers[i].reset()
                    self.controllers[i].close()

    def get_weight(self, target):
        pos_w = target.new_zeros(self.num_classes)
        neg_w = target.new_zeros(self.num_classes)
        for i in range(self.num_classes):
            pid_out = self.controllers[i].PID_calc(self.pn_diff[i], 0)
            if 0 - self.pn_diff[i] > 0:
                pos_w[i] = self.map_func(pid_out)
                neg_w[i] = self.map_func(-pid_out)
            else:
                pos_w[i] = self.map_func(pid_out)
                neg_w[i] = self.map_func(-pid_out)
        return pos_w, neg_w


# ============================================================================
# FedAvg functions - Direct copy from FedGraB-main/util/fedavg.py
# ============================================================================

def FedAvg_noniid(w, dict_len):
    """Source: fedavg.py line 54-63"""
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * dict_len[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
        w_avg[k] = w_avg[k] / sum(dict_len)
    return w_avg


def FedAvg_Rod(backbone_w_locals, linear_w_locals, dict_len):
    """Source: fedavg.py line 65-68"""
    backbone_w_avg = FedAvg_noniid(backbone_w_locals, dict_len)
    linear_w_avg = FedAvg_noniid(linear_w_locals, dict_len)
    return backbone_w_avg, linear_w_avg


# ============================================================================
# FedGraB Server - Direct copy from FedGraB-main/fed_grab.py
# ============================================================================

class FedGraB(Server):
    """
    FedGraB Server - Direct copy from FedGraB-main/fed_grab.py
    
    Source: fed_grab.py line 41-114
    """
    
    def __init__(self, args, times):
        # ========== Source: options.py defaults ==========
        # line 6: rounds = 500
        # line 7: local_ep = 5
        # line 8: frac = 0.2 -> BUT fed_grab.py line 69 sets frac = 1
        # line 9: num_users = 40
        # line 10: local_bs = 10
        # line 11: lr = 0.03
        # line 12: momentum = 0.5
        # line 20: seed = 1
        
        args.num_clients = 40  # options.py line 9
        args.join_ratio = 1.0  # fed_grab.py line 69: args.frac = 1
        args.batch_size = 10   # options.py line 10
        args.local_learning_rate = 0.03  # options.py line 11
        args.local_epochs = 5  # options.py line 7
        
        # Use ResNet18 for CIFAR (feature_dim=512) from resnet_cifar.py
        # This matches FedGraB-main/model/model_res.py ResNet18 exactly:
        # - 4 stages: 64 -> 128 -> 256 -> 512
        # - 3x3 conv stem, no maxpool (CIFAR-optimized)
        # - num_blocks = [2, 2, 2, 2]
        from flcore.trainmodel.models import BaseHeadSplit
        print(f"\n[FedGraB] Using resnet18_cifar from resnet_cifar.py (feature_dim=512)")
        base_model = resnet18_cifar(num_classes=args.num_classes)
        args.head = copy.deepcopy(base_model.linear)
        base_model.linear = nn.Identity()
        args.model = BaseHeadSplit(base_model, args.head).to(args.device)
        
        super().__init__(args, times)
        
        # Source: fed_grab.py line 62-65
        block_expansion = 1  # ResNet18
        self.feature_dim = 512 * block_expansion
        
        # g_backbone is self.global_model.base
        # g_classifier - Source: fed_grab.py line 65
        self.g_classifier = nn.Linear(self.feature_dim, self.num_classes).to(self.device)
        
        # g_linears - one per client - Source: fed_grab.py line 74-76
        self.g_linears = [nn.Linear(self.feature_dim, self.num_classes).to(self.device) 
                         for _ in range(self.num_clients)]
        
        # g_pid_losses - one per client - Source: fed_grab.py line 78
        self.g_pid_losses = [PIDLOSS(device=self.device, 
                                      num_classes=self.num_classes,
                                      pidmask=["head", "middle"], 
                                      class_activation=False) 
                            for _ in range(self.num_clients)]
        
        # Initialize PIDLOSS - Source: fed_grab.py line 80-86
        all_classes = list(range(self.num_classes))
        for idx in range(self.num_clients):
            self.g_pid_losses[idx].get_3shotclass(head_class=[], middle_class=[], tail_class=all_classes)
            self.g_pid_losses[idx].apply_3shot_mask()
            self.g_pid_losses[idx].apply_class_activation()
        
        # Create clients
        self.set_slow_clients()
        self.set_clients(ClientGRAB)
        
        # Pass PIDLOSS and classifier to each client
        for idx, client in enumerate(self.clients):
            client.set_pid_loss(self.g_pid_losses[idx])
            client.set_classifier(self.g_linears[idx])
        
        # Tracking
        self.rs_global_3shot = []
        
        print(f"\n{'='*60}")
        print(f"FedGraB Configuration (source: options.py + fed_grab.py):")
        print(f"  num_clients (num_users): {self.num_clients} (source: 40)")
        print(f"  join_ratio (frac): {args.join_ratio} (source: 1.0, fed_grab.py line 69)")
        print(f"  batch_size (local_bs): {args.batch_size} (source: 10)")
        print(f"  learning_rate (lr): {args.local_learning_rate} (source: 0.03)")
        print(f"  momentum: 0.5 (source: 0.5)")
        print(f"  local_epochs (local_ep): {args.local_epochs} (source: 5)")
        print(f"  global_rounds (rounds): {self.global_rounds}")
        print(f"  feature_dim: {self.feature_dim}")
        print(f"{'='*60}\n")
    
    def train(self):
        """
        Main training loop - Direct copy from fed_grab.py line 89-113
        """
        for rnd in range(1, self.global_rounds + 1):
            s_t = time.time()
            
            # Source: fed_grab.py line 90
            backbone_w_locals, linear_w_locals = [], []
            
            # Source: fed_grab.py line 92: for idx in range(args.num_users)
            print(f"\n--- Round {rnd}/{self.global_rounds} ---")
            for idx in tqdm(range(self.num_clients), desc='Local training'):
                # Source: fed_grab.py line 94
                # local.update_weights_pid(net=copy.deepcopy(g_backbone), ...)
                backbone_w_local, linear_w_local = self.clients[idx].train(
                    g_backbone=copy.deepcopy(self.global_model.base)
                )
                
                # Source: fed_grab.py line 95-96
                backbone_w_locals.append(copy.deepcopy(backbone_w_local))
                linear_w_locals.append(copy.deepcopy(linear_w_local))
            
            # Source: fed_grab.py line 100
            dict_len = [self.clients[idx].train_samples for idx in range(self.num_clients)]
            
            # Source: fed_grab.py line 102
            backbone_w_avg, linear_w_avg = FedAvg_Rod(backbone_w_locals, linear_w_locals, dict_len)
            
            # Source: fed_grab.py line 106-107
            self.global_model.base.load_state_dict(copy.deepcopy(backbone_w_avg))
            self.g_classifier.load_state_dict(copy.deepcopy(linear_w_avg))
            
            # Source: fed_grab.py line 108 - globaltest
            global_acc, global_3shot = self._globaltest()
            self.rs_global_acc.append(global_acc)
            self.rs_global_3shot.append(global_3shot)
            
            # Source: fed_grab.py line 110 - broadcast global classifier to all clients
            self.g_linears = [copy.deepcopy(self.g_classifier) for _ in range(self.num_clients)]
            for idx, client in enumerate(self.clients):
                client.set_classifier(self.g_linears[idx])
            
            # Evaluate local metrics
            self._evaluate_local()
            
            # Time cost
            time_cost = time.time() - s_t
            self.Budget.append(time_cost)
            
            # Print - Source: fed_grab.py line 112-113
            print(f"Round {rnd}, Global Test Acc: {global_acc:.4f}")
            print(f"Round {rnd}, Global 3shot: [head: {global_3shot['head']:.4f}, middle: {global_3shot['middle']:.4f}, tail: {global_3shot['tail']:.4f}]")
            print(f"Time cost: {time_cost:.2f}s")
        
        # Training complete
        print(f"\n{'='*60}")
        print("FedGraB Training Complete")
        print(f"Best Global Acc: {max(self.rs_global_acc):.4f}")
        print(f"{'='*60}\n")
        
        self.save_results()
        self.save_global_model()
    
    @torch.no_grad()
    def _globaltest(self):
        """
        Global test - Source: update_baseline.py line 175-231 (globaltest function)
        """
        # Three-shot split (simplified: head=[0,1], middle=[2,3,4,5], tail=[6,7,8,9])
        # Source: update_baseline.py line 178-180
        three_shot_dict = {
            "head": [0, 1],
            "middle": [2, 3, 4, 5],
            "tail": [6, 7, 8, 9]
        }
        
        correct_3shot = {"head": 0, "middle": 0, "tail": 0}
        total_3shot = {"head": 0, "middle": 0, "tail": 0}
        
        self.global_model.base.eval()
        self.g_classifier.eval()
        
        correct = 0
        total = 0
        
        # Use global test loader
        if self.global_testloader is not None:
            for images, labels in self.global_testloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Source: update_baseline.py line 197-198
                feat = self.global_model.base(images)
                outputs = self.g_classifier(feat)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 3-shot counting - Source: update_baseline.py line 210-223
                for label in labels:
                    label_int = int(label.item())
                    if label_int in three_shot_dict["head"]:
                        total_3shot["head"] += 1
                    elif label_int in three_shot_dict["middle"]:
                        total_3shot["middle"] += 1
                    else:
                        total_3shot["tail"] += 1
                
                for i in range(len(predicted)):
                    if predicted[i] == labels[i]:
                        label_int = int(labels[i].item())
                        if label_int in three_shot_dict["head"]:
                            correct_3shot["head"] += 1
                        elif label_int in three_shot_dict["middle"]:
                            correct_3shot["middle"] += 1
                        else:
                            correct_3shot["tail"] += 1
        
        acc = correct / total if total > 0 else 0
        acc_3shot = {
            "head": correct_3shot["head"] / (total_3shot["head"] + 1e-10),
            "middle": correct_3shot["middle"] / (total_3shot["middle"] + 1e-10),
            "tail": correct_3shot["tail"] / (total_3shot["tail"] + 1e-10)
        }
        
        return acc, acc_3shot
    
    def _evaluate_local(self):
        """Evaluate local metrics"""
        total_samples = 0
        total_correct = 0
        total_loss = 0.0
        
        for client in self.clients:
            stats = client.test_metrics()
            total_samples += stats['num_samples']
            total_correct += stats['num_correct']
            total_loss += stats['loss'] * stats['num_samples']
        
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        self.rs_test_acc.append(avg_acc)
        self.rs_train_loss.append(avg_loss)
        self.rs_test_auc.append(0.0)
        
        print(f"Averaged Train Loss: {avg_loss:.4f}")
        print(f"Averaged Local Test Acc: {avg_acc:.4f}")
    
    def save_results(self):
        """Save results with 3-shot metrics"""
        ts = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.dataset}_{self.algorithm}_{ts}"
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        result_root = os.path.normpath(os.path.join(base_dir, '..', '..', '..', 'results'))
        run_dir = os.path.join(result_root, base_name)
        os.makedirs(run_dir, exist_ok=True)
        
        file_path = os.path.join(run_dir, f"{base_name}.h5")
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
            hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            hf.create_dataset('rs_global_acc', data=self.rs_global_acc)
            if self.rs_global_3shot:
                hf.create_dataset('rs_global_head', data=[s['head'] for s in self.rs_global_3shot])
                hf.create_dataset('rs_global_middle', data=[s['middle'] for s in self.rs_global_3shot])
                hf.create_dataset('rs_global_tail', data=[s['tail'] for s in self.rs_global_3shot])
        
        print(f"File path: {file_path}")
        
        # Plot curves
        self._plot_curves(run_dir, base_name)
    
    def _plot_curves(self, run_dir, base_name):
        """Generate accuracy curves"""
        try:
            rounds = list(range(len(self.rs_global_acc)))
            
            plt.figure(figsize=(10, 6))
            plt.plot(rounds, self.rs_global_acc, label='Global Acc', linewidth=2)
            if self.rs_global_3shot:
                plt.plot(rounds, [s['head'] for s in self.rs_global_3shot], label='Head', linestyle='--')
                plt.plot(rounds, [s['middle'] for s in self.rs_global_3shot], label='Middle', linestyle='--')
                plt.plot(rounds, [s['tail'] for s in self.rs_global_3shot], label='Tail', linestyle='--')
            plt.xlabel('Rounds')
            plt.ylabel('Test Accuracy')
            plt.title('FedGraB Global Test Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            svg_path = os.path.join(run_dir, f"{base_name}_global.svg")
            plt.savefig(svg_path, format='svg')
            plt.close()
            print(f"Saved curve: {svg_path}")
        except Exception as e:
            print(f"Warning: Failed to generate plots: {e}")
