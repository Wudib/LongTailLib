import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        # 不再固定种子，使用全局种子设置
        self.args = args
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        # 恢复为基础的交叉熵损失，不在基类做类别加权/平滑
        self.loss = nn.CrossEntropyLoss()
        # Use SGD with configurable momentum and weight decay for stronger local training
        _momentum = float(getattr(args, 'local_momentum', 0.0))
        _wd = getattr(args, 'weight_decay', None)
        if _wd is None:
            _wd = 0.0
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=_momentum, weight_decay=_wd
        )
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

        # 不在基类默认启用加权采样或FedProx，避免影响其它方法


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    # 移除基类中的类别计数工具，避免耦合方法逻辑
        
    def set_parameters(self, model):
        # Use load_state_dict to copy BOTH parameters AND buffers (e.g., BatchNorm running_mean/running_var)
        # This is critical for algorithms like CReFF that use BatchNorm
        # Previously only copied parameters(), missing buffers!
        self.model.load_state_dict(model.state_dict())

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        # 诊断：统计测试标签分布与输出有限性
        test_label_counts = np.zeros(self.num_classes, dtype=np.int64)
        nonfinite_output_batches = 0
        total_batches = 0
        
        with torch.no_grad():
            for x, y in testloaderfull:
                total_batches += 1
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                # 诊断：检测输出中是否存在 NaN/Inf
                if not torch.isfinite(output).all():
                    nonfinite_output_batches += 1

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                # 诊断：更新标签计数
                y_cpu = y.detach().cpu().numpy()
                uniq, cnts = np.unique(y_cpu, return_counts=True)
                for u, c in zip(uniq, cnts):
                    if 0 <= u < self.num_classes:
                        test_label_counts[u] += c

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        # 诊断：输出测试标签分布与非有限输出批次数量
        missing_classes = [int(i) for i in np.where(test_label_counts == 0)[0]]



        # 防御性处理：替换评估分数中的 NaN/Inf，避免 sklearn 抛错
        nonfinite_prob = (~np.isfinite(y_prob)).sum()
        if nonfinite_prob > 0:

            y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1e6, neginf=-1e6)

        try:
            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        except Exception as e:

            # 若仍异常，返回一个合理的默认值，避免训练流程中断
            auc = 0.0
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    @torch.no_grad()
    def eval_external_model_stats(self, external_model):
        """
        Evaluate an external/global model on this client's test set.

        Returns (correct, total). If the external model cannot be evaluated
        (e.g., it does not produce class logits), returns (0, 0).
        """
        try:
            testloader = self.load_test_data()
            external_model = external_model.to(self.device)
            external_model.eval()

            correct = 0
            total = 0
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                    x = x[0]
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = external_model(x)

                # Validate output shape: need class logits for accuracy
                if output.dim() < 2 or output.size(1) != self.num_classes:
                    # Incompatible output for accuracy computation
                    return 0, 0

                pred = torch.argmax(output, dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

            return correct, total
        except Exception:
            return 0, 0

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
