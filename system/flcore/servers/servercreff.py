"""
CReFF: Classifier Re-training with Federated Features

Reference:
    Shang et al. "Federated Learning on Non-IID Data via Local and Global Distillation"
    https://github.com/shangxinyi/CReFF-FL
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

from flcore.servers.serverbase import Server
from flcore.clients.clientcreff import clientCREFF, Local
from flcore.trainmodel.resnet_cifar import resnet8_cifar
from utils.data_utils import read_client_data


class TensorDataset(Dataset):
    """Dataset wrapper for synthetic features"""
    def __init__(self, images, labels):
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def match_loss(gw_syn, gw_real, args):
    """Compute gradient matching loss between synthetic and real gradients"""
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)
    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)
    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
            torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)
    else:
        exit('DC error: unknown distance function')

    return dis


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:
        tmp = 'do nothing'
    elif len(shape) == 1:
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (
        torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    return dis_weight


class Global(object):
    """Global model for CReFF with synthetic feature learning"""
    def __init__(self,
                 num_classes: int,
                 device: str,
                 args,
                 num_of_feature):
        self.device = device
        self.num_classes = num_classes
        self.fedavg_acc = []
        self.fedavg_many = []
        self.fedavg_medium = []
        self.fedavg_few = []
        self.ft_acc = []
        self.ft_many = []
        self.ft_medium = []
        self.ft_few = []
        self.num_of_feature = num_of_feature
        self.feature_syn = torch.randn(size=(args.num_classes * self.num_of_feature, 256), dtype=torch.float,
                                       requires_grad=True, device=args.device)
        self.label_syn = torch.tensor([np.ones(self.num_of_feature) * i for i in range(args.num_classes)], dtype=torch.long,
                                      requires_grad=False, device=args.device).view(-1)
        self.optimizer_feature = SGD([self.feature_syn, ], lr=args.lr_feature)
        self.criterion = CrossEntropyLoss().to(args.device)
        self.syn_model = resnet8_cifar(num_classes=args.num_classes, scaling=4).to(device)
        self.feature_net = nn.Linear(256, num_classes).to(args.device)

    def update_feature_syn(self, args, global_params, list_clients_gradient):
        feature_net_params = self.feature_net.state_dict()
        for name_param in reversed(global_params):
            if name_param == 'classifier.bias':
                feature_net_params['bias'] = global_params[name_param]
            if name_param == 'classifier.weight':
                feature_net_params['weight'] = global_params[name_param]
                break
        self.feature_net.load_state_dict(feature_net_params)
        self.feature_net.train()
        net_global_parameters = list(self.feature_net.parameters())
        gw_real_all = {class_index: [] for class_index in range(self.num_classes)}
        for gradient_one in list_clients_gradient:
            for class_num, gradient in gradient_one.items():
                gw_real_all[class_num].append(gradient)
        gw_real_avg = {class_index: [] for class_index in range(args.num_classes)}
        for i in range(args.num_classes):
            gw_real_temp = []
            list_one_class_client_gradient = gw_real_all[i]

            if len(list_one_class_client_gradient) != 0:
                weight_temp = 1.0 / len(list_one_class_client_gradient)
                for name_param in range(2):
                    list_values_param = []
                    for one_gradient in list_one_class_client_gradient:
                        list_values_param.append(one_gradient[name_param] * weight_temp)
                    value_global_param = sum(list_values_param)
                    gw_real_temp.append(value_global_param)
                gw_real_avg[i] = gw_real_temp
        for ep in range(args.match_epoch):
            loss_feature = torch.tensor(0.0).to(args.device)
            for c in range(args.num_classes):
                if len(gw_real_avg[c]) != 0:
                    feature_syn = self.feature_syn[c * self.num_of_feature:(c + 1) * self.num_of_feature].reshape((self.num_of_feature, 256))
                    lab_syn = torch.ones((self.num_of_feature,), device=args.device, dtype=torch.long) * c
                    output_syn = self.feature_net(feature_syn)
                    loss_syn = self.criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_global_parameters, create_graph=True)
                    loss_feature += match_loss(gw_syn, gw_real_avg[c], args)
            self.optimizer_feature.zero_grad()
            loss_feature.backward()
            self.optimizer_feature.step()

    def feature_re_train(self, args, fedavg_params, batch_size_local_training):
        feature_syn_train_ft = copy.deepcopy(self.feature_syn.detach())
        label_syn_train_ft = copy.deepcopy(self.label_syn.detach())
        dst_train_syn_ft = TensorDataset(feature_syn_train_ft, label_syn_train_ft)
        ft_model = nn.Linear(256, self.num_classes).to(args.device)
        optimizer_ft_net = SGD(ft_model.parameters(), lr=args.lr_net)
        ft_model.train()
        for epoch in range(args.crt_epoch):
            trainloader_ft = DataLoader(dataset=dst_train_syn_ft,
                                        batch_size=batch_size_local_training,
                                        shuffle=True)
            for data_batch in trainloader_ft:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = ft_model(images)
                loss_net = self.criterion(outputs, labels)
                optimizer_ft_net.zero_grad()
                loss_net.backward()
                optimizer_ft_net.step()
        ft_model.eval()
        feature_net_params = ft_model.state_dict()
        for name_param in reversed(fedavg_params):
            if name_param == 'classifier.bias':
                fedavg_params[name_param] = feature_net_params['bias']
            if name_param == 'classifier.weight':
                fedavg_params[name_param] = feature_net_params['weight']
                break
        return copy.deepcopy(ft_model.state_dict()), copy.deepcopy(fedavg_params)

    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        return fedavg_global_params

    def global_eval(self, fedavg_params, data_test, batch_size_test):
        self.syn_model.load_state_dict(fedavg_params)
        self.syn_model.eval()
        with torch.no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.syn_model(images)
                _, predicts = torch.max(outputs, -1)
                num_corrects += torch.sum(predicts.cpu() == labels.cpu()).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return self.syn_model.state_dict()


class FedCReFF(Server):
    """CReFF Server: Classifier Re-training with Federated Features"""
    
    def __init__(self, args, times):
        self._set_creff_defaults(args)
        super().__init__(args, times)
        
        # CReFF-specific parameters
        self.lr_feature = args.lr_feature
        self.lr_net = args.lr_net
        self.num_of_feature = args.num_of_feature
        self.match_epoch = args.match_epoch
        self.crt_epoch = args.crt_epoch
        self.batch_real = args.batch_real
        self.dis_metric = args.dis_metric
        self.lr_local_training = args.lr_local_training
        self.batch_size_local_training = args.batch_size_local_training
        self.num_epochs_local_training = args.num_epochs_local_training
        
        self.set_slow_clients()
        self.set_clients(clientCREFF)
        
        # Preload client data for gradient computation
        self.list_client2data = []
        self.original_dict_per_client = {}
        for client_id in range(self.num_clients):
            train_data = read_client_data(self.dataset, client_id, is_train=True)
            self.list_client2data.append(train_data)
            class_counts = {}
            for _, label in train_data:
                class_counts[int(label)] = class_counts.get(int(label), 0) + 1
            self.original_dict_per_client[client_id] = class_counts
        
        self.creff_global = Global(
            num_classes=self.num_classes,
            device=self.device,
            args=args,
            num_of_feature=self.num_of_feature
        )
        
        temp_model = nn.Linear(256, self.num_classes).to(self.device)
        self.syn_params = temp_model.state_dict()
        
        print(f"\n{'='*60}")
        print(f"CReFF Configuration:")
        print(f"  num_clients: {self.num_clients}")
        print(f"  join_ratio: {self.join_ratio}")
        print(f"  global_rounds: {self.global_rounds}")
        print(f"  lr_local_training: {self.lr_local_training}")
        print(f"  lr_feature: {self.lr_feature}")
        print(f"  lr_net: {self.lr_net}")
        print(f"  num_of_feature: {self.num_of_feature}")
        print(f"  match_epoch: {self.match_epoch}")
        print(f"  crt_epoch: {self.crt_epoch}")
        print(f"{'='*60}\n")
    
    def _set_creff_defaults(self, args):
        """Set CReFF default parameters"""
        import re
        if not hasattr(args, 'num_clients') or args.num_clients is None or args.num_clients == 0:
            match = re.search(r'NC(\d+)', args.dataset)
            args.num_clients = int(match.group(1)) if match else 20
        
        if not hasattr(args, 'lr_local_training') or args.lr_local_training is None:
            args.lr_local_training = 0.1
        if not hasattr(args, 'lr_feature') or args.lr_feature is None:
            args.lr_feature = 1.0
        if not hasattr(args, 'lr_net') or args.lr_net is None:
            args.lr_net = 0.01
        if not hasattr(args, 'num_of_feature') or args.num_of_feature is None:
            args.num_of_feature = 100
        if not hasattr(args, 'match_epoch') or args.match_epoch is None:
            args.match_epoch = 100
        if not hasattr(args, 'crt_epoch') or args.crt_epoch is None:
            args.crt_epoch = 300
        if not hasattr(args, 'batch_real') or args.batch_real is None:
            args.batch_real = 32
        if not hasattr(args, 'batch_size_local_training') or args.batch_size_local_training is None:
            args.batch_size_local_training = args.batch_size if hasattr(args, 'batch_size') else 32
        if not hasattr(args, 'num_epochs_local_training') or args.num_epochs_local_training is None:
            args.num_epochs_local_training = args.local_epochs if hasattr(args, 'local_epochs') else 5
        if not hasattr(args, 'dis_metric') or args.dis_metric is None:
            args.dis_metric = 'ours'

    def train(self):
        """Main training loop"""
        random_state = np.random.RandomState(7)
        
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            
            global_params = self.creff_global.download_params()
            syn_feature_params = copy.deepcopy(global_params)
            for name_param in reversed(syn_feature_params):
                if name_param == 'classifier.bias':
                    syn_feature_params[name_param] = self.syn_params['bias']
                if name_param == 'classifier.weight':
                    syn_feature_params[name_param] = self.syn_params['weight']
                    break
            
            self.send_models()
            
            list_clients_gradient = []
            list_dicts_local_params = []
            list_nums_local_data = []
            
            for client in self.selected_clients:
                client_id = client.id
                data_client = self.list_client2data[client_id]
                list_nums_local_data.append(len(data_client))
                
                local_model = Local(
                    data_client=data_client,
                    class_list=self.original_dict_per_client[client_id],
                    args=self.args
                )
                truth_gradient = local_model.compute_gradient(copy.deepcopy(syn_feature_params), self.args)
                list_clients_gradient.append(copy.deepcopy(truth_gradient))
                
                local_params = local_model.local_train(self.args, copy.deepcopy(global_params))
                list_dicts_local_params.append(copy.deepcopy(local_params))
            
            fedavg_params = self.creff_global.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
            self.creff_global.update_feature_syn(self.args, copy.deepcopy(syn_feature_params), list_clients_gradient)
            self.syn_params, ft_params = self.creff_global.feature_re_train(
                self.args, copy.deepcopy(fedavg_params), self.batch_size_local_training
            )
            
            self.creff_global.syn_model.load_state_dict(copy.deepcopy(fedavg_params))
            self.global_model.load_state_dict(ft_params)
            self.Budget.append(time.time() - s_t)
            
            if i % self.eval_gap == 0:
                print(f"\n-------------Round {i}-------------")
                self.evaluate()
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])
        
        print("\n=== Training Complete ===")
        print(f"Average time cost per round: {sum(self.Budget)/len(self.Budget):.2f}s")
        
        self.save_results()
        self.save_global_model()
