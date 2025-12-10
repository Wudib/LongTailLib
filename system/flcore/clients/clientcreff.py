"""
CReFF: Classifier Re-training with Federated Features

Reference:
    Shang et al. "Federated Learning on Non-IID Data via Local and Global Distillation"
    https://github.com/shangxinyi/CReFF-FL
"""

from torchvision.transforms import transforms
import numpy as np
from torch import unsqueeze
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from flcore.clients.clientbase import Client
from flcore.trainmodel.resnet_cifar import resnet8_cifar


def get_class_num(class_list):
    """Extract class indices and counts from class distribution dict"""
    index = []
    compose = []
    for class_index, j in class_list.items():
        index.append(class_index)
        compose.append(j)
    return index, compose


class Local(object):
    """Local client for gradient computation and local training"""
    def __init__(self, data_client, class_list: dict, args):
        self.data_client = data_client
        self.device = args.device
        self.class_compose = class_list
        self.criterion = CrossEntropyLoss().to(args.device)
        self.local_model = resnet8_cifar(num_classes=args.num_classes, scaling=4).to(args.device)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training)

    def compute_gradient(self, global_params, args):
        list_class, per_class_compose = get_class_num(self.class_compose)
        indices_class = {class_index: [] for class_index in list_class}

        images_all = [unsqueeze(self.data_client[i][0], dim=0) for i in range(len(self.data_client))]
        labels_all = [int(self.data_client[i][1]) for i in range(len(self.data_client))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)

        def get_images(c, n):
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        self.local_model.load_state_dict(global_params)
        self.local_model.eval()
        self.local_model.classifier.train()
        net_parameters = list(self.local_model.classifier.parameters())
        criterion = CrossEntropyLoss().to(args.device)
        truth_gradient_all = {index: [] for index in list_class}
        truth_gradient_avg = {index: [] for index in list_class}

        for num_compute in range(10):
            for c, num in zip(list_class, per_class_compose):
                img_real = get_images(c, args.batch_real)
                lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                feature_real, output_real = self.local_model(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))
                truth_gradient_all[c].append(gw_real)
        
        for i in list_class:
            gw_real_temp = []
            gradient_all = truth_gradient_all[i]
            weight = 1.0 / len(gradient_all)
            for name_param in range(len(gradient_all[0])):
                list_values_param = []
                for client_one in gradient_all:
                    list_values_param.append(client_one[name_param] * weight)
                value_global_param = sum(list_values_param)
                gw_real_temp.append(value_global_param)
            truth_gradient_avg[i] = gw_real_temp
        return truth_gradient_avg

    def local_train(self, args, global_params):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        
        for _ in range(args.num_epochs_local_training):
            data_loader = DataLoader(dataset=self.data_client,
                                     batch_size=args.batch_size_local_training,
                                     shuffle=True)
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                images = transform_train(images)
                _, outputs = self.local_model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return self.local_model.state_dict()


class clientCREFF(Client):
    """CReFF client with support for (feature, output) tuple model"""
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.criterion = CrossEntropyLoss().to(self.device)
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])

    def train(self):
        """Local training with data augmentation"""
        trainloader = self.load_train_data()
        self.model.train()
        
        for _ in range(self.local_epochs):
            for images, labels in trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                images = self.transform_train(images)
                _, outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
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
                y_prob.append(output.detach().cpu().numpy())
                
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
