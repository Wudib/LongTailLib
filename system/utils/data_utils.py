import numpy as np
import os
import torch
import torchvision.transforms as transforms
from collections import defaultdict


def read_data(dataset, idx, is_train=True):
    """
    从.npz文件读取客户端数据
    
    Args:
        dataset: 数据集名称
        idx: 客户端ID
        is_train: 是否为训练集
    
    Returns:
        包含'x'和'y'的字典
    """
    if is_train:
        data_dir = os.path.join('../dataset', dataset, 'train/')
    else:
        data_dir = os.path.join('../dataset', dataset, 'test/')

    file = data_dir + str(idx) + '.npz'
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data


def read_client_data(dataset, idx, is_train=True, few_shot=0, transform=None):
    """
    读取客户端数据，支持动态数据增强
    
    Args:
        dataset: 数据集名称
        idx: 客户端ID
        is_train: 是否为训练集
        few_shot: few-shot学习的样本数限制
        transform: 数据增强transform（如果为None，自动根据数据集和is_train选择）
    """
    data = read_data(dataset, idx, is_train)
    
    # 如果没有指定transform，自动获取默认transform
    if transform is None and "News" not in dataset and "Shakespeare" not in dataset:
        transform = get_transforms(dataset, is_train)
    
    if "News" in dataset:
        data_list = process_text(data)
    elif "Shakespeare" in dataset:
        data_list = process_Shakespeare(data)
    else:
        data_list = process_image(data, transform)

    if is_train and few_shot > 0:
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
    return data_list


def process_image(data, transform=None):
    """
    处理图像数据，支持动态transform
    
    Args:
        data: 包含'x'和'y'的字典，其中'x'是numpy数组（uint8, shape: N x H x W x C）
        transform: torchvision transform（可选）
    """
    # 保持numpy格式，不预先转换为tensor（transform中会处理）
    X = data['x']  # numpy array, uint8, (N, H, W, C)
    y = data['y']  # numpy array or list, int
    
    if transform is not None:
        # 创建带transform的dataset
        class TransformedDataset:
            def __init__(self, X, y, transform):
                self.X = X  # numpy array
                self.y = y
                self.transform = transform
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                # X[idx]是uint8 numpy array
                x = self.X[idx]  
                y = self.y[idx]
                
                # Ensure x is numpy array with correct shape (H, W, C) and uint8 dtype
                if isinstance(x, np.ndarray):
                    # Check and fix shape
                    if x.ndim == 3:
                        if x.shape[0] == 3 or x.shape[0] == 1:  # (C, H, W) format
                            x = x.transpose(1, 2, 0)  # Convert to (H, W, C)
                        # else: already (H, W, C)
                    
                    # Ensure uint8 dtype for ToPILImage
                    if x.dtype != np.uint8:
                        if x.dtype in [np.float32, np.float64]:
                            # Assuming data is in [0, 1] range, scale to [0, 255]
                            x = (x * 255).astype(np.uint8)
                        else:
                            x = x.astype(np.uint8)
                    # x is now (H, W, C) uint8
                
                # transform会处理numpy -> PIL -> augment -> Tensor -> Normalize
                if self.transform:
                    x = self.transform(x)
                else:
                    # 如果没有transform，手动转tensor
                    x = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
                
                return x, torch.tensor(y, dtype=torch.long)
        
        return TransformedDataset(X, y, transform)
    else:
        # 无transform时，手动转换为tensor并归一化
        X_tensor = torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255.0  # (N, C, H, W), [0, 1]
        y_tensor = torch.tensor(y, dtype=torch.long)
        return [(x, y) for x, y in zip(X_tensor, y_tensor)]


def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


def process_Shakespeare(data):
    X = torch.Tensor(data['x']).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


def get_transforms(dataset, is_train=True):
    """
    获取数据集对应的标准transform
    
    Args:
        dataset: 数据集名称
        is_train: 是否为训练集（训练集应用数据增强，测试集只normalize）
    
    Returns:
        torchvision.transforms.Compose对象
    """
    dataset_lower = dataset.lower()
    
    # CIFAR-10和CIFAR-100
    if 'cifar10' in dataset_lower or 'cifar100' in dataset_lower:
        # CIFAR标准normalize参数（与CReFF源码一致）
        # 重要：这些参数来自CIFAR10数据集的统计值，必须使用这些值才能与预训练模型兼容
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]  # 修正为CReFF使用的标准参数
        )
        
        if is_train:
            # 训练集：ToTensor + 数据增强 + normalize
            return transforms.Compose([
                transforms.ToPILImage(),  # numpy array -> PIL Image
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # PIL -> Tensor, [0, 255] uint8 -> [0, 1] float32
                normalize  # 标准化到均值0方差1
            ])
        else:
            # 测试集：ToTensor + normalize
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                normalize
            ])
    
    # MNIST和FashionMNIST
    elif 'mnist' in dataset_lower or 'fashion' in dataset_lower:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        
        if is_train:
            return transforms.Compose([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                normalize
            ])
        else:
            return transforms.Compose([normalize])
    
    # 其他数据集：默认只normalize
    else:
        # 假设是3通道图像
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet标准
            std=[0.229, 0.224, 0.225]
        )
        
        if is_train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                normalize
            ])
        else:
            return transforms.Compose([normalize])