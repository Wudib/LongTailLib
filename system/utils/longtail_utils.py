import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def focal_loss(logits, labels, alpha=None, gamma=2.0):
    """
    实现Focal Loss，关注难分类样本
    Args:
        logits: 模型输出的logits，形状为[batch_size, num_classes]
        labels: 真实标签，形状为[batch_size]
        alpha: 类别权重，可以是标量或者形状为[num_classes]的张量
        gamma: 聚焦参数，越大越关注难分类样本
    Returns:
        focal_loss: 计算得到的focal loss
    """
    num_classes = logits.size(1)
    
    # 计算交叉熵损失
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    
    # 计算预测概率
    pt = torch.exp(-ce_loss)
    
    # 如果alpha不为None，应用类别权重
    if alpha is not None:
        if isinstance(alpha, torch.Tensor):
            # 如果alpha是张量，确保形状正确
            assert alpha.size(0) == num_classes, "alpha的大小应该等于类别数量"
            alpha_t = alpha.gather(0, labels)
        else:
            # 如果alpha是标量，对所有类别使用相同的权重
            alpha_t = alpha
        
        # 应用alpha权重
        focal_weight = alpha_t * (1 - pt) ** gamma
    else:
        # 不使用alpha权重
        focal_weight = (1 - pt) ** gamma
    
    # 计算最终的focal loss
    focal_loss = focal_weight * ce_loss
    
    # 返回平均损失
    return focal_loss.mean()


def class_balanced_loss(logits, labels, samples_per_class, beta=0.9999):
    """
    实现Class-Balanced Loss，根据有效样本数调整损失权重
    Args:
        logits: 模型输出的logits，形状为[batch_size, num_classes]
        labels: 真实标签，形状为[batch_size]
        samples_per_class: 每个类别的样本数量，形状为[num_classes]
        beta: 超参数，控制有效样本数的计算
    Returns:
        cb_loss: 计算得到的class-balanced loss
    """
    num_classes = logits.size(1)
    
    # 确保samples_per_class是张量
    if not isinstance(samples_per_class, torch.Tensor):
        samples_per_class = torch.tensor(samples_per_class, dtype=torch.float32, device=logits.device)
    
    # 计算每个类别的权重：(1-beta)/(1-beta^n)
    effective_num = 1.0 - torch.pow(beta, samples_per_class)
    weights = (1.0 - beta) / effective_num
    
    # 归一化权重
    weights = weights / torch.sum(weights) * num_classes
    
    # 获取每个样本对应类别的权重
    weights_per_sample = weights[labels]
    
    # 计算交叉熵损失
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    
    # 应用权重
    cb_loss = weights_per_sample * ce_loss
    
    # 返回平均损失
    return cb_loss.mean()


def logit_adjustment_loss(logits, labels, tau=1.0, samples_per_class=None):
    """
    实现Logit Adjustment Loss，在推理阶段调整logits
    Args:
        logits: 模型输出的logits，形状为[batch_size, num_classes]
        labels: 真实标签，形状为[batch_size]
        tau: 温度参数
        samples_per_class: 每个类别的样本数量，形状为[num_classes]
    Returns:
        la_loss: 计算得到的logit adjustment loss
    """
    num_classes = logits.size(1)
    
    # 确保samples_per_class是张量
    if samples_per_class is None:
        # 如果没有提供样本数量，假设均匀分布
        prior_logits = torch.zeros(num_classes, device=logits.device)
    else:
        if not isinstance(samples_per_class, torch.Tensor):
            samples_per_class = torch.tensor(samples_per_class, dtype=torch.float32, device=logits.device)
        
        # 计算先验概率的对数
        prior = samples_per_class / torch.sum(samples_per_class)
        prior_logits = torch.log(prior + 1e-8)  # 添加小值避免log(0)
    
    # 调整logits
    adjusted_logits = logits + tau * prior_logits
    
    # 计算交叉熵损失
    la_loss = F.cross_entropy(adjusted_logits, labels)
    
    return la_loss


def balanced_softmax_loss(logits, labels, samples_per_class):
    """
    实现Balanced Softmax Loss，通过样本数量调整logits
    Args:
        logits: 模型输出的logits，形状为[batch_size, num_classes]
        labels: 真实标签，形状为[batch_size]
        samples_per_class: 每个类别的样本数量，形状为[num_classes]
    Returns:
        bs_loss: 计算得到的balanced softmax loss
    """
    # 确保samples_per_class是张量
    if not isinstance(samples_per_class, torch.Tensor):
        samples_per_class = torch.tensor(samples_per_class, dtype=torch.float32, device=logits.device)
    
    # 计算调整因子
    adjustment = torch.log(samples_per_class + 1e-8)  # 添加小值避免log(0)
    
    # 调整logits
    logits = logits + adjustment
    
    # 计算交叉熵损失
    bs_loss = F.cross_entropy(logits, labels)
    
    return bs_loss


def evaluate_longtail(model, dataloader, device, num_classes, head_classes=None, middle_classes=None, tail_classes=None):
    """
    评估模型在长尾分布下的性能
    Args:
        model: 待评估模型
        dataloader: 测试数据加载器
        device: 计算设备
        num_classes: 类别数量
        head_classes: 头部类别列表，如果为None则自动划分
        middle_classes: 中部类别列表，如果为None则自动划分
        tail_classes: 尾部类别列表，如果为None则自动划分
    Returns:
        metrics: 包含各组类别性能指标的字典
    """
    model.eval()
    
    # 收集每个类别的预测结果
    all_preds = [[] for _ in range(num_classes)]
    all_labels = [[] for _ in range(num_classes)]
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 获取预测类别
            _, preds = torch.max(output, 1)
            
            # 按类别收集预测结果
            for i in range(len(target)):
                label = target[i].item()
                pred = preds[i].item()
                all_labels[label].append(label)
                all_preds[label].append(pred)
    
    # 如果没有提供类别划分，自动划分
    if head_classes is None or middle_classes is None or tail_classes is None:
        # 计算每个类别的样本数量
        class_samples = [len(labels) for labels in all_labels]
        
        # 按样本数量排序
        sorted_classes = np.argsort(class_samples)[::-1]  # 降序排列
        
        # 划分头部、中部、尾部类别
        head_size = num_classes // 3
        middle_size = num_classes // 3
        tail_size = num_classes - head_size - middle_size
        
        head_classes = sorted_classes[:head_size].tolist()
        middle_classes = sorted_classes[head_size:head_size+middle_size].tolist()
        tail_classes = sorted_classes[head_size+middle_size:].tolist()
    
    # 计算各组类别的准确率
    head_acc = calculate_group_accuracy(all_preds, all_labels, head_classes)
    middle_acc = calculate_group_accuracy(all_preds, all_labels, middle_classes)
    tail_acc = calculate_group_accuracy(all_preds, all_labels, tail_classes)
    
    # 计算总体准确率
    all_preds_flat = [pred for class_preds in all_preds for pred in class_preds]
    all_labels_flat = [label for class_labels in all_labels for label in class_labels]
    overall_acc = sum([1 for p, l in zip(all_preds_flat, all_labels_flat) if p == l]) / len(all_preds_flat) if all_preds_flat else 0
    
    # 计算平衡准确率（每个类别的准确率平均）
    class_acc = []
    for i in range(num_classes):
        if all_labels[i]:
            acc = sum([1 for p, l in zip(all_preds[i], all_labels[i]) if p == l]) / len(all_labels[i])
            class_acc.append(acc)
    balanced_acc = sum(class_acc) / len(class_acc) if class_acc else 0
    
    # 返回评估指标
    metrics = {
        'overall_acc': overall_acc,
        'balanced_acc': balanced_acc,
        'head_acc': head_acc,
        'middle_acc': middle_acc,
        'tail_acc': tail_acc,
        'head_classes': head_classes,
        'middle_classes': middle_classes,
        'tail_classes': tail_classes
    }
    
    return metrics


def calculate_group_accuracy(all_preds, all_labels, group_classes):
    """
    计算一组类别的准确率
    Args:
        all_preds: 按类别收集的预测结果
        all_labels: 按类别收集的真实标签
        group_classes: 类别组
    Returns:
        group_acc: 该组类别的准确率
    """
    group_correct = 0
    group_total = 0
    
    for cls in group_classes:
        correct = sum([1 for p, l in zip(all_preds[cls], all_labels[cls]) if p == l])
        total = len(all_labels[cls])
        
        group_correct += correct
        group_total += total
    
    return group_correct / group_total if group_total > 0 else 0