import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt


batch_size = 10
train_ratio = 0.75 # merge original training set and test set, then split it manually. 
alpha = 0.1 # for Dirichlet distribution. 100 for exdir
imbalance_factor = 0.1 # long-tail IF (>=1 recommended). If <1, treated as legacy param and auto-converted; larger value => more imbalanced

def check(config_path, train_path, test_path, num_clients, niid=False, 
        balance=True, partition=None, longtail=False, longtail_type=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['non_iid'] == niid and \
            config['balance'] == balance and \
            config['partition'] == partition and \
            config['alpha'] == alpha and \
            config['batch_size'] == batch_size and \
            config.get('longtail', False) == longtail and \
            config.get('longtail_type', None) == longtail_type and \
            config.get('imbalance_factor', imbalance_factor) == imbalance_factor:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

# ===== Long-tail helpers (IF >= 1, legacy <1 auto-converted) =====

def create_longtail_distribution(dataset, num_classes, imbalance_factor=0.1, distribution='exponential'):
    """创建长尾分布的全局类别样本配额
    Args:
        dataset: (dataset_content, dataset_label)
        num_classes: 类别数
        imbalance_factor: IF = n_max / n_min (>=1)。若 <1，按旧语义（λ样式）转换为 IF。
        distribution: 'exponential' 或 'power_law'
    Returns:
        dict: {class_idx: np.ndarray(sample_indices)}
    """
    dataset_content, dataset_label = dataset
    indices_per_class = {}
    for i in range(num_classes):
        indices = np.where(dataset_label == i)[0]
        np.random.shuffle(indices)
        indices_per_class[i] = indices

    max_samples_available = max((len(indices_per_class[i]) for i in range(num_classes)), default=0)

    # 规范化为 IF
    if imbalance_factor is None:
        IF = 1.0
    elif imbalance_factor >= 1:
        IF = float(imbalance_factor)
    else:
        C = max(2, num_classes)
        if distribution == 'exponential':
            IF = float(np.exp((C - 1) / (imbalance_factor * C)))
        elif distribution == 'power_law':
            IF = float(C ** (1.0 / (imbalance_factor * C)))
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    class_samples = {}
    if num_classes <= 1:
        if num_classes == 1:
            class_samples[0] = indices_per_class[0]
        return class_samples

    if distribution == 'exponential':
        # n_i = n_max * exp(-k * i), k = ln(IF)/(C-1)
        k = np.log(IF) / (num_classes - 1)
        n_max = max_samples_available
        for i in range(num_classes):
            desired = int(max(1, round(n_max * np.exp(-k * i))))
            desired = min(desired, len(indices_per_class[i]))
            class_samples[i] = indices_per_class[i][:desired]
    elif distribution == 'power_law':
        # n_i = n_max * (i+1)^(-a), a = ln(IF)/ln(C)
        a = np.log(IF) / np.log(num_classes)
        n_max = max_samples_available
        for i in range(num_classes):
            desired = int(max(1, round(n_max * ((i + 1) ** (-a)))))
            desired = min(desired, len(indices_per_class[i]))
            class_samples[i] = indices_per_class[i][:desired]
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    return class_samples


def imbalanced_split(dataset, num_clients, num_classes, longtail_type='global_longtail', 
                    imbalance_factor=0.1, distribution='exponential', alpha=0.1):
    """按不同长尾类型将数据索引分配至客户端
    longtail_type: 'global_longtail' | 'local_longtail' | 'mixed_longtail'
    """
    dataset_content, dataset_label = dataset

    if longtail_type == 'global_longtail':
        # 先在全局构造长尾配额, 再用 Dirichlet 分给各客户端
        class_samples = create_longtail_distribution(dataset, num_classes, imbalance_factor, distribution)
        dataidx_map = {i: [] for i in range(num_clients)}
        for class_idx, indices in class_samples.items():
            indices = np.array(indices)
            if len(indices) == 0:
                continue
            np.random.shuffle(indices)
            dir_alpha = alpha if alpha is not None else 0.1
            proportions = np.random.dirichlet(np.repeat(dir_alpha, num_clients))
            split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            splits = np.split(indices, split_points)
            for client_idx in range(num_clients):
                if len(splits[client_idx]) > 0:
                    dataidx_map[client_idx].extend(splits[client_idx].tolist())
        return dataidx_map

    elif longtail_type == 'local_longtail':
        # 先近似均衡分配，再在每个客户端内部构造长尾
        dataidx_map = {i: [] for i in range(num_clients)}
        K = num_classes
        for k in range(K):
            idx_k = np.where(dataset_label == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(max(alpha, 1e-6) * 10, num_clients))
            proportions = proportions / proportions.sum()
            cuts = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            splits = np.split(idx_k, cuts)
            for j in range(num_clients):
                dataidx_map[j].extend(splits[j].tolist())
        # 在每个客户端内部进行长尾裁剪
        for client_idx in range(num_clients):
            client_data = np.array(dataidx_map[client_idx])
            client_labels = dataset_label[client_data]
            uniq, counts = np.unique(client_labels, return_counts=True)
            if len(uniq) <= 1:
                continue
            order = np.random.permutation(uniq)
            total = len(client_data)
            # 以 IF 生成指数权重, 归一后分配目标样本数
            Cc = len(order)
            if imbalance_factor is None:
                eff_if = 1.0
            elif imbalance_factor >= 1:
                eff_if = float(imbalance_factor)
            else:
                C = max(2, num_classes)
                if distribution == 'exponential':
                    eff_if = float(np.exp((C - 1) / (imbalance_factor * C)))
                else:
                    eff_if = float(C ** (1.0 / (imbalance_factor * C)))
            k = np.log(max(eff_if, 1.0)) / max(Cc - 1, 1)
            weights = np.array([np.exp(-k * i) for i in range(Cc)], dtype=float)
            weights = weights / (weights.sum() + 1e-12)
            target_per_cls = {order[i]: max(1, int(round(total * weights[i]))) for i in range(Cc)}
            # 选择样本（不超过现有数量）
            new_client = []
            for cls in uniq:
                cls_mask = (client_labels == cls)
                cls_indices = client_data[cls_mask]
                t = min(len(cls_indices), target_per_cls.get(cls, len(cls_indices)))
                if t > 0:
                    pick = np.random.choice(cls_indices, size=t, replace=False)
                    new_client.extend(pick.tolist())
            dataidx_map[client_idx] = new_client
        return dataidx_map

    elif longtail_type == 'mixed_longtail':
        # 一部分客户端(3/4)构造局部长尾，剩余保持均衡
        num_longtail = max(1, num_clients * 3 // 4)
        longtail_clients = set(np.random.choice(range(num_clients), size=num_longtail, replace=False))
        balanced_clients = [i for i in range(num_clients) if i not in longtail_clients]

        # 为每个类别准备样本池
        pools = {c: np.where(dataset_label == c)[0].tolist() for c in range(num_classes)}
        for c in pools:
            np.random.shuffle(pools[c])

        total = len(dataset_label)
        base_per_client = total // num_clients

        dataidx_map = {i: [] for i in range(num_clients)}
        # 长尾客户端
        for cid in longtail_clients:
            order = np.random.permutation(num_classes)
            # 目标按指数权重分配
            if imbalance_factor is None:
                eff_if = 1.0
            elif imbalance_factor >= 1:
                eff_if = float(imbalance_factor)
            else:
                C = max(2, num_classes)
                if distribution == 'exponential':
                    eff_if = float(np.exp((C - 1) / (imbalance_factor * C)))
                else:
                    eff_if = float(C ** (1.0 / (imbalance_factor * C)))
            k = np.log(max(eff_if, 1.0)) / max(num_classes - 1, 1)
            weights = np.array([np.exp(-k * i) for i in range(num_classes)], dtype=float)
            weights = weights / (weights.sum() + 1e-12)
            t_counts = [max(1, int(round(base_per_client * w))) for w in weights]
            for i, cls in enumerate(order):
                need = t_counts[i]
                avail = len(pools[cls])
                take = min(need, avail)
                if take > 0:
                    dataidx_map[cid].extend(pools[cls][:take])
                    pools[cls] = pools[cls][take:]
        # 均衡客户端
        for cid in balanced_clients:
            per_cls = base_per_client // max(num_classes, 1)
            for cls in range(num_classes):
                take = min(per_cls, len(pools[cls]))
                if take > 0:
                    dataidx_map[cid].extend(pools[cls][:take])
                    pools[cls] = pools[cls][take:]
        return dataidx_map

    else:
        raise ValueError(f"不支持的长尾分布类型: {longtail_type}")


def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None, longtail=False, longtail_type=None, imbalance_factor=None, distribution='exponential', train_path=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    # guarantee that each client must have at least one batch of data for testing. 
    least_samples = int(min(batch_size / (1-train_ratio), len(dataset_label) / num_clients / 2))

    dataidx_map = {}

    if longtail:
        # 使用长尾分布划分数据
        if imbalance_factor is None:
            imbalance_factor = globals()['imbalance_factor']
        
        dataidx_map = imbalanced_split(
            data, num_clients, num_classes, 
            longtail_type=longtail_type, 
            imbalance_factor=imbalance_factor,
            distribution=distribution,
            alpha=alpha
        )
        
        # 最小样本保护：若任一客户端样本量小于 least_samples，则重试重新划分（最多50次）
        max_retries = 50
        retry_cnt = 0
        while True:
            sizes = [len(v) for v in dataidx_map.values()] if isinstance(dataidx_map, dict) else []
            min_size_client = min(sizes) if sizes else 0
            if min_size_client >= least_samples:
                break
            if retry_cnt >= max_retries:
                print(f"[Warn] Long-tail split does not meet least_samples={least_samples} after {max_retries} retries (min={min_size_client}). Proceed anyway.")
                break
            retry_cnt += 1
            dataidx_map = imbalanced_split(
                data, num_clients, num_classes,
                longtail_type=longtail_type,
                imbalance_factor=imbalance_factor,
                distribution=distribution,
                alpha=alpha
            )
        
        # 获取数据集保存目录（从train_path提取）
        dataset_dir = os.path.dirname(train_path)
        if not dataset_dir:
            dataset_dir = '.'
    elif not niid:
        partition = 'pat'
        class_per_client = num_classes

    if not longtail and partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])
    
        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            if len(selected_clients) == 0:
                break
            selected_clients = selected_clients[:int(np.ceil((num_clients/num_classes)*class_per_client))]
    
            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))
    
            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1
    
    elif not longtail and partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)
    
        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')
    
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1
    
        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    
    elif not longtail and partition == 'exdir':
        r'''This strategy comes from https://arxiv.org/abs/2311.03154
        See details in https://github.com/TsingZ0/PFLlib/issues/139
    
        This version in PFLlib is slightly different from the original version 
        Some changes are as follows:
        n_nets -> num_clients, n_class -> num_classes
        '''
        C = class_per_client
        
        '''The first level: allocate labels to clients
        clientidx_map (dict, {label: clientidx}), e.g., C=2, num_clients=5, num_classes=10
            {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [4, 5], 5: [5, 6], 6: [6, 7], 7: [7, 8], 8: [8, 9], 9: [9, 0]}
        '''
        min_size_per_label = 0
        # You can adjust the `min_require_size_per_label` to meet you requirements
        min_require_size_per_label = max(C * num_clients // num_classes // 2, 1)
        if min_require_size_per_label < 1:
            raise ValueError
        clientidx_map = {}
        while min_size_per_label < min_require_size_per_label:
            # initialize
            for k in range(num_classes):
                clientidx_map[k] = []
            # allocate
            for i in range(num_clients):
                labelidx = np.random.choice(range(num_classes), C, replace=False)
                for k in labelidx:
                    clientidx_map[k].append(i)
            min_size_per_label = min([len(clientidx_map[k]) for k in range(num_classes)])
        
        '''The second level: allocate data idx'''
        dataidx_map = {}
        y_train = dataset_label
        min_size = 0
        min_require_size = 10
        K = num_classes
        N = len(y_train)
        print("\n*****clientidx_map*****")
        print(clientidx_map)
        print("\n*****Number of clients per label*****")
        print([len(clientidx_map[i]) for i in range(len(clientidx_map))])
    
        # ensure per client' sampling size >= min_require_size (is set to 10 originally in [3])
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # Balance
                # Case 1 (original case in Dir): Balance the number of sample per client
                proportions = np.array([p * (len(idx_j) < N / num_clients and j in clientidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))])
                # Case 2: Don't balance
                #proportions = np.array([p * (j in label_netidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # process the remainder samples
                '''Note: Process the remainder data samples (yipeng, 2023-11-14).
                There are some cases that the samples of class k are not allocated completely, i.e., proportions[-1] < len(idx_k)
                In these cases, the remainder data samples are assigned to the last client in `clientidx_map[k]`.
                '''
                if proportions[-1] != len(idx_k):
                    for w in range(clientidx_map[k][-1], num_clients-1):
                        proportions[w] = len(idx_k)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))] 
                min_size = min([len(idx_j) for idx_j in idx_batch])
    
        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            dataidx_map[j] = idx_batch[j]
    
    elif not longtail:
        raise NotImplementedError

    # Shuffle each client's data indices to avoid class-ordered data
    # This is important for training stability, especially when batch size is small
    for client in range(num_clients):
        if client in dataidx_map and len(dataidx_map[client]) > 0:
            idxs = np.array(dataidx_map[client])
            np.random.shuffle(idxs)  # Shuffle to randomize class order
            dataidx_map[client] = idxs.tolist()

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
    
    # 统一在任意划分设置下生成可视化柱状图（Dirichlet、pat、exdir、longtail）
    # 保留长尾分布时的详细打印信息
    if longtail:
        print(f"\n*****Long-tail Distribution Information*****")
        print(f"Distribution Type: {longtail_type}")
        print(f"Distribution Function: {distribution}")
        print(f"Imbalance Factor (input): {imbalance_factor}")
        
        # 计算并打印最终生效的 IF（兼容旧参数转换）
        def _effective_IF(val, dist, C):
            if val is None:
                return 1.0
            if val >= 1:
                return float(val)
            Cx = max(2, C)
            if dist == 'exponential':
                return float(np.exp((Cx - 1) / (val * Cx)))
            elif dist == 'power_law':
                return float((Cx) ** (1.0 / (val * Cx)))
            else:
                return 1.0
        _eff_if = _effective_IF(imbalance_factor, distribution, num_classes)
        if imbalance_factor is None:
            print(f"Effective IF (used): {_eff_if:.4f} (default)")
        elif imbalance_factor >= 1:
            print(f"Effective IF (used): {_eff_if:.4f} (input IF)")
        else:
            print(f"Effective IF (used): {_eff_if:.4f} (converted from legacy param {imbalance_factor})")
    
    # 计算全局类别分布
    global_class_counts = np.zeros(num_classes)
    for client_y in y:
        for i in range(num_classes):
            global_class_counts[i] += np.sum(np.array(client_y) == i)

    if longtail:
        print("\n*****Global Class Distribution*****")
        for i in range(num_classes):
            print(f"Class {i}: {int(global_class_counts[i])} samples")
        # 打印实际观察到的全局 IF（max/min）
        if num_classes > 0 and np.any(global_class_counts > 0):
            observed_max = int(np.max(global_class_counts))
            observed_min = int(np.min(global_class_counts[global_class_counts > 0]))
            observed_if = observed_max / observed_min if observed_min > 0 else float('inf')
            print(f"Observed Global IF (max/min): {observed_if:.4f} (max={observed_max}, min={observed_min})")
            if longtail_type in ('local_longtail', 'mixed_longtail'):
                print("Note: For local/mixed long-tail, global IF may not equal the target IF; IF mainly shapes per-client distributions.")

    # 汇总所有客户端标签用于全局可视化
    all_client_labels = []
    for client_y in y:
        all_client_labels.extend(client_y)

    # 确定保存目录（与原逻辑一致：使用 train_path 的目录）
    if train_path is not None:
        dataset_dir = os.path.dirname(train_path)
        if not dataset_dir:
            dataset_dir = '.'
    else:
        dataset_dir = '.'

    # 确定文件后缀与标题（以便区分不同划分设置）
    if longtail:
        suffix = str(longtail_type) if longtail_type else 'longtail'
        global_title = f"Global Class Distribution ({suffix})"
    else:
        if partition in ['dir', 'exdir', 'pat']:
            suffix = partition
        else:
            suffix = 'noniid' if niid else 'iid'
        global_title = f"Global Class Distribution ({suffix})"

    # 保存全局类别分布柱状图
    visualize_class_distribution(
        all_client_labels, num_classes,
        title=global_title,
        save_path=os.path.join(dataset_dir, f"global_distribution_{suffix}.png")
    )

    # 保存前10个客户端的类别分布柱状图
    for client in range(min(10, num_clients)):
        visualize_class_distribution(
            y[client], num_classes,
            title=f"Client {client} Class Distribution ({suffix})",
            save_path=os.path.join(dataset_dir, f"client_{client}_distribution_{suffix}.png")
        )

    # 打印每个客户端的样本统计
    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic

def visualize_class_distribution(y, num_classes, title="Class Distribution", save_path=None):
    """可视化类别分布
    Args:
        y: 标签列表或数组
        num_classes: 类别数量
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    # 计算每个类别的样本数量
    class_counts = np.zeros(num_classes)
    for i in range(num_classes):
        class_counts[i] = np.sum(np.array(y) == i)
    
    # 创建柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_classes), class_counts)
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.xticks(range(num_classes))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存或显示图表
    if save_path:
        # Ensure parent directory exists before saving the figure
        try:
            parent = os.path.dirname(save_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        except Exception:
            pass
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
                num_classes, statistic, niid=False, balance=True, partition=None, 
                longtail=False, longtail_type=None, imbalance_factor=None, seed=None):
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size,
        'longtail': longtail,
        'longtail_type': longtail_type,
        'imbalance_factor': imbalance_factor if imbalance_factor else imbalance_factor,
        'seed': seed,
    }

    # gc.collect()
    print("Saving to disk.\n")

    # Ensure output directories exist before writing files
    try:
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
    except Exception:
        pass

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")


class ImageDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing file names
            image_folder (str): Path to the folder containing the images
            transform (callable, optional): Optional transform to be applied to the image
        """
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the file name from the DataFrame
        img_name = self.dataframe.iloc[idx]['file_name']
        img_label = self.dataframe.iloc[idx]['class']
        img_path = os.path.join(self.image_folder, img_name)
        
        # Load the image using PIL
        image = Image.open(img_path).convert('RGB')  # Ensure RGB if not grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_label


def split_data(X, y):
    """
    Split per-client data into train/test sets using the global train_ratio.
    Returns two lists of dicts: train_data and test_data, each element is {'x': array, 'y': array}.
    This mirrors the behavior used by various generate_*.py scripts.
    """
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_ratio, shuffle=True
        )

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print(f"Total number of samples: {sum(num_samples['train'] + num_samples['test'])}")
    print(f"The number of train samples: {num_samples['train']}")
    print(f"The number of test samples: {num_samples['test']}")
    print()

    return train_data, test_data