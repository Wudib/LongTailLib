import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file, imbalance_factor


random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
num_clients = 20
dir_path = "Cifar100/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition, longtail=False, longtail_type=None, imb_factor=None):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition, longtail, longtail_type):
        return
        
    # Get Cifar100 data - 保持原始numpy格式
    trainset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=True, download=True, transform=None)
    testset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=False, download=True, transform=None)
    
    # 直接使用原始numpy数据
    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data)  # 保持uint8格式
    dataset_image.extend(testset.data)
    dataset_label.extend(trainset.targets)
    dataset_label.extend(testset.targets)
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=10, longtail=longtail, 
                                    longtail_type=longtail_type, imbalance_factor=imb_factor, train_path=train_path)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, longtail, longtail_type)
    
    # ========== Generate Global Test Set ==========
    # Save complete original test set (10000 samples, balanced distribution)
    # This is used for Global Test Accuracy evaluation (aligned with source code)
    global_test_path = os.path.join(dir_path, "global_test.npz")
    global_test_data = {
        'x': testset.data,  # Original test images (10000, 32, 32, 3), uint8
        'y': np.array(testset.targets)  # Original test labels (10000,)
    }
    np.savez_compressed(global_test_path, data=global_test_data)
    print(f"[Info] Saved global test set: {global_test_path}")
    print(f"       - Samples: {len(testset.data)}")
    print(f"       - Classes: {num_classes} (balanced, ~{len(testset.data)//num_classes} per class)")


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    
    # 长尾分布参数
    longtail = False
    longtail_type = None
    imb_factor = None
    
    if len(sys.argv) > 4:
        longtail = True if sys.argv[4] == "longtail" else False
        
        if longtail and len(sys.argv) > 5:
            # 长尾类型映射
            longtail_type_map = {
                "global": "global_longtail",
                "local": "local_longtail",
                "mixed": "mixed_longtail",
                "-": "global_longtail"
            }
            user_type = sys.argv[5] if sys.argv[5] in ["global", "local", "mixed", "-"] else "global"
            longtail_type = longtail_type_map[user_type]
            
            if len(sys.argv) > 6:
                try:
                    imb_factor = float(sys.argv[6])
                    if imb_factor <= 0 or imb_factor > 1:
                        print("Warning: Imbalance factor should be in (0, 1], using default value.")
                        imb_factor = None
                except ValueError:
                    print("Warning: Invalid imbalance factor, using default value.")
    
    print(f"Generating CIFAR-100 dataset with settings:")
    print(f"  - Non-IID: {niid}")
    print(f"  - Balanced: {balance}")
    print(f"  - Partition: {partition}")
    print(f"  - Long-tail: {longtail}")
    print(f"  - Long-tail Type: {user_type if longtail and 'user_type' in locals() else '-'}")
    print(f"  - Imbalance Factor: {imb_factor if imb_factor is not None else imbalance_factor}")
    
    generate_dataset(dir_path, num_clients, niid, balance, partition, longtail, longtail_type, imb_factor)