import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file, imbalance_factor
import utils.dataset_utils as du


random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
num_clients = 40
dir_path = "Cifar10/"  # will be overridden dynamically in __main__


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
        
    # Get Cifar10 data
    # 重要：不使用transform，保持原始numpy格式，后续在训练时动态应用transform
    # 这样可以保证数据格式一致性，并在训练时添加Normalize
    
    # Robust download with fallback to local cache if SSL/network fails
    try:
        trainset = torchvision.datasets.CIFAR10(
            root=dir_path+"rawdata", train=True, download=True, transform=None)
        testset = torchvision.datasets.CIFAR10(
            root=dir_path+"rawdata", train=False, download=True, transform=None)
    except Exception as e:
        print("[Warn] Failed to download CIFAR-10 automatically due to network/SSL issue:", e)
        print("[Info] Attempting to use existing local CIFAR-10 data if available...")
        try:
            trainset = torchvision.datasets.CIFAR10(
                root=dir_path+"rawdata", train=True, download=False, transform=None)
            testset = torchvision.datasets.CIFAR10(
                root=dir_path+"rawdata", train=False, download=False, transform=None)
            print("[Info] Found existing local CIFAR-10 data and will proceed without downloading.")
        except Exception as e2:
            print("[Error] Local CIFAR-10 data not found.")
            print("Please manually download CIFAR-10 (Python version) from:")
            print("  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
            print(f"Then extract it so that the folder exists: {os.path.join(dir_path, 'rawdata', 'cifar-10-batches-py')}")
            raise

    # 直接使用原始numpy数据，不通过DataLoader转换
    # trainset.data: numpy array (50000, 32, 32, 3), dtype=uint8, range=[0, 255]
    # trainset.targets: list of integers
    dataset_image = []
    dataset_label = []

    # CIFAR10原始数据已经是numpy格式
    dataset_image.extend(trainset.data)  # 保持uint8格式，shape (N, 32, 32, 3)
    dataset_image.extend(testset.data)
    dataset_label.extend(trainset.targets)  # list of int
    dataset_label.extend(testset.targets)
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    # 设置长尾分布的不平衡因子
    if imb_factor is not None:
        imb_factor_value = imb_factor
    else:
        imb_factor_value = imbalance_factor
        
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=2,
                                    longtail=longtail, longtail_type=longtail_type, 
                                    imbalance_factor=imb_factor_value, train_path=train_path)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, longtail, longtail_type, imb_factor_value)
    
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
                    if imb_factor <= 0:
                        print("Warning: Imbalance factor should be > 0, using default value.")
                        imb_factor = None
                    elif imb_factor <= 1:
                        print("Note: legacy imbalance factor (<=1) provided; it will be auto-converted to IF in dataset_utils.")
                except ValueError:
                    print("Warning: Invalid imbalance factor, using default value.")
    
    # 目录命名中的α标签（仅用于命名展示，不影响实际划分alpha）
    alpha_tag = sys.argv[7] if len(sys.argv) > 7 else "0.5"
    # Optional: make alpha_tag also set the Dirichlet alpha used for partition
    if len(sys.argv) > 7:
        try:
            du.alpha = float(sys.argv[7])
        except ValueError:
            print("Warning: Invalid alpha value; using default:", du.alpha)

    # Optional CLI override: set num_clients (8th arg) BEFORE constructing dir path
    if len(sys.argv) > 8:
        try:
            num_clients_cli = int(sys.argv[8])
            if num_clients_cli > 0:
                num_clients = num_clients_cli
            else:
                print("Warning: num_clients should be > 0; using default:", num_clients)
        except ValueError:
            print("Warning: Invalid num_clients argument; using default:", num_clients)
    
    # 根据参数动态构造数据集目录名，并固定到dataset目录下
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if longtail:
        if imb_factor is not None and imb_factor >= 1:
            IF_tag = f"IF{int(imb_factor)}"
        else:
            IF_tag = "IFauto"
        type_tag = user_type if longtail and 'user_type' in locals() else 'global'
        dir_name = f"Cifar10-{IF_tag}-α{alpha_tag}-{type_tag}-NC{num_clients}"
    else:
        dir_name = f"Cifar10-NC{num_clients}"
    dir_path = os.path.join(script_dir, dir_name) + os.sep

    print(f"Generating CIFAR-10 dataset with settings:")
    print(f"  - Non-IID: {niid}")
    print(f"  - Balanced: {balance}")
    print(f"  - Partition: {partition}")
    print(f"  - Long-tail: {longtail}")
    print(f"  - Long-tail Type: {user_type if longtail and 'user_type' in locals() else '-'}")
    print(f"  - Imbalance Factor: {imb_factor if imb_factor is not None else imbalance_factor}")
    print(f"  - Output Dir: {dir_path}")
    
    # Optional CLI overrides: 7th sets alpha, 8th sets num_clients (applied earlier)
    # Usage example:
    #   python generate_Cifar10.py noniid - dir longtail global 50 0.5 40
    # 其中 '0.5' 用于命名与设置 alpha；'40' 覆盖客户端数，并体现到目录名中。

    print(f"[debug] alpha={du.alpha}, num_clients={num_clients}")

generate_dataset(dir_path, num_clients, niid, balance, partition, longtail, longtail_type, imb_factor)