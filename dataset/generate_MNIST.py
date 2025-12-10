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
num_clients = 20
# 将输出目录统一到项目下的 dataset/MNIST/，便于后续分析脚本读取
dir_path = os.path.join("MNIST1") + os.sep


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition, longtail=False, longtail_type=None, imb_factor=None, alpha_override=None, seed=None):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition, longtail, longtail_type):
        return

    # # FIX HTTP Error 403: Forbidden
    # from six.moves import urllib
    # opener = urllib.request.build_opener()
    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    # urllib.request.install_opener(opener)

    # Get MNIST data - 保持原始numpy格式
    trainset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=True, download=True, transform=None)
    testset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=False, download=True, transform=None)
    
    # 直接使用原始numpy数据
    dataset_image = []
    dataset_label = []

    # MNIST data is (N, 28, 28), need to add channel dimension -> (N, 28, 28, 1)
    dataset_image.extend(trainset.data.numpy()[:, :, :, np.newaxis])
    dataset_image.extend(testset.data.numpy()[:, :, :, np.newaxis])
    dataset_label.extend(trainset.targets.numpy())
    dataset_label.extend(testset.targets.numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # 设置长尾分布的不平衡因子
    if imb_factor is not None:
        imb_factor_value = imb_factor
    else:
        imb_factor_value = imbalance_factor

    # 覆盖 Dirichlet alpha（控制客户端间类别分配的均衡程度）
    if alpha_override is not None:
        try:
            du.alpha = float(alpha_override)
        except Exception:
            print("Warning: invalid alpha_override, keep default du.alpha")
        
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=2,
                                    longtail=longtail, longtail_type=longtail_type, 
                                    imbalance_factor=imb_factor_value, train_path=train_path)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, longtail, longtail_type, imb_factor_value, seed=seed)
    
    # ========== Generate Global Test Set ==========
    # Save complete original test set (10000 samples, balanced distribution)
    # This is used for Global Test Accuracy evaluation (aligned with source code)
    global_test_path = os.path.join(dir_path, "global_test.npz")
    # MNIST: (N, 28, 28) -> (N, 28, 28, 1) for consistency
    global_test_images = testset.data.numpy()[:, :, :, np.newaxis]
    global_test_data = {
        'x': global_test_images,  # Original test images (10000, 28, 28, 1), uint8
        'y': testset.targets.numpy()  # Original test labels (10000,)
    }
    np.savez_compressed(global_test_path, data=global_test_data)
    print(f"[Info] Saved global test set: {global_test_path}")
    print(f"       - Samples: {len(testset.data)}")
    print(f"       - Classes: {num_classes} (balanced, ~{len(testset.data)//num_classes} per class)")


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    # 处理长尾分布参数
    longtail = False
    longtail_type = None
    imb_factor = None
    alpha_override = None
    seed = None
    
    # 解析 longtail 及后续的可选参数（alpha/seed/num_clients）
    pos = 4
    if len(sys.argv) > 4:
        if sys.argv[4] == "longtail":
            longtail = True
            # 映射长尾类型到dataset_utils.py中支持的格式
            if len(sys.argv) > 5:
                longtail_type_map = {
                    "global": "global_longtail",
                    "local": "local_longtail",
                    "mixed": "mixed_longtail",
                    "-": "global_longtail"
                }
                user_type = sys.argv[5] if sys.argv[5] in ["global", "local", "mixed", "-"] else "global"
                longtail_type = longtail_type_map[user_type]
            pos = 6
            # 可选：不平衡因子
            if len(sys.argv) > pos:
                try:
                    imb_factor = float(sys.argv[pos])
                    if imb_factor <= 0 or imb_factor > 1:
                        print("Warning: Imbalance factor should be in (0, 1], using default value.")
                        imb_factor = None
                    else:
                        pos += 1
                except ValueError:
                    print("Warning: Invalid imbalance factor, using default value.")
        elif sys.argv[4] == "-":
            # 兼容旧用法的占位符
            pos = 5
    # 无论是否长尾，后续均可按顺序传入 alpha、seed、num_clients
    if len(sys.argv) > pos:
        try:
            alpha_override = float(sys.argv[pos]); pos += 1
        except ValueError:
            print("Warning: Invalid alpha, fall back to default in dataset_utils.py")
    if len(sys.argv) > pos:
        try:
            seed = int(sys.argv[pos]); pos += 1
        except ValueError:
            print("Warning: Invalid seed, ignore.")
    if len(sys.argv) > pos:
        try:
            num_clients = int(sys.argv[pos]); pos += 1
        except ValueError:
            print("Warning: Invalid num_clients, use default 20.")

    print(f"Generating MNIST dataset with settings:")
    print(f"  - Non-IID: {niid}")
    print(f"  - Balanced: {balance}")
    print(f"  - Partition: {partition}")
    print(f"  - Long-tail: {longtail}")
    print(f"  - Long-tail Type: {user_type if longtail and 'user_type' in locals() else '-'}")
    print(f"  - Imbalance Factor: {imb_factor if imb_factor is not None else imbalance_factor}")
    print(f"  - Dirichlet alpha: {alpha_override if alpha_override is not None else du.alpha}")
    print(f"  - Seed: {seed if seed is not None else '-'}")
    print(f"  - Num Clients: {num_clients}")

    # 设置全局随机种子
    if seed is not None:
        try:
            import torch
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    generate_dataset(dir_path, num_clients, niid, balance, partition, longtail, longtail_type, imb_factor, alpha_override, seed)