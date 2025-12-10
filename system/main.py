#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import subprocess
import sys

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.servermoon import MOON
from flcore.servers.serverlc import FedLC
from flcore.servers.servercreff import FedCReFF
from flcore.servers.serverclip2fl import FedCLIP2FL
from flcore.servers.serverrucr import FedRUCR
from flcore.servers.serverfedetf import FedETF
from flcore.servers.serverloge import FedLoGe
from flcore.servers.serverfedic import FedIC
from flcore.servers.servergrab import FedGraB
from flcore.servers.serverfednh import FedNHNative

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.resnet_cifar import resnet8_cifar, resnet8_cifar_512
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

# Performance optimizations (benchmark setting moved to after random seed configuration)
try:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        # Enable mixed precision training for faster computation (Ampere+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass


def run(args):
    
    # Set random seed (following best practices)
    if args.random_seed == 0:
        # Use time-based random seed for robustness testing
        seed = int(time.time()) % 10000
        print(f"Random seed: {seed} (time-based)")
    else:
        # Use fixed seed for reproducibility
        seed = args.random_seed
        print(f"Random seed: {seed} (fixed)")
    
    # Set all random seeds
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Configure cudnn based on reproducibility needs
        if args.random_seed != 0:
            # Fixed seed: prioritize reproducibility over speed
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # Random seed: prioritize speed (for robustness testing)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    time_list = []
    reporter = MemReporter()
    model_str = args.model
    
    # 标准化模型名称（支持小写输入）
    model_name_mapping = {
        'resnet18': 'ResNet18',
        'resnet20': 'ResNet20', 
        'resnet10': 'ResNet10',
        'resnet8': 'ResNet8',
        'resnet34': 'ResNet34',
        'cnn': 'CNN',
        'dnn': 'DNN',
        'mlr': 'MLR'
    }
    if model_str.lower() in model_name_mapping:
        model_str = model_name_mapping[model_str.lower()]

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # FedLoGe uses internal server implementation

        # Generate args.model
        if model_str == "MLR": # convex
            if "MNIST" in args.dataset:
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "CNN": # non-convex
            # CCVR requires (feature, output) tuple format
            if args.algorithm == "CCVR":
                from flcore.trainmodel.cnn import CNN_FeatureOutput
                args.model = CNN_FeatureOutput(num_classes=args.num_classes, feature_dim=256).to(args.device)
            elif "MNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "Omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "DNN": # non-convex
            if "MNIST" in args.dataset:
                args.model = DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet18":
            # FedGraB 和 FedLoGe 使用 resnet_cifar.py 中的 CIFAR 风格 ResNet18
            # 它们在各自的 server 中初始化模型，这里跳过
            if args.algorithm in ["FedGraB", "fedgrab", "FedLoGe", "fedloge"]:
                pass  # Model will be created in server's __init__
            else:
                # 其他算法使用 resnet.py 的 ImageNet 风格 resnet18
                args.model = resnet18(num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet20" or model_str == "resnet20":
            # FedETF uses specialized ResNet20 with ETF classifier
            if args.algorithm == "FedETF" or args.algorithm == "fedetf":
                from flcore.trainmodel.resnet_fedetf import ResNet20_FedETF
                args.model = ResNet20_FedETF(num_classes=args.num_classes, device=args.device).to(args.device)
            else:
                # Other algorithms use standard resnet20
                args.model = resnet20(num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet8":
            # CReFF, FEDIC, CCVR, RUCR use ResNet8 with feature_dim=256 (returns feature+logit)
            if args.algorithm in ["CReFF", "FedIC", "fedic", "CCVR", "RUCR"]:
                args.model = resnet8_cifar(num_classes=args.num_classes, scaling=4).to(args.device)
            # CLIP2FL uses ResNet8 with feature_dim=512 (has add_mlp layer)
            elif args.algorithm == "CLIP2FL":
                args.model = resnet8_cifar_512(num_classes=args.num_classes, scaling=4).to(args.device)
            else:
                # Other algorithms use standard resnet8
                args.model = resnet8(num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "AlexNet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif model_str == "GoogleNet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False, 
                                                      num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "MobileNet":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif model_str == "LSTM":
            args.model = LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "BiLSTM":
            args.model = BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, 
                                                   output_size=args.num_classes, num_layers=1, 
                                                   embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                                                   embedding_length=args.feature_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size, 
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, 
                                          num_classes=args.num_classes, max_len=args.max_len).to(args.device)
        
        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "HARCNN":
            if args.dataset == 'HAR':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                    pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'PAMAP2':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                    pool_kernel_size=(1, 2)).to(args.device)

        else:
            raise NotImplementedError

        # 打印模型结构（跳过在 server 中创建模型的算法）
        if args.algorithm not in ["FedGraB", "fedgrab", "FedLoGe", "fedloge"]:
            print(args.model)

        # select algorithm (kept only)
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)

        elif args.algorithm == "FedLC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedLC(args, i)

        elif args.algorithm == "CReFF":
            # CReFF uses full end-to-end model (backbone + classifier)
            # DO NOT split head like other algorithms
            server = FedCReFF(args, i)

        elif args.algorithm == "CLIP2FL":
            # CLIP2FL uses full end-to-end model (backbone + classifier)
            # DO NOT split head like other algorithms
            server = FedCLIP2FL(args, i)

        elif args.algorithm == "RUCR":
            # RUCR uses full model that returns (feature, output) tuple
            # No head split needed
            from flcore.servers.serverrucr import FedRUCR
            server = FedRUCR(args, i)

        elif args.algorithm == "FedNH":
            args.model = args.model.cpu()
            server = FedNHNative(args, i)
        
        elif args.algorithm == "FedNH2" or args.algorithm == "fednh2":
            # FedNH2: Direct copy from FedNH-main source code
            server = FedNH2(args, i)

        elif args.algorithm == "FedLoGe":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedLoGe(args, i)
        
        elif args.algorithm == "fedloge" or args.algorithm == "fedloge2":
            # FedLoGe: Direct copy from source code
            # Uses standard ResNet (no BaseHeadSplit wrapper)
            server = FedLoGe(args, i)

        elif args.algorithm == "FedGraB":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGraB(args, i)
        
        
        elif args.algorithm == "FedETF" or args.algorithm == "fedetf":
            # FedETF: ResNet with integrated ETF classifier
            # Parameters are set in serverfedetf.py __init__
            from flcore.servers.serverfedetf import FedETF
            server = FedETF(args, i)
            
        elif args.algorithm == "CCVR":
            from flcore.servers.serverccvr import FedCCVR
            server = FedCCVR(args, i)
        
        elif args.algorithm == "RUCR":
            from flcore.servers.serverrucr import FedRUCR
            server = FedRUCR(args, i)
        
        elif args.algorithm == "FedIC" or args.algorithm == "fedic":
            from flcore.servers.serverfedic import FedIC
            server = FedIC(args, i)
        
        elif args.algorithm == "FedGraB" or args.algorithm == "fedgrab":
            # FedGraB: Direct copy from source code
            # Model is initialized inside servergrab.py using resnet18_cifar
            from flcore.servers.servergrab import FedGraB
            server = FedGraB(args, i)
            
        elif args.algorithm == "FedCross":
            raise NotImplementedError("FedCross is requested to be kept but its server implementation is not present in the repo.")
            
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


def main():
    script_path = os.path.abspath(__file__)
    raise SystemExit(subprocess.call([sys.executable, script_path] + sys.argv[1:]))

if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-seed', "--random_seed", type=int, default=7,
                        help="Random seed (default=7 for reproducibility, 0 for time-based random)")
    parser.add_argument('-lbs', "--batch_size", type=int, default=32)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.10,
                        help="Local learning rate")
    parser.add_argument('--local_momentum', type=float, default=0.9,
                        help="Local SGD momentum for client optimizers")
    
    # 条件性参数注册 - 避免不同算法间的参数冲突
    # 注意：这里需要在解析算法参数之前进行预解析
    import sys
    temp_args = []
    for i, arg in enumerate(sys.argv):
        if arg in ['-algo', '--algorithm'] and i + 1 < len(sys.argv):
            temp_args.append(sys.argv[i + 1].lower())
            break
    
    # 根据算法注册相应的参数
    if 'fedic' in temp_args:
        FedIC.register_cli_aliases(parser)
    elif 'rucr' in temp_args:
        FedRUCR.register_cli_args(parser)
    elif 'ccvr' in temp_args:
        pass  # CCVR parameters are handled in __init__
    elif 'clip2fl' in temp_args:
        pass  # CLIP2FL parameters are handled in __init__
    elif 'creff' in temp_args:
        pass  # CReFF parameters are handled in __init__
    elif 'fedetf' in temp_args:
        pass  # FedETF parameters are handled in __init__
    elif 'fedloge' in temp_args or 'fedloge2' in temp_args:
        pass  # FedLoGe parameters are handled in __init__
    
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=True)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=200)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=10, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.4,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Total running times")
    
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    
    # CReFF & CLIP2FL parameters (aligned with source code defaults)
    parser.add_argument('--lr_feature', type=float, default=0.1,
                        help='Learning rate for updating synthetic features')
    parser.add_argument('--lr_net', type=float, default=0.01,
                        help='Learning rate for network parameters')
    parser.add_argument('--num_of_feature', type=int, default=100,
                        help='Number of synthetic features per class')
    parser.add_argument('--match_epoch', type=int, default=100,
                        help='Epochs for feature gradient matching')
    parser.add_argument('--crt_epoch', type=int, default=300,
                        help='Epochs for classifier retraining on synthetic features')
    parser.add_argument('--batch_real', type=int, default=32,
                        help='Batch size for real data gradient computation')
    parser.add_argument('--dis_metric', type=str, default='ours',
                        help='Distance metric for gradient matching: ours/mse/cos')
    parser.add_argument('--num_epochs_local_training', type=int, default=10,
                        help='Local training epochs (CReFF/FEDIC)')
    parser.add_argument('--batch_size_local_training', type=int, default=128,
                        help='Local training batch size (CReFF: 32, FEDIC: 128)')
    parser.add_argument('--lr_local_training', type=float, default=0.1,
                        help='Local training learning rate (FEDIC: 0.1)')
    
    # FEDIC-specific parameters (aligned with FEDIC-main/options.py)
    parser.add_argument('--num_online_clients', type=int, default=8,
                        help='FEDIC: number of clients per round (default: 8)')
    parser.add_argument('--total_steps', type=int, default=100,
                        help='FEDIC: total steps (default: 100)')
    parser.add_argument('--server_steps', type=int, default=100,
                        help='FEDIC: server-side training steps (default: 100)')
    parser.add_argument('--mini_batch_size', type=int, default=20,
                        help='FEDIC: labeled batch size for server (default: 20)')
    parser.add_argument('--batch_size_test', type=int, default=500,
                        help='FEDIC: test batch size (default: 500)')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='FEDIC: KD temperature (default: 2.0)')
    parser.add_argument('--ensemble_ld', type=float, default=0.0,
                        help='FEDIC: ensemble KD weight (default: 0.0)')
    parser.add_argument('--ld', type=float, default=0.5,
                        help='FEDIC: KD loss weight (default: 0.5)')
    
    # CLIP2FL-specific parameters
    parser.add_argument('--T', type=float, default=3.0,
                        help='Temperature for knowledge distillation (CLIP2FL)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for KD loss (CLIP2FL)')
    parser.add_argument('--contrast_alpha', type=float, default=0.001,
                        help='Weight for contrastive loss on server (CLIP2FL)')
    parser.add_argument('--ins_temp', type=float, default=0.1,
                        help='Temperature for instance-level supervision (CLIP2FL)')
    
    # FedETF-specific parameters (aligned with FedETF-main/args.py)
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='FedETF: SGD momentum (default: 0.9)')
    parser.add_argument('--local_wd_rate', type=float, default=5e-4,
                        help='FedETF: local weight decay rate (default: 5e-4)')
    parser.add_argument('--scaling_train', type=float, default=1.0,
                        help='FedETF: scaling hyperparameter for training (default: 1.0)')
    parser.add_argument('--skip_personalization', action='store_true',
                        help='FedETF: skip personalization phase for quick testing')
    
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # FedProx / FedIC / MOON / FedLC 所需超参
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.01)
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # CCVR
    parser.add_argument('-L', "--L", type=float, default=1.0)

    # Upstream-style aliases to ease migration from reference CReFF scripts
    # NOTE: Upstream-style aliases for rounds/online clients/local epochs/batch size
    # are already registered via FedIC.register_cli_aliases(parser). Avoid duplicates here.
    parser.add_argument('--num_classrs', type=int, default=None,
                        help='Alias for number of classes (-ncl/--num_classes)')
    parser.add_argument('--non_iid_alpha', type=float, default=None,
                        help='Dirichlet alpha for non-iid label distribution (record only)')
    parser.add_argument('--imb_factor', type=float, default=None,
                        help='Global long-tail imbalance factor (record only)')

    args = parser.parse_args()
    # Resolve device: default to GPU if available, else fallback to CPU
    if args.device == "cuda":
        if torch.cuda.is_available():
            dev_id = getattr(args, "device_id", "0")
            try:
                args.device = f"cuda:{int(dev_id)}"
            except Exception:
                args.device = "cuda:0"
        else:
            print("[Warn] CUDA not available; falling back to CPU.")
            args.device = "cpu"

    # Normalize algorithm and model names (case-insensitive; ignore '-', '_', and spaces)
    def _norm_key(s):
        if not isinstance(s, str):
            return s
        return ''.join(ch for ch in s.lower() if ch.isalnum())

    _algo_map = {
        # traditional kept
        'fedavg': 'FedAvg',
        'fedprox': 'FedProx',
        'moon': 'MOON',
        'fedlc': 'FedLC',
        'fedcross': 'FedCross',  # placeholder; server not present in repo yet
        # long-tail kept
        'creff': 'CReFF',
        'clip2fl': 'CLIP2FL',
        'rucr': 'RUCR',
        'fedetf': 'FedETF',
        'fedloge': 'FedLoGe',
        'fedloge2': 'FedLoGe',  # Direct copy from source
        'fednh': 'FedNH',
        'fedic': 'FedIC',
        'fedgrab': 'FedGraB',
        'ccvr': 'CCVR',
    }

    _model_map = {
        'cnn': 'CNN',
        'dnn': 'DNN',
        'resnet18': 'ResNet18',
        'resnet10': 'ResNet10',
        'resnet8': 'ResNet8',
        'resnet34': 'ResNet34',
        'alexnet': 'AlexNet',
        'googlenet': 'GoogleNet',
        'mobilenet': 'MobileNet',
        'lstm': 'LSTM',
        'bilstm': 'BiLSTM',
        'fasttext': 'fastText',
        'textcnn': 'TextCNN',
        'transformer': 'Transformer',
        'amazonmlp': 'AmazonMLP',
        'harcnn': 'HARCNN',
    }

    # Normalize dataset names to match folder casing under ../dataset
    _dataset_map = {
        'mnist': 'MNIST',
        'mnist1': 'MNIST1',
        'cifar10': 'Cifar10',
        'cifar100': 'Cifar100',
        'omniglot': 'Omniglot',
        'digit5': 'Digit5',
        'agnews': 'AG_News',
        'sogounews': 'SogouNews',
        'har': 'HAR',
        'pamap2': 'PAMAP2',
        'iwildcam': 'iWildCam',
        'flowers102': 'Flowers102',
        'gtsrb': 'GTSRB',
        'country211': 'Country211',
        'domainnet': 'DomainNet',
        'covidx': 'COVIDx',
        'emnist': 'EMNIST',
        'femnist': 'FEMNIST',
        'kvasir': 'kvasir',
        'camelyon17': 'Camelyon17',
    }

    if isinstance(args.algorithm, str):
        args.algorithm = _algo_map.get(_norm_key(args.algorithm), args.algorithm)
    if isinstance(args.model, str):
        args.model = _model_map.get(_norm_key(args.model), args.model)
    if isinstance(args.dataset, str):
        args.dataset = _dataset_map.get(_norm_key(args.dataset), args.dataset)

    # CRITICAL FIX: Apply algorithm-specific parameter mappings BEFORE calling run(args)
    # Harmonize official CLIP2FL flags with internal ones (single, canonical block)
    if args.algorithm == "CLIP2FL":
        # If user didn't explicitly set alpha (it's the parser default) and didn't pass kd_alpha,
        # neutralize alpha to 0.0 so KD is OFF by default for CLIP2FL.
        try:
            _alpha_default = parser.get_default('alpha')
        except Exception:
            _alpha_default = 1.0
        if (getattr(args, 'alpha', None) == _alpha_default) and getattr(args, 'kd_alpha', None) in (None, 0.0):
            args.alpha = 0.0
        # Prefer explicit alpha; fall back to kd_alpha only if alpha not set and kd_alpha > 0
        if (getattr(args, 'alpha', None) is None or args.alpha == 0.0) and getattr(args, 'kd_alpha', None) not in (None, 0.0):
            args.alpha = args.kd_alpha
        # Map lr_local_training to internal local_learning_rate when provided
        if getattr(args, 'lr_local_training', None) is not None:
            args.local_learning_rate = args.lr_local_training
        # Map crt_epoch (classifier retraining epochs) to internal ctr_epoch
        if getattr(args, 'crt_epoch', None) is not None:
            args.ctr_epoch = args.crt_epoch
        # Map lr_net (classifier retraining LR) to internal ctr_lr
        if getattr(args, 'lr_net', None) is not None:
            args.ctr_lr = args.lr_net

    if args.algorithm == "RUCR":
        # Keep RUCR defaults as defined by parser (aligned with options.py); do not override here.
        pass

    # CRITICAL FIX: Call run(args) AFTER all parameter mappings are complete
    run(args)

