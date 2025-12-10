"""
CLIP2FL: Integrating CLIP into Federated Learning for Long-Tailed Data

Reference:
    Lu et al. "CLIP2FL: Integrating Pretrained CLIP into Federated Learning for Long-Tailed Data"
    https://github.com/Laughing-q/CLIP2FL
"""

import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from flcore.clients.clientbase import Client


# DiffAugment for data augmentation
class ParamDiffAug:
    def __init__(self):
        self.aug_mode = 'S'  # 'M' for multiple, 'S' for single
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5
        self.Siamese = False
        self.latestseed = -1


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:
        randb[:] = randb[0].clone()
    x = x + (randb - 0.5) * ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:
        rands[:] = rands[0].clone()
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:
        randc[:] = randc[0].clone()
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:
        translation_x[:] = translation_x[0].clone()
        translation_y[:] = translation_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:
        offset_x[:] = offset_x[0].clone()
        offset_y[:] = offset_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese:
        randf[:] = randf[0].clone()
    return torch.where(randf < prob, x.flip(3), x)


def rand_scale(x, param):
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0, 0], [0, sy[i], 0]] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=False).to(x.device)
    x = F.grid_sample(x, grid, align_corners=False)
    return x


def rand_rotate(x, param):
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
              [torch.sin(theta[i]), torch.cos(theta[i]), 0]] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese:
        theta[:] = theta[0].clone()
    grid = F.affine_grid(theta, x.shape, align_corners=False).to(x.device)
    x = F.grid_sample(x, grid, align_corners=False)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}


def DiffAugment(x, strategy='', seed=-1, param=None):
    """
    Differentiable Siamese Augmentation - aligned with CLIP2FL source code
    """
    if param is None:
        param = ParamDiffAug()
    
    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True
    param.latestseed = seed
    
    if strategy == 'None' or strategy == 'none' or not strategy:
        return x
    
    if param.aug_mode == 'M':  # Multiple augmentations
        for p in strategy.split('_'):
            if p in AUGMENT_FNS:
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
    elif param.aug_mode == 'S':  # Single random augmentation
        pbties = strategy.split('_')
        set_seed_DiffAug(param)
        p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
        if p in AUGMENT_FNS:
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
    
    return x.contiguous()


# ==================== End DiffAugment ====================


class KDLoss(nn.Module):
    """
    Knowledge Distillation Loss
    Aligned with CLIP2FL-main/main.py
    """
    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        kd_loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T
        return kd_loss


def get_class_num(class_list):
    """
    Extract non-zero classes and their counts
    """
    index = []
    compose = []
    for class_index, j in enumerate(class_list):
        if j != 0:
            index.append(class_index)
            compose.append(j)
    return index, compose


class Local(object):
    """Local client for gradient computation and local training with CLIP KD"""
    def __init__(self, data_client, class_list: list, args, clip_model=None, text_features=None):
        from torch.optim import SGD
        from torch.nn import CrossEntropyLoss
        from flcore.trainmodel.resnet_cifar import resnet8_cifar_512
        
        self.data_client = data_client
        self.device = args.device
        self.class_compose = class_list
        self.criterion = CrossEntropyLoss().to(args.device)
        self.kd_criterion = KDLoss(T=args.T).to(args.device)
        self.local_model = resnet8_cifar_512(num_classes=args.num_classes, scaling=4).to(args.device)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training)
        self.clip_model = clip_model
        self.text_features = text_features
        self.dsa = getattr(args, 'dsa', True)
        self.dsa_strategy = getattr(args, 'dsa_strategy', 'color_crop_cutout_flip_scale_rotate')
        self.dsa_param = ParamDiffAug()

    def compute_gradient(self, global_params, args):
        """Compute real feature gradients"""
        from torch.nn import CrossEntropyLoss
        from torch import unsqueeze
        
        list_class, per_class_compose = get_class_num(self.class_compose)
        indices_class = {class_index: [] for class_index in list_class}

        images_all = [unsqueeze(self.data_client[i][0], dim=0) for i in range(len(self.data_client))]
        labels_all = [int(self.data_client[i][1]) for i in range(len(self.data_client))]
        for i, lab in enumerate(labels_all):
            if lab in indices_class:
                indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)

        def get_images(c, n):
            if len(indices_class[c]) == 0:
                return None
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        self.local_model.load_state_dict(global_params)
        self.local_model.eval()
        self.local_model.classifier.train()
        net_parameters = list(self.local_model.classifier.parameters())
        criterion = CrossEntropyLoss().to(args.device)
        truth_gradient_all = {index: [] for index in list_class}

        for num_compute in range(10):
            for c, num in zip(list_class, per_class_compose):
                img_real = get_images(c, args.batch_real)
                if img_real is None or img_real.shape[0] == 0:
                    continue
                if self.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, self.dsa_strategy, seed=seed, param=self.dsa_param)

                lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                feature_real, output_real = self.local_model(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))
                truth_gradient_all[c].append(gw_real)

        truth_gradient_avg = {}
        for i in list_class:
            gradient_all = truth_gradient_all[i]
            if len(gradient_all) == 0:
                continue
            gw_real_temp = []
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
        """Local training with CLIP KD"""
        from torch.utils.data import DataLoader
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        
        # Pre-compute normalization tensors
        if self.clip_model is not None:
            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
            clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
            cifar_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(self.device)
            cifar_std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(self.device)
        
        for _ in range(args.num_epochs_local_training):
            data_loader = DataLoader(dataset=self.data_client,
                                     batch_size=args.batch_size_local_training,
                                     shuffle=True)
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                images_original = images.clone()
                images = transform_train(images)

                _, outputs = self.local_model(images)
                outputs = outputs.float()

                loss1 = self.criterion(outputs, labels)
                loss = loss1
                
                if self.clip_model is not None and self.text_features is not None:
                    with torch.no_grad():
                        clip_images = images_original * cifar_std + cifar_mean
                        clip_images = torch.clamp(clip_images, 0, 1)
                        clip_images = F.interpolate(clip_images, size=(224, 224), mode='bicubic', align_corners=False)
                        clip_images = (clip_images - clip_mean) / clip_std
                        
                        image_features = self.clip_model.encode_image(clip_images)
                        image_features = image_features.float()
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        clip_logits = 100. * image_features @ self.text_features.T
                    
                    loss2 = self.kd_criterion(outputs, clip_logits)
                    loss = loss1 + args.alpha * loss2
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return self.local_model.state_dict()


class clientCLIP2FL(Client):
    """
    CLIP2FL Client - 100% aligned with source code
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.args = args
        self.criterion = nn.CrossEntropyLoss().to(args.device)
        self.kd_criterion = KDLoss(T=getattr(args, 'T', 3.0)).to(args.device)
        
        # DiffAugment parameters (aligned with source code options.py)
        self.dsa = getattr(args, 'dsa', True)  # Default True in source
        self.dsa_strategy = getattr(args, 'dsa_strategy', 'color_crop_cutout_flip_scale_rotate')
        self.dsa_param = ParamDiffAug()
        
        # Get class composition from data
        self.class_compose = self._get_class_composition()
        
        # Load CLIP model and text features
        self.load_clip_and_text_features(args)
        
    def _get_class_composition(self):
        """
        Compute class composition (number of samples per class) in client's data.
        """
        trainloader = self.load_train_data()
        class_counts = [0] * self.num_classes
        
        for _, labels in trainloader:
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            for label in labels:
                class_counts[int(label)] += 1
        
        return class_counts
    
    def load_clip_and_text_features(self, args):
        """
        Load CLIP model and generate text features
        """
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', args.device)
            self.clip_model.eval()
            
            # Get class names
            if args.num_classes == 10:
                label_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
            elif args.num_classes == 100:
                label_name = [f'class {i}' for i in range(100)]
            else:
                label_name = [f'class {i}' for i in range(args.num_classes)]
            
            # Generate text features
            text_inputs = clip.tokenize([f"a photo of a {c}" for c in label_name]).to(args.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_inputs)
            
            text_features = text_features.float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            self.text_features = text_features
            print(f"[Client {self.id}] CLIP model loaded successfully")
            
        except Exception as e:
            print(f"[Client {self.id}] Warning: CLIP not available ({e}). Proceeding without CLIP.")
            self.clip_model = None
            self.text_features = None
    
    def compute_gradient(self, global_params, args):
        """
        Compute real feature gradients - aligned with source code
        """
        # Get class composition
        list_class, per_class_compose = get_class_num(self.class_compose)
        
        # Load all client data
        trainloader = self.load_train_data()
        images_all = []
        labels_all = []
        indices_class = {class_index: [] for class_index in list_class}
        
        for x, y in trainloader:
            if type(x) == type([]):
                x = x[0]
            images_all.append(x)
            labels_all.append(y)
        
        if len(images_all) == 0:
            return {}
        
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.cat(labels_all, dim=0)
        if not isinstance(labels_all, torch.Tensor):
            labels_all = torch.tensor(labels_all, dtype=torch.long)
        labels_all = labels_all.to(args.device)
        
        # Build index mapping: class -> list of sample indices
        for i, lab in enumerate(labels_all.tolist()):
            if lab in indices_class:
                indices_class[lab].append(i)
        
        def get_images(c, n):
            """Get random n images from class c"""
            if c not in indices_class or len(indices_class[c]) == 0:
                return None
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]
        
        # Load global params
        self.model.load_state_dict(global_params)
        self.model.eval()
        
        # Set classifier to train mode
        if hasattr(self.model, 'classifier'):
            self.model.classifier.train()
            net_parameters = list(self.model.classifier.parameters())
        elif hasattr(self.model, 'head'):
            self.model.head.train()
            net_parameters = list(self.model.head.parameters())
        elif hasattr(self.model, 'fc'):
            self.model.fc.train()
            net_parameters = list(self.model.fc.parameters())
        else:
            raise ValueError("Model must have 'classifier', 'head', or 'fc' attribute")
        
        criterion = nn.CrossEntropyLoss().to(args.device)
        
        # Gradients of all classes
        truth_gradient_all = {index: [] for index in list_class}
        
        # Repeat 10 times for stability (aligned with source code line 351)
        for num_compute in range(10):
            for c, num in zip(list_class, per_class_compose):
                img_real = get_images(c, getattr(args, 'batch_real', 32))
                if img_real is None or img_real.shape[0] == 0:
                    continue
                
                # Apply DiffAugment (aligned with source code line 355-357)
                if self.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, self.dsa_strategy, seed=seed, param=self.dsa_param)
                
                lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                
                # Forward pass
                output = self.model(img_real)
                if isinstance(output, tuple):
                    feature_real, output_real = output
                else:
                    output_real = output
                
                loss_real = criterion(output_real, lab_real)
                
                # Compute real feature gradients of class c
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))
                truth_gradient_all[c].append(gw_real)
        
        # Average gradients across 10 repetitions
        truth_gradient_avg = {}
        for i in list_class:
            gradient_all = truth_gradient_all[i]
            if len(gradient_all) == 0:
                continue
            
            gw_real_temp = []
            weight = 1.0 / len(gradient_all)
            for name_param in range(len(gradient_all[0])):
                list_values_param = []
                for client_one in gradient_all:
                    list_values_param.append(client_one[name_param] * weight)
                value_global_param = sum(list_values_param)
                gw_real_temp.append(value_global_param)
            
            truth_gradient_avg[i] = gw_real_temp
        
        return truth_gradient_avg
    
    def train(self):
        """
        Local training with CLIP-guided KD - aligned with source code
        
        Source code (main.py line 388-413):
        - images: augmented with RandomCrop + RandomHorizontalFlip, fed to local model
        - clip_images: ORIGINAL images with CLIP preprocess (224x224), fed to CLIP model
        - loss = CE(outputs, labels) + alpha * KD(outputs, clip_logits)
        """
        trainloader = self.load_train_data()
        start_time = time.time()
        
        self.model.train()
        
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        
        # Data augmentation (aligned with source code line 382-384)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])
        
        # Pre-compute normalization tensors for efficiency
        if self.clip_model is not None:
            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
            clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
            cifar_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(self.device)
            cifar_std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(self.device)
        
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                    images_original = x[0]
                else:
                    x = x.to(self.device)
                    images_original = x
                y = y.to(self.device)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                # IMPORTANT: Source code applies augmentation ONLY to images for model
                # clip_images use ORIGINAL images (line 393-394)
                images = transform_train(images_original)
                
                # Forward pass
                output = self.model(images)
                if isinstance(output, tuple):
                    _, outputs = output
                else:
                    outputs = output
                
                outputs = outputs.float()
                
                # CE loss
                loss1 = self.criterion(outputs, y)
                
                # CLIP KD loss
                loss = loss1
                if self.clip_model is not None and self.text_features is not None:
                    with torch.no_grad():
                        # Prepare CLIP images from ORIGINAL (non-augmented) images
                        # This aligns with source code: clip_images comes from Clip_Indices2Dataset
                        # which uses the original dataset with CLIP preprocess, NOT augmented
                        
                        # Step 1: Denormalize from CIFAR normalization
                        clip_images = images_original * cifar_std + cifar_mean
                        
                        # Step 2: Clamp to [0, 1] range
                        clip_images = torch.clamp(clip_images, 0, 1)
                        
                        # Step 3: Resize to 224x224 (CLIP input size) using bicubic interpolation
                        clip_images = F.interpolate(clip_images, size=(224, 224), mode='bicubic', align_corners=False)
                        
                        # Step 4: Apply CLIP normalization
                        clip_images = (clip_images - clip_mean) / clip_std
                        
                        # Encode with CLIP (aligned with source code line 403-406)
                        image_features = self.clip_model.encode_image(clip_images)
                        image_features = image_features.float()
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        clip_logits = (100.0 * image_features @ self.text_features.T)
                    
                    # KD loss (Eq. 1 in paper, source code line 407-410)
                    loss2 = self.kd_criterion(outputs, clip_logits)
                    loss = loss1 + getattr(self.args, 'alpha', 1.0) * loss2
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
