import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from sklearn import metrics
from flcore.clients.clientavg import clientAVG


class clientFedNH(clientAVG):
    """Native FedNH client.

    - Trains backbone locally using prototype-based logits.
    - Maintains a per-class prototype matrix and uploads classwise sums and counts.
    - Evaluates using prototype classifier rather than the linear head.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.fednh_client_adv = getattr(args, 'FedNH_client_adv_prototype_agg', False)
        self.fednh_scale = getattr(args, 'FedNH_scale', 20.0)  # FedNH-main default: 20.0

        # Infer embedding dimension from model definition
        d = self._infer_feature_dim(self.model)
        self.embed_dim = d
        # Local prototype matrix (following FedNH-main FedUH.py lines 59-60: orthogonal init)
        proto_init = torch.nn.init.orthogonal_(torch.rand(self.num_classes, d))
        self.prototype = nn.Parameter(proto_init.clone().to(self.device), requires_grad=False)
        # Train-time scale parameter
        self.scaling_train = nn.Parameter(torch.tensor(float(self.fednh_scale), device=self.device), requires_grad=False)
        
        # Debug: print scaling for first client only
        if id == 0:
            print(f"[FedNH Client {id}] scaling_train={self.scaling_train.item()}")

        # Build count_by_class dict (only classes this client has) - following FedNH-main client.py line 43
        # In FedNH-main: self.count_by_class = Counter(self.trainset.targets.numpy())
        # We build it manually since we use dataloader
        from collections import Counter
        self.count_by_class = {}
        trainloader = self.load_train_data()
        for _, y in trainloader:
            for label in y.numpy():
                self.count_by_class[label] = self.count_by_class.get(label, 0) + 1
        
        # Build count_by_class_full tensor (all classes, use 1e-12 for missing classes) - FedNH-main FedNH.py line 26-27
        # Source: temp = [self.count_by_class[cls] if cls in self.count_by_class.keys() else 1e-12 for ...]
        temp = [self.count_by_class[cls] if cls in self.count_by_class.keys() else 1e-12 for cls in range(self.num_classes)]
        self.count_by_class_full = torch.tensor(temp).to(self.device)
        
        # Compute label distribution for PM(L) metric (FedNH-main FedUH.py line 140-142)
        total_samples = sum(self.count_by_class.values())
        self.label_dist = {cls: count / total_samples for cls, count in self.count_by_class.items()}

        # Buffers used to upload to server
        self.proto_sum = torch.zeros(self.num_classes, d)
        self.proto_sum_adv = torch.zeros(self.num_classes, d)

    # --------------------- helpers ---------------------
    def set_prototype(self, global_proto: torch.Tensor):
        """Receive global prototype from server."""
        assert global_proto.shape == (self.num_classes, self.embed_dim)
        self.prototype.data = global_proto.detach().clone().to(self.device)
        # keep normalized
        self.prototype.data = F.normalize(self.prototype.data, p=2, dim=1)

    def _infer_feature_dim(self, model: nn.Module) -> int:
        # Prefer taking in_features of final classifier
        if hasattr(model, 'fc') and isinstance(getattr(model, 'fc'), nn.Linear):
            return int(model.fc.in_features)
        # If model has fc1 sequence, use the first linear out_features
        if hasattr(model, 'fc1'):
            fc1 = getattr(model, 'fc1')
            if isinstance(fc1, nn.Sequential):
                for layer in fc1:
                    if isinstance(layer, nn.Linear):
                        return int(layer.out_features)
            if isinstance(fc1, nn.Linear):
                return int(fc1.out_features)
        # ResNet-style: avgpool -> fc, use fc.in_features if present
        if hasattr(model, 'fc') and isinstance(getattr(model, 'fc'), nn.Linear):
            return int(model.fc.in_features)
        # Fallback
        return 512

    def _forward_to_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings before final classifier.
        Heuristics based on common models in this repo.
        """
        m = self.model
        # ResNet-style path: convs -> layers -> avgpool -> flatten
        try:
            if hasattr(m, 'avgpool') and hasattr(m, 'fc') and hasattr(m, 'conv1') and hasattr(m, 'layers'):
                out = m.conv1(x)
                if hasattr(m, 'bn1'):
                    out = m.bn1(out)
                out = m.relu(out)
                out = m.maxpool(out)
                for i in range(len(m.layers)):
                    layer = getattr(m, f'layer_{i}')
                    out = layer(out)
                out = m.avgpool(out)
                return out
        except Exception:
            pass
        # CNN path: conv1 -> conv2 -> flatten -> fc1 gives features
        if hasattr(m, 'conv1') and hasattr(m, 'conv2') and hasattr(m, 'fc1'):
            out = m.conv1(x)
            out = m.conv2(out)
            out = torch.flatten(out, 1)
            fc1 = getattr(m, 'fc1')
            if isinstance(fc1, nn.Sequential):
                emb = fc1(out)
            else:
                emb = m.fc1(out)
            return emb
        # MLP/DNN path: use fc1 output as embedding
        if hasattr(m, 'fc1'):
            x_in = x
            if x_in.ndim == 4:
                x_in = x_in.view(x_in.size(0), -1)
            fc1 = getattr(m, 'fc1')
            if isinstance(fc1, nn.Sequential):
                return fc1(x_in)
            elif isinstance(fc1, nn.Linear):
                return fc1(x_in)
        # fallback to full forward
        out = m(x)
        return out

    def _proto_logits(self, emb: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings and prototypes, then compute scaled dot-product
        emb_norm = F.normalize(emb, p=2, dim=1)
        proto_norm = F.normalize(self.prototype, p=2, dim=1)
        logits = self.scaling_train * torch.matmul(emb_norm, proto_norm.t())
        return logits

    # -------------------- training ---------------------
    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # Standard training loop (following FedNH-main)
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                    x_in = x[0]
                else:
                    x_in = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                emb = self._forward_to_embedding(x_in)
                logits = self._proto_logits(emb)
                loss = self.loss(logits, y)

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping (following FedNH-main FedUH.py line 107)
                torch.nn.utils.clip_grad_norm_(
                    parameters=filter(lambda p: p.requires_grad, self.model.parameters()),
                    max_norm=10
                )
                self.optimizer.step()

        # After training, estimate prototype (following FedNH-main lines 29-58)
        self._estimate_prototype_after_training()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        self.model.cpu()

    # -------------------- metrics ----------------------
    @torch.no_grad()
    def test_metrics(self):
        """Basic test metrics for compatibility (uniform weighting)"""
        testloader = self.load_test_data()
        self.model.to(self.device)
        self.model.eval()

        tot_correct = 0
        tot_auc = 0
        num_samples = 0

        y_true = []
        y_score = []

        for x, y in testloader:
            if isinstance(x, list):
                x[0] = x[0].to(self.device)
                x_in = x[0]
            else:
                x_in = x.to(self.device)
            y = y.to(self.device)

            emb = self._forward_to_embedding(x_in)
            logits = self._proto_logits(emb)

            pred = torch.argmax(logits, dim=1)
            tot_correct += (pred == y).sum().item()
            num_samples += y.size(0)

            y_true.append(y.detach().cpu())
            y_score.append(F.softmax(logits.detach(), dim=1).cpu())

        # compute AUC if possible
        try:
            y_true_np = torch.cat(y_true).numpy()
            y_score_np = torch.cat(y_score).numpy()
            y_true_1hot = metrics.label_binarize(y_true_np, classes=list(range(self.num_classes)))
            auc = metrics.roc_auc_score(y_true_1hot, y_score_np, average='macro')
        except Exception:
            auc = 0.0

        self.model.cpu()
        return tot_correct, num_samples, auc

    @torch.no_grad()
    def test_metrics_personalized(self):
        """Personalized test metrics with three criteria: uniform, validclass, labeldist (FedNH-main)"""
        testloader = self.load_test_data()
        self.model.to(self.device)
        self.model.eval()
        
        # Count samples per class in test set
        test_count_per_class = {}
        for _, y in testloader:
            for label in y.numpy():
                test_count_per_class[label] = test_count_per_class.get(label, 0) + 1
        
        # Convert to tensor (FedNH-main FedUH.py line 134)
        test_count_per_class_tensor = torch.tensor(
            [test_count_per_class.get(cls, 0) * 1.0 for cls in range(self.num_classes)]
        )
        test_correct_per_class = torch.zeros(self.num_classes)
        
        # Define weight per class for three criteria (FedNH-main FedUH.py line 137-142)
        weight_per_class_dict = {
            'uniform': torch.ones(self.num_classes),  # All classes weighted equally
            'validclass': torch.zeros(self.num_classes),  # Only classes this client has
            'labeldist': torch.zeros(self.num_classes)  # Weighted by client's label distribution
        }
        for cls in self.label_dist.keys():
            weight_per_class_dict['labeldist'][cls] = self.label_dist[cls]
            weight_per_class_dict['validclass'][cls] = 1.0
        
        # Run inference and count correct predictions per class
        for x, y in testloader:
            if isinstance(x, list):
                x_in = x[0].to(self.device)
            else:
                x_in = x.to(self.device)
            y = y.to(self.device)
            
            emb = self._forward_to_embedding(x_in)
            logits = self._proto_logits(emb)
            pred = torch.argmax(logits, dim=1)
            
            # Count correct predictions per class (FedNH-main FedUH.py line 152-153)
            classes_in_batch = torch.unique(y).cpu().numpy()
            for cls in classes_in_batch:
                test_correct_per_class[cls] += ((pred == y) * (y == cls)).sum().item()
        
        # Compute accuracy by criteria (FedNH-main FedUH.py line 154-157)
        acc_by_criteria = {}
        for criteria_name in weight_per_class_dict.keys():
            weights = weight_per_class_dict[criteria_name]
            weighted_correct = (weights * test_correct_per_class).sum()
            weighted_total = (weights * test_count_per_class_tensor).sum()
            acc_by_criteria[criteria_name] = (weighted_correct / weighted_total).item() if weighted_total > 0 else 0.0
        
        self.model.cpu()
        return acc_by_criteria, test_correct_per_class, test_count_per_class_tensor

    @torch.no_grad()
    def _estimate_prototype_after_training(self):
        """Estimate prototype after training (following FedNH-main lines 29-58)"""
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()
        
        d = self.embed_dim
        prototype = torch.zeros(self.num_classes, d, device=self.device)
        
        # Accumulate embeddings (only iterate, accumulation done per-class)
        for x, y in trainloader:
            if isinstance(x, list):
                x_in = x[0].to(self.device)
            else:
                x_in = x.to(self.device)
            y = y.to(self.device)
            
            # Extract normalized embeddings
            emb = self._forward_to_embedding(x_in)
            emb_norm = F.normalize(emb, p=2, dim=1)
            
            # Accumulate per class (only classes shown in this batch)
            classes_shown = torch.unique(y).cpu().numpy()
            for cls in classes_shown:
                mask = (y == cls)
                prototype[cls] += emb_norm[mask].sum(dim=0)
        
        # Compute mean and normalize (following source lines 45-53)
        # CRITICAL: Only process classes this client actually has
        scaled_prototype = torch.zeros_like(prototype)
        for cls in self.count_by_class.keys():
            # Mean
            prototype[cls] /= self.count_by_class[cls]
            # Normalize
            proto_norm = torch.norm(prototype[cls]).clamp(min=1e-12)
            prototype[cls] = torch.div(prototype[cls], proto_norm)
            # Scale by count for aggregation (line 53)
            scaled_prototype[cls] = prototype[cls] * self.count_by_class[cls]
        
        # Store for upload (keep on device, matching FedNH-main line 57)
        self.proto_sum = scaled_prototype.detach()
        
        # Adversarial aggregation if enabled
        if self.fednh_client_adv:
            self._estimate_prototype_adv()
    
    @torch.no_grad()
    def _estimate_prototype_adv(self):
        """Adversarial prototype estimation (following FedNH-main lines 60-93)"""
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()
        
        embeddings = []
        labels = []
        weights = []
        
        for x, y in trainloader:
            if isinstance(x, list):
                x_in = x[0].to(self.device)
            else:
                x_in = x.to(self.device)
            y = y.to(self.device)
            
            emb = self._forward_to_embedding(x_in)
            logits = self._proto_logits(emb)
            emb_norm = F.normalize(emb, p=2, dim=1)
            
            # Confidence weighting
            prob = F.softmax(logits, dim=1)
            prob_true = torch.gather(prob, dim=1, index=y.view(-1, 1))
            
            embeddings.append(emb_norm)
            labels.append(y)
            weights.append(prob_true)
        
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        weights = torch.cat(weights, dim=0).view(-1, 1)
        
        # CRITICAL: Only process classes this client actually has (following FedNH-main line 83)
        prototype_adv = torch.zeros(self.num_classes, self.embed_dim, device=self.device)
        for cls in self.count_by_class.keys():
            mask = (labels == cls)
            if mask.any():
                weights_cls = weights[mask]
                emb_cls = embeddings[mask]
                prototype_adv[cls] = (emb_cls * weights_cls).sum(dim=0) / weights_cls.sum()
                # Normalize
                proto_norm = torch.norm(prototype_adv[cls]).clamp(min=1e-12)
                prototype_adv[cls] = torch.div(prototype_adv[cls], proto_norm)
        
        # Store for upload (keep on device, matching FedNH-main)
        self.proto_sum_adv = prototype_adv.detach()
    
    # -------------------- upload API -------------------
    @torch.no_grad()
    def get_upload_package(self):
        """Package prototype statistics for server aggregation (matching FedNH-main line 57, 92)."""
        pkg = {
            'scaled_prototype': self.proto_sum,
            'count_by_class_full': self.count_by_class_full
        }
        if self.fednh_client_adv:
            pkg['adv_agg_prototype'] = self.proto_sum_adv
        return pkg

    # -------------------- extra metrics for FedNH --------------------
    @torch.no_grad()
    def train_metrics(self):
        """Compute training loss on local train set using prototype logits."""
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        num_samples = 0
        for x, y in trainloader:
            if isinstance(x, list):
                x[0] = x[0].to(self.device)
                x_in = x[0]
            else:
                x_in = x.to(self.device)
            y = y.to(self.device)

            emb = self._forward_to_embedding(x_in)
            logits = self._proto_logits(emb)
            loss = self.loss(logits, y)
            total_loss += loss.item() * y.size(0)
            num_samples += y.size(0)

        self.model.cpu()
        return total_loss / max(num_samples, 1), num_samples
    
    @torch.no_grad()
    def train_metrics_acc(self):
        """Compute training accuracy on local train set using prototype logits."""
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()
        tot_correct = 0
        num_samples = 0
        for x, y in trainloader:
            if isinstance(x, list):
                x[0] = x[0].to(self.device)
                x_in = x[0]
            else:
                x_in = x.to(self.device)
            y = y.to(self.device)
            emb = self._forward_to_embedding(x_in)
            logits = self._proto_logits(emb)
            pred = torch.argmax(logits, dim=1)
            tot_correct += (pred == y).sum().item()
            num_samples += y.size(0)
        self.model.cpu()
        return tot_correct, num_samples

    @torch.no_grad()
    def test_metrics_per_class(self):
        """Return per-class correct counts and totals on test set using prototype logits."""
        testloader = self.load_test_data()
        # move to device for safe forward
        self.model.to(self.device)
        self.model.eval()
        correct_by_class = torch.zeros(self.num_classes, dtype=torch.long)
        total_by_class = torch.zeros(self.num_classes, dtype=torch.long)
        for x, y in testloader:
            if isinstance(x, list):
                x[0] = x[0].to(self.device)
                x_in = x[0]
            else:
                x_in = x.to(self.device)
            y = y.to(self.device)
            emb = self._forward_to_embedding(x_in)
            logits = self._proto_logits(emb)
            pred = torch.argmax(logits, dim=1)
            for cls in range(self.num_classes):
                mask = (y == cls)
                if mask.any():
                    total_by_class[cls] += int(mask.sum().item())
                    correct_by_class[cls] += int((pred[mask] == cls).sum().item())
        # move back to CPU to keep server-side aggregation consistent
        self.model.cpu()
        return correct_by_class.cpu(), total_by_class.cpu()