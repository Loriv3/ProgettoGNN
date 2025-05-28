import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import BatchNorm, LayerNorm
import numpy as np


class NoiseRobustGINClassifier(torch.nn.Module):
    """
    GIN classifier specifically designed for noisy ogbg-ppa dataset
    Handles symmetric and asymmetric label noise
    """
    def __init__(self, input_dim, hidden_dim, output_dim=6, num_layers=5, dropout=0.5,
                 use_mixup=True, use_label_smoothing=True, smoothing_factor=0.1,
                 use_self_training=False, confidence_threshold=0.9):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim
        self.use_mixup = use_mixup
        self.use_label_smoothing = use_label_smoothing
        self.smoothing_factor = smoothing_factor
        self.use_self_training = use_self_training
        self.confidence_threshold = confidence_threshold
        
        # Track training statistics for noise estimation
        self.register_buffer('loss_history', torch.zeros(1000))
        self.register_buffer('loss_idx', torch.tensor(0))
        
        # Layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # Input projection - PPA graphs have node features
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout * 0.3)
        )
        
        # GIN layers with noise-robust design
        for i in range(num_layers):
            # Wider MLPs help with noise robustness
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim * 3),
                torch.nn.BatchNorm1d(hidden_dim * 3),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout * 0.4),
                torch.nn.Linear(hidden_dim * 3, hidden_dim * 2),
                torch.nn.BatchNorm1d(hidden_dim * 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout * 0.3),
                torch.nn.Linear(hidden_dim * 2, hidden_dim)
            )
            
            self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(BatchNorm(hidden_dim))
        
        # Multi-scale pooling for better representation
        # Concatenate different pooling methods
        self.pool_methods = [global_add_pool, global_mean_pool, global_max_pool]
        pooled_dim = hidden_dim * 3
        
        # Classifier with uncertainty estimation
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(pooled_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.BatchNorm1d(hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout * 0.7)
        )
        
        # Final prediction layers
        self.final_layer = torch.nn.Linear(hidden_dim // 2, output_dim)
        
        # Uncertainty estimation head (for noise detection)
        self.uncertainty_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 4, 1),
            torch.nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better training with noise"""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def forward(self, data, return_uncertainty=False, apply_mixup=False, mixup_alpha=0.2):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # GIN layers with residual-like connections for noise robustness
        layer_outputs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            
            # Skip connection for deeper layers (helps with noise)
            if i > 0 and x.size() == x_new.size():
                x = x + x_new
            else:
                x = x_new
                
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)
        
        # Multi-scale pooling
        pooled_features = []
        for pool_fn in self.pool_methods:
            pooled_features.append(pool_fn(x, batch))
        
        graph_embed = torch.cat(pooled_features, dim=1)
        
        # Mixup augmentation during training
        if apply_mixup and self.training and self.use_mixup:
            graph_embed = self._apply_mixup(graph_embed, mixup_alpha)
        
        # Classification
        features = self.classifier(graph_embed)
        logits = self.final_layer(features)
        
        if return_uncertainty:
            uncertainty = self.uncertainty_head(features)
            return logits, uncertainty
        
        return logits
    
    def _apply_mixup(self, features, alpha=0.2):
        """Apply mixup augmentation to features"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
            batch_size = features.size(0)
            index = torch.randperm(batch_size).to(features.device)
            mixed_features = lam * features + (1 - lam) * features[index, :]
            return mixed_features
        return features


class NoiseRobustLoss(torch.nn.Module):
    """
    Combined loss function for handling label noise in PPA dataset
    """
    def __init__(self, num_classes=6, alpha=0.1, beta=1.0, gamma=2.0, 
                 use_focal=True, use_label_smoothing=True, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # Weight for uncertainty loss
        self.beta = beta    # Weight for consistency loss
        self.gamma = gamma  # Focal loss gamma
        self.use_focal = use_focal
        self.use_label_smoothing = use_label_smoothing
        self.smoothing = smoothing
        
    def forward(self, logits, targets, uncertainty=None, epoch=0):
        # Label smoothing for noise robustness
        if self.use_label_smoothing:
            targets_smooth = self._smooth_labels(targets)
            ce_loss = F.cross_entropy(logits, targets_smooth)
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Focal loss for hard examples (often noisy)
        if self.use_focal:
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.gamma
            ce_loss = focal_weight * ce_loss
        
        ce_loss = ce_loss.mean()
        total_loss = ce_loss
        
        # Uncertainty-based reweighting
        if uncertainty is not None:
            uncertainty_loss = torch.mean(uncertainty)
            # Reweight samples based on uncertainty
            sample_weights = 1.0 / (1.0 + uncertainty.squeeze())
            weighted_ce = torch.mean(sample_weights * ce_loss)
            total_loss = weighted_ce + self.alpha * uncertainty_loss
        
        return total_loss
    
    def _smooth_labels(self, targets):
        """Apply label smoothing"""
        confidence = 1.0 - self.smoothing
        smooth_targets = torch.full((targets.size(0), self.num_classes), 
                                   self.smoothing / (self.num_classes - 1), 
                                   device=targets.device)
        smooth_targets.scatter_(1, targets.unsqueeze(1), confidence)
        return smooth_targets


class MetaLearningGIN(NoiseRobustGINClassifier):
    """
    Meta-learning approach for noise adaptation
    Learns to adapt to different noise types and levels
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Meta-learning components
        self.meta_net = torch.nn.Sequential(
            torch.nn.Linear(self.output_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, self.output_dim),
            torch.nn.Sigmoid()
        )
        
    def forward(self, data, return_meta_weights=False, **kwargs):
        logits = super().forward(data, **kwargs)
        
        if return_meta_weights:
            # Generate sample-specific weights based on prediction confidence
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                meta_weights = self.meta_net(probs)
            return logits, meta_weights
        
        return logits


# Training utilities
def create_ppa_model(input_dim, config=None):
    """Factory function for PPA dataset"""
    default_config = {
        'hidden_dim': 128,
        'num_layers': 5,
        'dropout': 0.6,  # Higher dropout for noise
        'use_mixup': True,
        'use_label_smoothing': True,
        'smoothing_factor': 0.15,  # Aggressive smoothing for noise
    }
    
    if config:
        default_config.update(config)
    
    return NoiseRobustGINClassifier(input_dim=input_dim, **default_config)


def get_noise_robust_optimizer(model, lr=0.001, weight_decay=1e-4):
    """Optimizer configuration for noisy data"""
    return torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )


def get_scheduler(optimizer, total_epochs=200):
    """Learning rate scheduler that helps with noise"""
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )


# Example training loop adaptation for noise
class NoiseRobustTrainer:
    """Training utilities specifically for noisy PPA data"""
    
    @staticmethod
    def train_epoch(model, loader, optimizer, criterion, device, epoch):
        model.train()
        total_loss = 0
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Get predictions with uncertainty
            logits, uncertainty = model(batch, return_uncertainty=True, 
                                      apply_mixup=True, mixup_alpha=0.3)
            
            # Compute loss
            loss = criterion(logits, batch.y, uncertainty, epoch)
            loss.backward()
            
            # Gradient clipping for stability with noise
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    @staticmethod
    def evaluate(model, loader, device):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)
                pred = logits.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        
        return correct / total