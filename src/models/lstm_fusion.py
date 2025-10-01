"""
LSTM model architectures for multi-timeframe trading.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

from ..utils.config import get_config, ModelConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)

class MultiTimeframeLSTM(nn.Module):
    """LSTM model for multi-timeframe trading signals."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super(MultiTimeframeLSTM, self).__init__()

        self.config = config or get_config().model

        # Architecture parameters
        self.input_size = self.config.input_size
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_layers
        self.dropout = self.config.dropout
        self.output_mode = self.config.output_mode
        self.num_classes = self.config.num_classes

        # Build model based on architecture
        if self.config.architecture == "single_lstm":
            self.model = self._build_single_lstm()
        elif self.config.architecture == "multi_lstm_fusion":
            self.model = self._build_multi_lstm_fusion()
        else:
            raise ValueError(f"Unknown architecture: {self.config.architecture}")

        # Initialize weights
        self._initialize_weights()

    def _build_single_lstm(self) -> nn.Module:
        """Build single LSTM architecture with concatenated features."""

        # LSTM layer
        lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )

        # Output layers
        if self.output_mode == "regression":
            output_layer = nn.Linear(self.hidden_size, 1)
        else:  # classification
            output_layer = nn.Linear(self.hidden_size, self.num_classes)

        model = nn.Sequential(
            lstm,
            nn.LayerNorm(self.hidden_size) if self.config.use_layer_norm else nn.Identity(),
            nn.Dropout(self.dropout),
            output_layer
        )

        return model

    def _build_multi_lstm_fusion(self) -> nn.Module:
        """Build multi-LSTM fusion architecture."""

        # Separate LSTMs for each timeframe
        # Note: In practice, you'd have separate feature extractors per timeframe
        # This is a simplified version assuming input features are already separated

        # Timeframe-specific LSTMs
        tf_lstms = nn.ModuleList([
            nn.LSTM(
                input_size=self.input_size // 4,  # Assuming 4 timeframes
                hidden_size=self.hidden_size // 2,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                batch_first=True
            ) for _ in range(4)  # 4 timeframes
        ])

        # Fusion strategy
        if self.config.fusion_strategy == "concatenate":
            fusion_input_size = self.hidden_size * 2  # 4 LSTMs * (hidden_size // 2) / 2
        elif self.config.fusion_strategy == "attention":
            fusion_input_size = self.hidden_size // 2
        elif self.config.fusion_strategy == "dense":
            fusion_input_size = self.hidden_size // 2
        else:
            raise ValueError(f"Unknown fusion strategy: {self.config.fusion_strategy}")

        # Fusion layers
        if self.config.fusion_strategy == "concatenate":
            fusion = nn.Sequential(
                nn.Linear(fusion_input_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
        elif self.config.fusion_strategy == "attention":
            fusion = MultiHeadAttention(
                embed_dim=self.hidden_size // 2,
                num_heads=4
            )
        elif self.config.fusion_strategy == "dense":
            fusion = nn.Sequential(
                nn.Linear(fusion_input_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )

        # Output layer
        if self.output_mode == "regression":
            output_layer = nn.Linear(self.hidden_size, 1)
        else:  # classification
            output_layer = nn.Linear(self.hidden_size, self.num_classes)

        model = nn.Sequential(
            tf_lstms,
            fusion,
            nn.LayerNorm(self.hidden_size) if self.config.use_layer_norm else nn.Identity(),
            nn.Dropout(self.dropout),
            output_layer
        )

        return model

    def _initialize_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""

        if self.config.architecture == "single_lstm":
            return self._forward_single_lstm(x)
        elif self.config.architecture == "multi_lstm_fusion":
            return self._forward_multi_lstm_fusion(x)
        else:
            raise ValueError(f"Unknown architecture: {self.config.architecture}")

    def _forward_single_lstm(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for single LSTM."""

        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.model[0](x)  # LSTM layer

        # Use last timestep output
        last_hidden = lstm_out[:, -1, :]

        # Apply layer norm and dropout
        if self.config.use_layer_norm:
            last_hidden = self.model[1](last_hidden)
        else:
            last_hidden = self.model[1](last_hidden)

        last_hidden = self.model[2](last_hidden)  # dropout

        # Final output
        output = self.model[3](last_hidden)

        return output

    def _forward_multi_lstm_fusion(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for multi-LSTM fusion."""

        # Split input into timeframe features
        # Assuming input is concatenated: [tf1_features, tf2_features, tf3_features, tf4_features]
        batch_size, seq_len, _ = x.shape
        tf_size = self.input_size // 4

        tf_inputs = [
            x[:, :, i*tf_size:(i+1)*tf_size] for i in range(4)
        ]

        # Process each timeframe through its LSTM
        tf_outputs = []
        for i, (lstm, tf_input) in enumerate(zip(self.model[0], tf_inputs)):
            lstm_out, _ = lstm(tf_input)
            # Use last timestep output
            last_hidden = lstm_out[:, -1, :]
            tf_outputs.append(last_hidden)

        # Stack timeframe outputs
        tf_stack = torch.stack(tf_outputs, dim=1)  # (batch_size, 4, hidden_size//2)

        # Apply fusion strategy
        if self.config.fusion_strategy == "concatenate":
            # Concatenate all timeframe representations
            fused = tf_stack.view(batch_size, -1)  # (batch_size, 4 * hidden_size//2)
            fused = self.model[1](fused)  # fusion layers

        elif self.config.fusion_strategy == "attention":
            # Use attention to fuse timeframe representations
            # Reshape for attention: (batch_size, 4, hidden_size//2)
            fused = self.model[1](tf_stack, tf_stack, tf_stack)  # self-attention
            fused = fused.mean(dim=1)  # (batch_size, hidden_size//2)

        elif self.config.fusion_strategy == "dense":
            # Use dense layers for fusion
            # Flatten and process through dense layers
            fused = tf_stack.view(batch_size, -1)
            fused = self.model[1](fused)

        # Apply layer norm and dropout
        if self.config.use_layer_norm:
            fused = self.model[2](fused)
        else:
            fused = self.model[2](fused)

        fused = self.model[3](fused)  # dropout

        # Final output
        output = self.model[4](fused)

        return output

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)

        # Linear transformations
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)

        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out(context)

        return output

class FocalLoss(nn.Module):
    """Focal Loss for classification tasks."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_loss_function(loss_name: str, **kwargs):
    """Get loss function by name."""

    if loss_name == "mse":
        return nn.MSELoss(**kwargs)
    elif loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == "focal_loss":
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

def create_model(config: Optional[ModelConfig] = None) -> MultiTimeframeLSTM:
    """Create LSTM model with given configuration."""

    return MultiTimeframeLSTM(config)

def load_model(model_path: str, config: Optional[ModelConfig] = None) -> MultiTimeframeLSTM:
    """Load trained model from file."""

    model = MultiTimeframeLSTM(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    logger.info(f"Loaded model from {model_path}")
    return model

def save_model(model: MultiTimeframeLSTM, model_path: str):
    """Save trained model to file."""

    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_summary(model: MultiTimeframeLSTM) -> str:
    """Get model architecture summary."""

    summary = f"""
Model Architecture: {model.config.architecture}
Input Size: {model.input_size}
Hidden Size: {model.hidden_size}
Num Layers: {model.num_layers}
Output Mode: {model.config.output_mode}
Parameters: {count_parameters(model):,}
"""

    return summary.strip()
