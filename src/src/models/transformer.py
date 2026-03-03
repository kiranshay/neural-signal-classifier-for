```python
"""
Transformer model implementation for neural signal classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer input sequences.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(0), :]


class NeuralTransformer(nn.Module):
    """
    Transformer model for neural signal classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        use_cls_token: bool = True
    ):
        """
        Initialize Neural Transformer.
        
        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            use_cls_token: Whether to use CLS token for classification
        """
        super(NeuralTransformer, self).__init__()
        
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # CLS token (if used)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False  # (seq_len, batch, d_model)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            src_key_padding_mask: Padding mask for variable length sequences
            
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Handle different input shapes
        if len(x.shape) == 2:
            # Single time step: (batch_size, input_dim)
            # Add sequence dimension
            x = x.unsqueeze(1)  # Shape: (batch_size, 1, input_dim)
        
        batch_size, seq_len, input_dim = x.shape
        
        # Project input to model dimension
        x = self.input_projection(x)  # Shape: (batch_size, seq_len, d_model)
        x = x * math.sqrt(self.d_model)  # Scale embeddings
        
        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(1, batch_size, -1)  # Shape: (1, batch_size, d_model)
            x = torch.cat([cls_tokens, x.transpose(0, 1)], dim=0)  # Shape: (seq_len+1, batch_size, d_model)
            
            # Update padding mask if provided
            if src_key_padding_mask is not None:
                cls_mask = torch.zeros(batch_size, 1, device=x.device, dtype=torch.bool)
                src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
        else:
            x = x.transpose(0, 1)  # Shape: (seq_len, batch_size, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Extract features for classification
        if self.use_cls_token:
            # Use CLS token representation
            features = encoded[0]  # Shape: (batch_size, d_model)
        else:
            # Global average pooling over sequence
            if src_key_padding_mask is not None:
                # Mask out padded positions
                mask = ~src_key_padding_mask.transpose(0, 1).unsqueeze(-1)  # (seq_len, batch_size, 1)
                encoded = encoded * mask
                seq_lengths = mask.sum(dim=0).squeeze(-1)  # (batch_size,)
                features = encoded.sum(dim=0) / seq_lengths.unsqueeze(-1)
            else:
                features = encoded.mean(dim=0)  # Shape: (batch_size, d_model)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class MultiScaleTransformer(nn.Module):
    """
    Multi-scale transformer that processes signals at different temporal resolutions.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        scales: list = [1, 2, 4],
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize multi-scale transformer.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            scales: List of temporal scales (downsampling factors)
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers per scale
            dropout: Dropout probability
        """
        super(MultiScaleTransformer, self).__init__()
        
        self.scales = scales
        self.transformers = nn.ModuleList()
        
        # Create transformer for each scale
        for scale in scales:
            transformer = NeuralTransformer(
                input_dim=input_dim,
                num_classes=d_model,  # Output features, not classes
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                use_cls_token=True
            )
            self.transformers.append(transformer)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * len(scales), d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Class logits
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        scale_features = []
        
        for i, scale in enumerate(self.scales):
            # Downsample input for this scale
            if scale > 1:
                # Simple average pooling for downsampling
                seq_len = x.shape[1]
                new_len = seq_len // scale
                if new_len > 0:
                    x_scaled = F.avg_pool1d(
                        x.transpose(1, 2), 
                        kernel_size=scale, 
                        stride=scale
                    ).transpose(1, 2)
                else:
                    x_scaled = x[:, ::scale, :]  # Fallback to simple subsampling
            else:
                x_scaled = x
            
            # Process through transformer
            features = self.transformers[i](x_scaled)
            scale_features.append(features)
        
        # Concatenate features from all scales
        combined_features = torch.cat(scale_features, dim=-1)
        
        # Final classification
        logits = self.fusion(combined_features)
        
        return log