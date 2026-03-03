```python
"""
Temporal Convolutional Network implementation for neural signal classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


class TemporalBlock(nn.Module):
    """
    Basic building block of TCN with dilated convolution and residual connection.
    """
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        """
        Initialize temporal block.
        
        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            dilation: Dilation factor
            padding: Padding size
            dropout: Dropout probability
        """
        super(TemporalBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size,
                     stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolutional layer
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size,
                     stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights using Xavier initialization."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal block."""
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove padding elements from the end of sequence."""
    
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for sequence modeling.
    """
    
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize TCN.
        
        Args:
            num_inputs: Number of input features
            num_channels: List of channels for each layer
            kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            )]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TCN."""
        return self.network(x)


class TCNClassifier(nn.Module):
    """
    Complete TCN-based classifier for neural signal classification.
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        num_channels: List[int] = [25, 25, 25, 25],
        kernel_size: int = 7,
        dropout: float = 0.3,
        sequence_length: int = 100
    ):
        """
        Initialize TCN classifier.
        
        Args:
            input_size: Number of input features
            num_classes: Number of output classes
            num_channels: Hidden layer sizes for TCN
            kernel_size: Convolution kernel size
            dropout: Dropout probability
            sequence_length: Expected sequence length
        """
        super(TCNClassifier, self).__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        
        # TCN backbone
        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size, dropout
        )
        
        # Global pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN classifier.
        
        Args:
            x: Input tensor of shape (batch_size, features) or (batch_size, sequence_length, features)
            
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Handle different input shapes
        if len(x.shape) == 2:
            # Single time step: (batch_size, features)
            # Reshape to (batch_size, features, 1)
            x = x.unsqueeze(-1)
        elif len(x.shape) == 3:
            # Sequence: (batch_size, sequence_length, features)
            # Transpose to (batch_size, features, sequence_length)
            x = x.transpose(1, 2)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Pass through TCN
        tcn_out = self.tcn(x)  # Shape: (batch_size, channels, sequence_length)
        
        # Global average pooling
        pooled = self.global_pool(tcn_out)  # Shape: (batch_size, channels, 1)
        pooled = pooled.squeeze(-1)  # Shape: (batch_size, channels)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


if __name__ == "__main__":
    # Test TCN implementation
    batch_size = 32
    input_size = 256  # Feature dimension
    num_classes = 5
    sequence_length = 100
    
    # Create model
    model = TCNClassifier(
        input_size=input_size,
        num_classes=num_classes,
        num_channels=[32, 32, 32, 32],
        kernel_size=3,
        dropout=0.2
    )
    
    # Test with single time step input
    x_single = torch.randn(batch_size, input_size)
    output_single = model(x_single)
    print(f"Single time step - Input shape: {x_single.shape}, Output shape: {output_single.shape}")
    
    # Test with sequence input
    x_seq = torch.randn(batch_size, sequence_length, input_size)
    output_seq = model(x_seq)
    print(f"Sequence - Input shape: {x_seq.shape}, Output shape: {output_seq.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
```
