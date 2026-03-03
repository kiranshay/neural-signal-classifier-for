```python
"""
Neural signal dataset handling for ECoG motor intent classification.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any, List
import scipy.io
import pywt
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ECoGDataset(Dataset):
    """
    Dataset class for ECoG neural signals with motor intent labels.
    
    Handles data loading, preprocessing, and wavelet transform feature extraction.
    """
    
    def __init__(
        self,
        data_path: str,
        subject_id: str,
        window_size: int = 1000,
        overlap: float = 0.5,
        sampling_rate: int = 1000,
        wavelet_name: str = 'db4',
        wavelet_levels: int = 6,
        normalize: bool = True,
        filter_freq: Optional[Tuple[float, float]] = (1.0, 200.0)
    ):
        """
        Initialize ECoG dataset.
        
        Args:
            data_path: Path to data directory
            subject_id: Subject identifier (e.g., 'S1', 'S2', 'S3')
            window_size: Size of time windows in samples
            overlap: Overlap between consecutive windows (0-1)
            sampling_rate: Sampling rate in Hz
            wavelet_name: Wavelet for decomposition
            wavelet_levels: Number of wavelet decomposition levels
            normalize: Whether to normalize features
            filter_freq: Bandpass filter frequencies (low, high)
        """
        self.data_path = Path(data_path)
        self.subject_id = subject_id
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.wavelet_name = wavelet_name
        self.wavelet_levels = wavelet_levels
        self.normalize = normalize
        self.filter_freq = filter_freq
        
        self.scaler = StandardScaler() if normalize else None
        
        # Load and preprocess data
        self._load_data()
        self._preprocess_data()
        self._extract_features()
        
    def _load_data(self) -> None:
        """Load raw ECoG data from MATLAB files."""
        try:
            # Load BCI Competition IV Dataset 4 format
            data_file = self.data_path / f"{self.subject_id}_data.mat"
            
            if not data_file.exists():
                raise FileNotFoundError(f"Data file not found: {data_file}")
            
            mat_data = scipy.io.loadmat(str(data_file))
            
            # Extract neural signals and labels
            self.raw_signals = mat_data['X']  # Shape: (trials, channels, samples)
            self.labels = mat_data['y'].flatten()  # Shape: (trials,)
            
            # Get metadata
            self.n_trials, self.n_channels, self.n_samples = self.raw_signals.shape
            
            logger.info(f"Loaded {self.n_trials} trials, {self.n_channels} channels, "
                       f"{self.n_samples} samples per trial")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _preprocess_data(self) -> None:
        """Apply preprocessing steps to neural signals."""
        from scipy import signal
        
        logger.info("Preprocessing neural signals...")
        
        # Apply bandpass filter if specified
        if self.filter_freq is not None:
            low_freq, high_freq = self.filter_freq
            nyquist = self.sampling_rate / 2
            low_norm = low_freq / nyquist
            high_norm = high_freq / nyquist
            
            b, a = signal.butter(4, [low_norm, high_norm], btype='band')
            
            for trial_idx in range(self.n_trials):
                for ch_idx in range(self.n_channels):
                    self.raw_signals[trial_idx, ch_idx, :] = signal.filtfilt(
                        b, a, self.raw_signals[trial_idx, ch_idx, :]
                    )
        
        # Convert to float32 for efficiency
        self.raw_signals = self.raw_signals.astype(np.float32)
        
    def _extract_features(self) -> None:
        """Extract wavelet and time-domain features from signals."""
        logger.info("Extracting wavelet features...")
        
        features_list = []
        windowed_labels = []
        
        step_size = int(self.window_size * (1 - self.overlap))
        
        for trial_idx in range(self.n_trials):
            trial_data = self.raw_signals[trial_idx]  # Shape: (channels, samples)
            trial_label = self.labels[trial_idx]
            
            # Create sliding windows
            for start_idx in range(0, self.n_samples - self.window_size + 1, step_size):
                end_idx = start_idx + self.window_size
                window_data = trial_data[:, start_idx:end_idx]
                
                # Extract features for this window
                window_features = self._extract_window_features(window_data)
                features_list.append(window_features)
                windowed_labels.append(trial_label)
        
        self.features = np.array(features_list, dtype=np.float32)
        self.windowed_labels = np.array(windowed_labels)
        
        # Normalize features if requested
        if self.normalize:
            self.features = self.scaler.fit_transform(
                self.features.reshape(-1, self.features.shape[-1])
            ).reshape(self.features.shape)
        
        logger.info(f"Extracted features shape: {self.features.shape}")
    
    def _extract_window_features(self, window_data: np.ndarray) -> np.ndarray:
        """
        Extract features from a single time window.
        
        Args:
            window_data: Shape (channels, samples)
            
        Returns:
            Combined feature vector
        """
        features = []
        
        for ch_idx in range(window_data.shape[0]):
            channel_signal = window_data[ch_idx, :]
            
            # Wavelet decomposition
            coeffs = pywt.wavedec(channel_signal, self.wavelet_name, level=self.wavelet_levels)
            
            # Statistical features from each level
            for coeff in coeffs:
                features.extend([
                    np.mean(coeff),
                    np.std(coeff),
                    np.var(coeff),
                    np.max(coeff),
                    np.min(coeff)
                ])
            
            # Time domain features
            features.extend([
                np.mean(channel_signal),
                np.std(channel_signal),
                np.var(channel_signal),
                np.max(channel_signal),
                np.min(channel_signal),
                np.median(channel_signal)
            ])
        
        return np.array(features)
    
    def __len__(self) -> int:
        """Return number of windows in dataset."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label)
        """
        features = torch.from_numpy(self.features[idx]).float()
        label = torch.tensor(self.windowed_labels[idx], dtype=torch.long)
        
        return features, label
    
    def get_feature_dim(self) -> int:
        """Get the dimension of feature vectors."""
        return self.features.shape[-1]


def create_dataloaders(
    data_path: str,
    subjects: List[str],
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.2,
    num_workers: int = 4,
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """
    Create train/validation/test dataloaders for multiple subjects.
    
    Args:
        data_path: Path to data directory
        subjects: List of subject IDs
        batch_size: Batch size for dataloaders
        val_split: Fraction for validation set
        test_split: Fraction for test set
        num_workers: Number of workers for data loading
        **dataset_kwargs: Additional arguments for ECoGDataset
        
    Returns:
        Dictionary containing train/val/test dataloaders
    """
    from torch.utils.data import random_split, ConcatDataset
    
    # Load datasets for all subjects
    datasets = []
    for subject_id in subjects:
        dataset = ECoGDataset(data_path, subject_id, **dataset_kwargs)
        datasets.append(dataset)
    
    # Combine all subjects
    combined_dataset = ConcatDataset(datasets)
    
    # Calculate split sizes
    total_size = len(combined_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    # Random split
    train_dataset, val_dataset, test_dataset = random_split(
        combined_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        ),
        'val': DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        ),
        'test': DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    }
    
    logger.info(f"Created dataloaders - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    return dataloaders


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create dataset for subject S1
    dataset = ECoGDataset(
        data_path="data/bci_competition_iv",
        subject_id="S1",
        window_size=500,
        overlap=0.5,
        normalize=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Feature dimension: {dataset.get_feature_dim()}")
    
    # Test data loading
    features, label = dataset[0]
    print(f"Sample features shape: {features.shape}")
    print(f"Sample label: {label}")
```
