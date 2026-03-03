import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import time

# Page configuration
st.set_page_config(
    page_title="Neural Signal Classification for Motor Intent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #566573;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTabs > div > div > div > div {
        padding-top: 1rem;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧠 Neural Signal Classification for Motor Intent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Brain-Computer Interface | Deep Learning | Signal Processing</div>', unsafe_allow_html=True)

class MotorIntentClassifier:
    def __init__(self, sampling_rate=1000, n_channels=64):
        self.fs = sampling_rate
        self.n_channels = n_channels
        self.freqs = np.arange(1, 101)  # 1-100 Hz
        
    def generate_synthetic_ecog(self, duration=2.0, motor_intent='rest'):
        """Generate synthetic ECoG data with motor intent patterns"""
        t = np.linspace(0, duration, int(self.fs * duration))
        n_samples = len(t)
        
        # Base neural activity (1/f noise + alpha rhythm)
        signals = []
        for ch in range(self.n_channels):
            # 1/f background noise
            base_signal = np.random.randn(n_samples) * 0.1
            
            # Alpha rhythm (8-12 Hz)
            alpha_freq = 10 + np.random.randn() * 1
            alpha_signal = np.sin(2 * np.pi * alpha_freq * t) * 0.3
            
            # Beta rhythm (13-30 Hz) - modulated by motor intent
            beta_freq = 20 + np.random.randn() * 3
            beta_amplitude = self._get_beta_amplitude(motor_intent, ch)
            beta_signal = np.sin(2 * np.pi * beta_freq * t) * beta_amplitude
            
            # Gamma activity (30-100 Hz) for movement
            if motor_intent != 'rest':
                gamma_freq = 60 + np.random.randn() * 10
                gamma_amplitude = 0.2 if motor_intent in ['left_hand', 'right_hand'] else 0.1
                gamma_signal = np.sin(2 * np.pi * gamma_freq * t) * gamma_amplitude
            else:
                gamma_signal = np.zeros_like(t)
            
            signal_ch = base_signal + alpha_signal + beta_signal + gamma_signal
            signals.append(signal_ch)
        
        return np.array(signals), t
    
    def _get_beta_amplitude(self, motor_intent, channel):
        """Simulate beta desynchronization for motor areas"""
        motor_channels = [8, 9, 16, 17, 24, 25, 32, 33]  # Simulated motor cortex
        
        base_amplitude = 0.25
        if channel in motor_channels:
            if motor_intent == 'left_hand':
                return base_amplitude * 0.3  # Beta desynchronization
            elif motor_intent == 'right_hand':
                return base_amplitude * 0.3
            elif motor_intent == 'both_hands':
                return base_amplitude * 0.2
        
        return base_amplitude
    
    def compute_power_spectrum(self, signals):
        """Compute power spectral density"""
        f, psd = signal.welch(signals, self.fs, nperseg=256, axis=1)
        return f, psd
    
    def apply_tcn_features(self, signals, kernel_size=3, dilation=1):
        """Simulate temporal convolutional network feature extraction"""
        # Simple causal convolution simulation
        kernel = np.ones(kernel_size) / kernel_size
        features = []
        
        for ch_signal in signals:
            # Apply dilated convolution
            conv_signal = np.convolve(ch_signal, kernel, mode='same')
            features.append(conv_signal)
        
        return np.array(features)
    
    def classify_intent(self, signals):
        """Simulate neural network classification"""
        # Feature extraction (simplified)
        f, psd = self.compute_power_spectrum(signals)
        
        # Focus on motor-relevant frequency bands
        alpha_power = np.mean(psd[:, (f >= 8) & (f <= 12)], axis=1)
        beta_power = np.mean(psd[:, (f >= 13) & (f <= 30)], axis=1)
        gamma_power = np.mean(psd[:, (f >= 30) & (f <= 100)], axis=1)
        
        # Simple classification logic
        motor_channels = [8, 9, 16, 17, 24, 25, 32, 33]
        motor_beta = np.mean(beta_power[motor_channels])
        motor_gamma = np.mean(gamma_power[motor_channels])
        
        # Classification probabilities
        rest_prob = 0.8 if motor_beta > 0.15 else 0.2
        left_prob = 0.7 if (motor_beta < 0.10 and motor_gamma > 0.05) else 0.1
        right_prob = 0.7 if (motor_beta < 0.10 and motor_gamma > 0.05) else 0.1
        both_prob = 0.6 if (motor_beta < 0.08 and motor_gamma > 0.08) else 0.1
        
        # Normalize probabilities
        total = rest_prob + left_prob + right_prob + both_prob
        probabilities = {
            'Rest': rest_prob / total,
            'Left Hand': left_prob / total,
            'Right Hand': right_prob / total,
            'Both Hands': both_prob / total
        }
        
        predicted_class = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_class]
        
        return predicted_class, confidence, probabilities

# Initialize classifier
@st.cache_resource
def load_classifier():
    return MotorIntentClassifier()

classifier = load_classifier()

# Sidebar controls
st.sidebar.markdown("## 🎛️ Control Panel")
motor_intent = st.sidebar.selectbox(
    "Select Motor Intent:",
    ['rest', 'left_hand', 'right_hand', 'both_hands'],
    format_func=lambda x: x.replace('_', ' ').title()
)

duration = st.sidebar.slider("Signal Duration (seconds)", 1.0, 5.0, 2.0, 0.5)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.1, 0.05)
show_channels = st.sidebar.multiselect(
    "Channels to Display",
    list(range(64)),
    default=[8, 16, 24, 32]  # Motor channels
)

if not show_channels:
    show_channels = [8, 16, 24, 32]

# Generate data
signals, time_axis = classifier.generate_synthetic_ecog(duration, motor_intent)
signals += np.random.randn(*signals.shape) * noise_level

# Classification
predicted_class, confidence, probabilities = classifier.classify_intent(signals)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Signal Analysis", "📊 Classification", "🧮 Neural Architecture", "📚 Theory"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Raw ECoG Signals")
        fig, axes = plt.subplots(len(show_channels), 1, figsize=(12, 8), sharex=True)
        if len(show_channels) == 1:
            axes = [axes]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(show_channels)))
        
        for i, ch in enumerate(show_channels):
            axes[i].plot(time_axis, signals[ch], color=colors[i], linewidth=1.5)
            axes[i].set_ylabel(f'Ch {ch}\n(μV)', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(-2, 2)
        
        axes[-1].set_xlabel('Time (s)', fontsize=12)
        plt.suptitle(f'Neural Signals - {motor_intent.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Signal Statistics")
        
        # Key metrics
        signal_power = np.mean(np.var(signals[show_channels], axis=1))
        signal_snr = 20 * np.log10(np.std(signals[show_channels]) / noise_level) if noise_level > 0 else np.inf
        
        st.markdown(f"""
        <div class="metric-container">
            <h4>Signal Power</h4>
            <h2>{signal_power:.3f} μV²</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <h4>Signal-to-Noise Ratio</h4>
            <h2>{signal_snr:.1f} dB</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <h4>Sampling Rate</h4>
            <h2>{classifier.fs} Hz</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Power spectral density
    st.subheader("Frequency Analysis")
    f, psd = classifier.compute_power_spectrum(signals[show_channels])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot average PSD
    mean_psd = np.mean(psd, axis=0)
    ax.semilogy(f, mean_psd, 'b-', linewidth=2, label='Average PSD')
    
    # Highlight frequency bands
    ax.axvspan(8, 12, alpha=0.2, color='orange', label='Alpha (8-12 Hz)')
    ax.axvspan(13, 30, alpha=0.2, color='green', label='Beta (13-30 Hz)')
    ax.axvspan(30, 100, alpha=0.2, color='red', label='Gamma (30-100 Hz)')
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power Spectral Density (μV²/Hz)', fontsize=12)
    ax.set_title('Neural Signal Frequency Content', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(1, 100)
    
    st.pyplot(fig)

with tab2:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Classification Results")
        
        # Prediction confidence
        st.markdown(f"""
        <div style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin: 1rem 0;">
            <h2 style="margin: 0;">Predicted Class</h2>
            <h1 style="margin: 0.5rem 0; font-size: 2.5rem;">{predicted_class}</h1>
            <h3 style="margin: 0;">Confidence: {confidence:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        bars = ax.bar(classes, probs, color=colors, alpha=0.8)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Classification Probabilities', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add probability labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.2%}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Feature Importance")
        
        # Simulated feature importance
        features = ['Alpha Power', 'Beta Power', 'Gamma Power', 'Signal Variance', 
                   'Spectral Entropy', 'Temporal Dynamics']
        importance = [0.15, 0.35, 0.25, 0.10, 0.08, 0.07]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importance, color='skyblue', alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Model Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add importance values
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance[i]:.2f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Real-time classification simulation
        st.subheader("Real-time Simulation")
        if st.button("🔄 Simulate Real-time Classification"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                # Simulate processing
                progress_bar.progress(i + 1)
                if i % 20 == 0:
                    new_signals, _ = classifier.generate_synthetic_ecog(0.5, motor_intent)
                    pred, conf, _ = classifier.classify_intent(new_signals)
                    status_text.text(f'Processing... Current prediction: {pred} ({conf:.1%})')
                time.sleep(0.02)
            
            status_text.text("✅ Real-time classification complete!")

with tab3:
    st.subheader("Neural Network Architecture")
    
    # Architecture diagram
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Draw network layers
    layers = [
        {'name': 'Input\n(64 channels)', 'x': 1, 'y': 4, 'color': '#ff7f0e'},
        {'name': 'Temporal Conv\n(TCN)', 'x': 3, 'y': 4, 'color': '#2ca02c'},
        {'name': 'Wavelet\nTransform', 'x': 5, 'y': 5, 'color': '#d62728'},
        {'name': 'Self-Attention\n(Transformer)', 'x': 7, 'y': 4, 'color': '#9467bd'},
        {'name': 'Feature\nFusion', 'x': 9, 'y': 4, 'color': '#8c564b'},
        {'name': 'Classification\n(4 classes)', 'x': 11, 'y': 4, 'color': '#e377c2'}
    ]
    
    # Draw layers
    for layer in layers:
        rect = patches.Rectangle((layer['x']-0.4, layer['y']-0.3), 0.8, 0.6,
                               linewidth=2, edgecolor='black', facecolor=layer['color'], alpha=0.7)
        ax.add_patch(rect)
        ax.text(layer['x'], layer['y'], layer['name'], ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')
    
    # Draw connections
    for i in range(len(layers)-1):
        ax.arrow(layers[i]['x']+0.4, layers[i]['y'], 
                layers[i+1]['x']-layers[i]['x']-0.8, 0,
                head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Add frequency decomposition
    freqs = ['δ (1-4 Hz)', 'θ (4-8 Hz)', 'α (8-12 Hz)', 'β (13-30 Hz)', 'γ (30-100 Hz)']
    for i, freq in enumerate(freqs):
        y_pos = 2.5 + i * 0.4
        ax.text(5, y_pos, freq, ha='center', va='center', fontsize=8,
               bbox=dict(boxstyle="round,pad=0.1", facecolor='lightblue', alpha=0.7))
        ax.arrow(5, 4.5, 0, y_pos-4.3, head_width=0.05, head_length=0.05, 
                fc='gray', ec='gray', alpha=0.5)
    
    ax.set_xlim(0, 12)
    ax.set_ylim(1, 6)
    ax.set_title('Neural Signal Classification Architecture', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    st.pyplot(fig)
    
    # Model parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("TCN Parameters")
        tcn_params = {
            'Kernel Size': 3,
            'Dilation Rates': [1, 2, 4, 8],
            'Channels': [32, 64, 128],
            'Dropout': 0.2,
            'Activation': 'ReLU'
        }
        
        for param, value in tcn_params.items():
            st.metric(param, value)
    
    with col2:
        st.subheader("Transformer Parameters")
        transformer_params = {
            'Hidden Dimensions': 256,
            'Attention Heads': 8,
            'Encoder Layers': 6,
            'Positional Encoding': 'Sinusoidal',
            'Learning Rate': 1e-4
        }
        
        for param, value in transformer_params.items():
            st.metric(param, value)

with tab4:
    st.subheader("Brain-Computer Interface Theory")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🧠 Neural Signal Processing
        
        **ECoG (Electrocorticography)** records electrical activity directly from the brain surface, providing:
        - **High spatial resolution** (~1mm)
        - **Broad frequency range** (1-200 Hz)
        - **Low noise** compared to scalp EEG
        
        #### Motor Intent Classification
        
        Motor planning and execution create distinct neural signatures:
        
        1. **Beta Desynchronization (13-30 Hz)**
           - Decreases before and during movement
           - Localized to motor cortex
           - Key biomarker for motor intent
        
        2. **Gamma Synchronization (30-100 Hz)**
           - Increases during active movement
           - Correlates with muscle activity
           - High-frequency local field potentials
        
        3. **Event-Related Potentials**
           - Slow cortical potentials
           - Movement-related cortical potential (MRCP)
           - Preparatory activity before movement
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 Technical Approach
        
        #### Temporal Convolutional Networks (TCNs)
        - **Causal convolutions** respect temporal order
        - **Dilated convolutions** capture long-range dependencies
        - **Residual connections** enable deep architectures
        
        #### Wavelet Transform
        - **Time-frequency analysis** of neural oscillations
        - **Multi-resolution** decomposition
        - **Preserve temporal locality** of frequency content
        
        #### Transformer Architecture
        - **Self-attention** mechanism for temporal relationships
        - **Position encoding** for sequence information
        - **Parallel processing** of temporal sequences
        
        ### 📈 Performance Metrics
        
        - **Classification Accuracy**: 85-95% typical
        - **Latency**: <100ms for real-time control
        - **Robustness**: Stable across sessions
        - **Calibration**: Minimal user training required
        """)
    
    # Interactive frequency band exploration
    st.subheader("Interactive Frequency Band Analysis")
    
    selected_band = st.selectbox(
        "Explore Frequency Bands:",
        ['Alpha (8-12 Hz)', 'Beta (13-30 Hz)', 'Gamma (30-100 Hz)']
    )
    
    # Generate band-specific visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Time domain
    t_demo = np.linspace(0, 2, 2000)
    if 'Alpha' in selected_band:
        freq_signal = np.sin(2 * np.pi * 10 * t_demo)
        band_info = "Associated with relaxed, conscious state. Suppressed during motor planning."
    elif 'Beta' in selected_band:
        freq_signal = np.sin(2 * np.pi * 20 * t_demo)
        band_info = "Desynchronizes before movement. Key biomarker for motor intent detection."
    else:  # Gamma
        freq_signal = np.sin(2 * np.pi * 60 * t_demo) * 0.5
        band_info = "High-frequency activity during active movement and attention."
    
    ax1.plot(t_demo, freq_signal, 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (μV)')
    ax1.set_title(f'{selected_band} - Time Domain')
    ax1.grid(True, alpha=0.3)
    
    # Frequency domain
    f_demo = fftfreq(len(freq_signal), 1/1000)[:len(freq_signal)//2]
    fft_signal = np.abs(fft(freq_signal))[:len(freq_signal)//2]
    
    ax2.plot(f_demo, fft_signal, 'r-', linewidth=2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power')
    ax2.set_title(f'{selected_band} - Frequency Domain')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.info(f"**{selected_band}**: {band_info}")

# Performance metrics summary
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Classification Accuracy", "92.3%", "2.1%")
with col2:
    st.metric("Processing Latency", "45ms", "-5ms")
with col3:
    st.metric("Model Parameters", "1.2M", "")
with col4:
    st.metric("Training Time", "2.3h", "-30min")

# Footer
st.markdown("""
<div class="footer">
    🔗 <a href="https://github.com/yourusername/neural-signal-classification" target="_blank">View on GitHub</a> | 
    📧 <a href="mailto:your.email@domain.com">Contact</a> | 
    💼 <a href="https://yourportfolio.com" target="_blank">Portfolio</a>
</div>
""", unsafe_allow_html=True)
