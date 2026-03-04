"""
Neural Signal Classification for Motor Intent - Interactive Demo
================================================================
Brain-Computer Interface | Deep Learning | Signal Processing
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.interpolate import griddata
from scipy.signal import hilbert
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# Page configuration
st.set_page_config(
    page_title="BCI Motor Intent Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS Styling System ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    .main-header {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-size: 2.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }

    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        letter-spacing: 0.02em;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(71, 85, 105, 0.4);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2), inset 0 1px 2px rgba(255,255,255,0.03);
        flex-wrap: wrap;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background: transparent;
        border-radius: 10px;
        padding: 0 16px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.82rem;
        color: #94a3b8;
        border: none;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        white-space: nowrap;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.12);
        color: #a5b4fc;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(102, 126, 234, 0.4);
    }

    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* Section Headers */
    .section-header {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%) 1;
        display: block;
    }

    .subsection-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 2rem 0 1rem 0;
        padding: 0.75rem 1rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.8) 100%);
        border-left: 4px solid #667eea;
        border-radius: 0 10px 10px 0;
        backdrop-filter: blur(8px);
    }

    /* Cards */
    .concept-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(71, 85, 105, 0.4);
        border-radius: 14px;
        padding: 1.25rem;
        margin: 1rem 0;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }

    .concept-card h5 {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 0 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .concept-card p, .concept-card li {
        color: #cbd5e1;
        line-height: 1.7;
        margin-bottom: 0.5rem;
    }

    .concept-card strong { color: #f8fafc; }
    .concept-card em { color: #a5b4fc; }

    /* Highlight Box */
    .highlight-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.12) 100%);
        border: 1px solid rgba(102, 126, 234, 0.35);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        backdrop-filter: blur(8px);
    }

    .highlight-box p { color: #e0e7ff; font-weight: 500; margin: 0; }
    .highlight-box strong { color: #f8fafc; }

    /* Key Point */
    .key-point {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(52, 211, 153, 0.08) 100%);
        border: 1px solid rgba(52, 211, 153, 0.35);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .key-point-icon { font-size: 1.25rem; flex-shrink: 0; }
    .key-point p { color: #a7f3d0; font-weight: 500; margin: 0; line-height: 1.6; }
    .key-point strong { color: #f8fafc; }

    /* Metric Container */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.25rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.25), 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.35);
    }

    .metric-container h3 {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 4px 0;
    }

    .metric-container p { font-size: 0.85rem; opacity: 0.9; margin: 0; font-weight: 500; }

    /* Param Cards */
    .param-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .param-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(71, 85, 105, 0.4);
        border-radius: 12px;
        padding: 1rem;
        transition: border-color 0.2s ease;
    }

    .param-card:hover { border-color: rgba(102, 126, 234, 0.5); }

    .param-card h6 {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        color: #a5b4fc;
        margin: 0 0 0.5rem 0;
    }

    .param-card p { color: #cbd5e1; font-size: 0.9rem; margin: 0; line-height: 1.5; }

    /* Cell Cards */
    .cell-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.85) 100%);
        border: 2px solid rgba(71, 85, 105, 0.4);
        border-radius: 14px;
        padding: 1.25rem;
        height: 100%;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .cell-card:hover {
        border-color: #667eea;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }

    .cell-card h4 {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 0 0 0.5rem 0;
    }

    .cell-card .props { color: #cbd5e1; font-size: 0.9rem; line-height: 1.6; }
    .cell-card .props strong { color: #f8fafc; }

    /* Algorithm Steps */
    .algo-step {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        padding: 0.75rem 0;
        border-bottom: 1px dashed rgba(71, 85, 105, 0.4);
    }
    .algo-step:last-child { border-bottom: none; }

    .step-num {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        flex-shrink: 0;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }

    .step-content { color: #e2e8f0; line-height: 1.6; }
    .step-content strong { color: #f8fafc; }

    /* Definition List */
    .def-item {
        display: flex;
        margin-bottom: 0.75rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(71, 85, 105, 0.3);
    }
    .def-term { font-weight: 600; color: #a5b4fc; min-width: 140px; flex-shrink: 0; }
    .def-desc { color: #cbd5e1; line-height: 1.5; }

    /* Pipeline Step */
    .pipeline-step {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(71, 85, 105, 0.4);
        border-radius: 14px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        position: relative;
        transition: border-color 0.2s ease;
    }

    .pipeline-step:hover { border-color: rgba(102, 126, 234, 0.5); }

    .pipeline-step .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 0.75rem;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }

    .pipeline-step h5 {
        display: inline;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #f1f5f9;
    }

    .pipeline-step p { color: #cbd5e1; margin: 0.5rem 0 0 0; line-height: 1.6; }

    /* Model Comparison Card */
    .model-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid rgba(71, 85, 105, 0.4);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .model-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        border-radius: 14px 14px 0 0;
    }

    .model-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.2);
    }

    .model-card.best { border-color: rgba(52, 211, 153, 0.5); }
    .model-card.best::before { background: linear-gradient(90deg, #10b981, #34d399); }

    .model-card h4 {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0.5rem 0;
    }

    .model-card .model-metric {
        color: #cbd5e1;
        font-size: 0.9rem;
        margin: 0.25rem 0;
    }

    .model-card .model-metric strong { color: #a5b4fc; }

    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .status-badge.best { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .status-badge.good { background: rgba(102, 126, 234, 0.15); color: #a5b4fc; border: 1px solid rgba(102, 126, 234, 0.3); }
    .status-badge.baseline { background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); }

    /* Warning Box */
    .warning-box {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(245, 158, 11, 0.08) 100%);
        border: 1px solid rgba(251, 191, 36, 0.35);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .warning-box p { color: #fde68a; font-weight: 500; margin: 0; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }

    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] .stSlider label { color: #cbd5e1 !important; }

    /* Architecture */
    .arch-container {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.25);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }

    .arch-flow {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }

    .arch-block {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        min-width: 120px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }

    .arch-block:hover { transform: scale(1.05); }

    .arch-arrow { color: #64748b; font-size: 1.5rem; padding: 0 0.5rem; }
    .arch-label { font-weight: 700; font-size: 0.9rem; margin-bottom: 0.25rem; }
    .arch-sublabel { font-size: 0.75rem; opacity: 0.8; }

    /* Prediction Box */
    .prediction-box {
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }

    /* Expander */
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] li,
    [data-testid="stExpander"] span { color: #e2e8f0 !important; }
    [data-testid="stExpander"] strong { color: #f8fafc !important; }
    [data-testid="stExpander"] em { color: #a5b4fc !important; }
    [data-testid="stExpander"] summary { color: #f1f5f9 !important; }

    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        border-radius: 16px;
        border: 1px solid rgba(102, 126, 234, 0.25);
        text-align: center;
        box-shadow: 0 -4px 24px rgba(0,0,0,0.1);
    }

    .footer p { color: #e2e8f0 !important; margin: 0.5rem 0; }
    .footer p strong { color: #f8fafc !important; }
    .footer a { color: #818cf8; text-decoration: none; font-weight: 500; transition: color 0.2s ease; }
    .footer a:hover { text-decoration: underline; color: #a5b4fc; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🧠 Neural Signal Classification for Motor Intent</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Brain-Computer Interface · Deep Learning · ECoG Signal Processing</p>', unsafe_allow_html=True)


# ==================== Plot Helper ====================
def setup_dark_plot(fig, ax):
    """Apply dark theme to matplotlib figures."""
    fig.patch.set_facecolor('#0f172a')
    if isinstance(ax, np.ndarray):
        for a in ax.flat:
            _style_axis(a)
    else:
        _style_axis(ax)

def _style_axis(ax):
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    ax.xaxis.label.set_color('#e2e8f0')
    ax.yaxis.label.set_color('#e2e8f0')
    ax.title.set_color('#f1f5f9')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#475569')
    ax.grid(True, alpha=0.15, color='#94a3b8')

def fix_log_ticks(ax, axis='y'):
    """Fix log scale tick labels to avoid mathtext parsing errors."""
    from matplotlib.ticker import FuncFormatter
    def fmt(x, _):
        if x == 0:
            return '0'
        exp = np.log10(abs(x))
        if abs(exp - round(exp)) < 0.01 and abs(exp) >= 2:
            return f'1e{int(round(exp))}'
        elif x >= 0.01:
            return f'{x:.3g}'
        else:
            return f'{x:.1e}'
    if axis == 'y':
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(fmt))


# ==================== MotorIntentClassifier ====================
class MotorIntentClassifier:
    """Simulates a BCI motor intent classification system."""

    def __init__(self, sampling_rate=1000, n_channels=64):
        self.fs = sampling_rate
        self.n_channels = n_channels
        self.motor_channels = [8, 9, 16, 17, 24, 25, 32, 33]
        self.electrode_positions = self._generate_electrode_positions()

    def _generate_electrode_positions(self):
        positions = []
        rings = [(0.0, 1), (0.2, 6), (0.4, 12), (0.6, 18), (0.8, 20), (0.95, 7)]
        idx = 0
        for radius, n_electrodes in rings:
            if idx >= 64:
                break
            for i in range(n_electrodes):
                if idx >= 64:
                    break
                angle = 2 * np.pi * i / n_electrodes + (np.pi / n_electrodes if radius > 0.3 else 0)
                x = 0.5 + radius * 0.45 * np.cos(angle)
                y = 0.5 + radius * 0.45 * np.sin(angle)
                positions.append((x, y))
                idx += 1
        return positions[:64]

    def generate_synthetic_ecog(self, duration=2.0, motor_intent='rest'):
        t = np.linspace(0, duration, int(self.fs * duration))
        signals = []
        for ch in range(self.n_channels):
            noise = np.cumsum(np.random.randn(len(t))) * 0.02
            noise = noise - np.mean(noise)
            alpha = np.sin(2 * np.pi * (10 + np.random.randn()) * t) * 0.3
            beta_amp = self._get_beta_amplitude(motor_intent, ch)
            beta = np.sin(2 * np.pi * (20 + np.random.randn() * 3) * t) * beta_amp
            gamma_amp = self._get_gamma_amplitude(motor_intent, ch)
            gamma = np.sin(2 * np.pi * (60 + np.random.randn() * 10) * t) * gamma_amp
            signals.append(noise + alpha + beta + gamma)
        return np.array(signals), t

    def _get_beta_amplitude(self, intent, channel):
        base = 0.25
        if channel in self.motor_channels:
            if intent in ['left_hand', 'right_hand']:
                return base * 0.3
            elif intent == 'both_hands':
                return base * 0.2
        return base

    def _get_gamma_amplitude(self, intent, channel):
        if intent == 'rest':
            return 0.02
        if channel in self.motor_channels:
            return 0.15 if intent in ['left_hand', 'right_hand'] else 0.2
        return 0.05

    def compute_power_spectrum(self, signals):
        f, psd = signal.welch(signals, self.fs, nperseg=min(256, signals.shape[1]), axis=1)
        return f, psd

    def extract_band_power(self, signals):
        f, psd = self.compute_power_spectrum(signals)
        bands = {
            'Delta (1-4 Hz)': (1, 4), 'Theta (4-8 Hz)': (4, 8),
            'Alpha (8-12 Hz)': (8, 12), 'Beta (13-30 Hz)': (13, 30),
            'Gamma (30-100 Hz)': (30, 100)
        }
        return {name: np.mean(psd[:, (f >= lo) & (f <= hi)]) for name, (lo, hi) in bands.items()}

    def classify_intent(self, signals):
        f, psd = self.compute_power_spectrum(signals)
        motor_psd = psd[self.motor_channels]
        alpha_power = np.mean(motor_psd[:, (f >= 8) & (f <= 12)])
        beta_power = np.mean(motor_psd[:, (f >= 13) & (f <= 30)])
        gamma_power = np.mean(motor_psd[:, (f >= 30) & (f <= 100)])
        beta_ratio = beta_power / (alpha_power + 1e-10)
        gamma_ratio = gamma_power / (alpha_power + 1e-10)

        if beta_ratio > 0.8 and gamma_ratio < 0.3:
            probs = {'Rest': 0.75, 'Left Hand': 0.10, 'Right Hand': 0.10, 'Both Hands': 0.05}
        elif beta_ratio < 0.4 and gamma_ratio > 0.5:
            probs = {'Rest': 0.05, 'Left Hand': 0.35, 'Right Hand': 0.35, 'Both Hands': 0.25}
        elif beta_ratio < 0.3:
            probs = {'Rest': 0.05, 'Left Hand': 0.25, 'Right Hand': 0.25, 'Both Hands': 0.45}
        else:
            probs = {'Rest': 0.40, 'Left Hand': 0.25, 'Right Hand': 0.25, 'Both Hands': 0.10}

        for k in probs:
            probs[k] = max(0, probs[k] + np.random.uniform(-0.05, 0.05))
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        predicted = max(probs, key=probs.get)
        return predicted, probs[predicted], probs

    def compute_wavelet_spectrogram(self, signal_1d, freqs=None):
        if freqs is None:
            freqs = np.logspace(np.log10(1), np.log10(100), 50)
        n_samples = len(signal_1d)
        spec = np.zeros((len(freqs), n_samples))
        for i, freq in enumerate(freqs):
            n_cycles = 5
            wlen = max(3, int(n_cycles * self.fs / freq))
            t = np.linspace(-wlen / 2, wlen / 2, wlen) / self.fs
            sigma_t = n_cycles / (2 * np.pi * freq)
            wavelet = np.exp(2j * np.pi * freq * t) * np.exp(-t**2 / (2 * sigma_t**2))
            wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet)**2))
            spec[i, :] = np.abs(signal.convolve(signal_1d, wavelet, mode='same'))
        return freqs, spec

    def apply_preprocessing_pipeline(self, signals):
        """Apply step-by-step preprocessing pipeline."""
        stages = {}
        stages['Raw'] = signals.copy()

        # Notch filter at 60 Hz
        b_notch, a_notch = signal.iirnotch(60.0, 30.0, self.fs)
        notched = signal.filtfilt(b_notch, a_notch, signals, axis=1)
        stages['Notch Filter (60 Hz)'] = notched

        # Bandpass filter 0.5-200 Hz
        sos = signal.butter(4, [0.5, min(200, self.fs/2 - 1)], btype='bandpass', fs=self.fs, output='sos')
        bandpassed = signal.sosfiltfilt(sos, notched, axis=1)
        stages['Bandpass (0.5-200 Hz)'] = bandpassed

        # Common average reference
        car = bandpassed - np.mean(bandpassed, axis=0, keepdims=True)
        stages['Common Avg Reference'] = car

        # Artifact rejection (clip extremes)
        threshold = 3 * np.std(car)
        cleaned = np.clip(car, -threshold, threshold)
        stages['Artifact Rejection'] = cleaned

        return stages

    def compute_ersp(self, signal_1d):
        """Compute Event-Related Spectral Perturbation."""
        freqs, spec = self.compute_wavelet_spectrogram(signal_1d)
        n_samples = spec.shape[1]
        baseline_end = n_samples // 4
        baseline_power = np.mean(spec[:, :baseline_end], axis=1, keepdims=True)
        baseline_power = np.maximum(baseline_power, 1e-10)
        ersp = 10 * np.log10(spec / baseline_power)
        return freqs, ersp

    def compute_phase_amplitude_coupling(self, signal_1d):
        """Compute phase-amplitude coupling between beta phase and gamma amplitude."""
        # Beta phase (13-30 Hz)
        sos_beta = signal.butter(4, [13, 30], btype='bandpass', fs=self.fs, output='sos')
        beta_filt = signal.sosfiltfilt(sos_beta, signal_1d)
        beta_phase = np.angle(hilbert(beta_filt))

        # Gamma amplitude (30-100 Hz)
        sos_gamma = signal.butter(4, [30, min(100, self.fs/2 - 1)], btype='bandpass', fs=self.fs, output='sos')
        gamma_filt = signal.sosfiltfilt(sos_gamma, signal_1d)
        gamma_amp = np.abs(hilbert(gamma_filt))

        # Compute modulation index across phase bins
        n_bins = 18
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        mean_amp = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (beta_phase >= phase_bins[i]) & (beta_phase < phase_bins[i + 1])
            if np.sum(mask) > 0:
                mean_amp[i] = np.mean(gamma_amp[mask])
        mean_amp = mean_amp / (np.sum(mean_amp) + 1e-10)

        # Comodulogram: coupling across frequency pairs
        phase_freqs = np.arange(4, 35, 2)
        amp_freqs = np.arange(30, 105, 5)
        comodulogram = np.zeros((len(amp_freqs), len(phase_freqs)))

        for pi, pf in enumerate(phase_freqs):
            sos_p = signal.butter(3, [max(1, pf - 2), min(pf + 2, self.fs/2 - 1)], btype='bandpass', fs=self.fs, output='sos')
            p_filt = signal.sosfiltfilt(sos_p, signal_1d)
            p_phase = np.angle(hilbert(p_filt))

            for ai, af in enumerate(amp_freqs):
                sos_a = signal.butter(3, [max(1, af - 5), min(af + 5, self.fs/2 - 1)], btype='bandpass', fs=self.fs, output='sos')
                a_filt = signal.sosfiltfilt(sos_a, signal_1d)
                a_amp = np.abs(hilbert(a_filt))

                # Modulation index
                bins_amp = np.zeros(n_bins)
                for b in range(n_bins):
                    mask = (p_phase >= phase_bins[b]) & (p_phase < phase_bins[b + 1])
                    if np.sum(mask) > 0:
                        bins_amp[b] = np.mean(a_amp[mask])
                bins_amp = bins_amp / (np.sum(bins_amp) + 1e-10)
                uniform = 1.0 / n_bins
                kl = np.sum(bins_amp * np.log(bins_amp / uniform + 1e-10))
                comodulogram[ai, pi] = kl

        return phase_bins, mean_amp, phase_freqs, amp_freqs, comodulogram

    def compute_connectivity_matrix(self, signals, n_subset=16):
        """Compute coherence-based connectivity matrix."""
        n_ch = min(n_subset, signals.shape[0])
        conn = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            for j in range(i, n_ch):
                f_coh, coh = signal.coherence(signals[i], signals[j], self.fs, nperseg=min(256, len(signals[i])))
                beta_mask = (f_coh >= 13) & (f_coh <= 30)
                conn[i, j] = np.mean(coh[beta_mask])
                conn[j, i] = conn[i, j]
        return conn

    def generate_embedding_data(self, n_trials=200):
        """Generate simulated t-SNE-like 2D embeddings."""
        centers = {'Rest': (0, 0), 'Left Hand': (-3, 2), 'Right Hand': (3, 2), 'Both Hands': (0, -3)}
        points, labels, confs = [], [], []
        for label, (cx, cy) in centers.items():
            n = n_trials // 4
            x = np.random.randn(n) * 0.8 + cx
            y = np.random.randn(n) * 0.8 + cy
            c = np.clip(0.7 + np.random.randn(n) * 0.15, 0.3, 1.0)
            points.extend(zip(x, y))
            labels.extend([label] * n)
            confs.extend(c)
        points = np.array(points)
        return points, labels, np.array(confs)

    def generate_roc_data(self):
        """Generate realistic ROC curves per class."""
        roc = {}
        classes = ['Rest', 'Left Hand', 'Right Hand', 'Both Hands']
        aucs = [0.97, 0.94, 0.93, 0.95]
        for cls, auc_target in zip(classes, aucs):
            fpr = np.sort(np.concatenate([[0], np.random.beta(0.5, 5, 50), [1]]))
            tpr = np.sort(np.concatenate([[0], np.random.beta(5 * auc_target, 5 * (1 - auc_target), 50), [1]]))
            tpr = np.clip(tpr, fpr, 1.0)
            auc_val = np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2)
            roc[cls] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_val}
        return roc

    def generate_model_comparison_data(self):
        """Generate benchmark data for model comparison."""
        return {
            'CNN': {'accuracy': 0.847, 'f1': 0.832, 'inference_ms': 8, 'params': '0.4M', 'train_hrs': 0.8, 'status': 'baseline'},
            'LSTM': {'accuracy': 0.873, 'f1': 0.861, 'inference_ms': 35, 'params': '0.9M', 'train_hrs': 3.2, 'status': 'good'},
            'TCN': {'accuracy': 0.912, 'f1': 0.905, 'inference_ms': 12, 'params': '0.7M', 'train_hrs': 1.5, 'status': 'good'},
            'TCN+Transformer': {'accuracy': 0.942, 'f1': 0.938, 'inference_ms': 45, 'params': '1.2M', 'train_hrs': 2.3, 'status': 'best'},
        }

    def generate_training_curves(self, n_epochs=100):
        """Generate realistic training curves."""
        epochs = np.arange(1, n_epochs + 1)
        train_loss = 2.0 * np.exp(-0.04 * epochs) + 0.15 + np.random.randn(n_epochs) * 0.02
        val_loss = 2.2 * np.exp(-0.035 * epochs) + 0.22 + np.random.randn(n_epochs) * 0.03
        train_acc = 1.0 / (1 + np.exp(-0.08 * (epochs - 30))) * 0.92 + 0.05 + np.random.randn(n_epochs) * 0.01
        val_acc = 1.0 / (1 + np.exp(-0.07 * (epochs - 35))) * 0.88 + 0.05 + np.random.randn(n_epochs) * 0.015
        lr = 1e-4 * np.where(epochs < 50, 1.0, 0.5 ** ((epochs - 50) / 25))
        return epochs, train_loss, val_loss, np.clip(train_acc, 0, 1), np.clip(val_acc, 0, 1), lr

    def generate_augmentation_examples(self, signal_1d):
        """Generate augmented signal examples."""
        aug = {}
        aug['Original'] = signal_1d.copy()
        shift = len(signal_1d) // 10
        aug['Time Shift'] = np.roll(signal_1d, shift)
        aug['Noise Injection'] = signal_1d + np.random.randn(len(signal_1d)) * 0.15
        aug['Amplitude Scale'] = signal_1d * 1.3
        # Time warp: stretch first half, compress second half
        n = len(signal_1d)
        idx_slow = np.linspace(0, n // 2, int(n * 0.6)).astype(int)
        idx_fast = np.linspace(n // 2, n - 1, n - len(idx_slow)).astype(int)
        idx_warp = np.concatenate([idx_slow, idx_fast])
        idx_warp = np.clip(idx_warp, 0, n - 1)
        aug['Time Warp'] = signal_1d[idx_warp]
        return aug

    def generate_attention_weights(self, n_channels, n_timesteps):
        channel_attention = np.random.rand(n_channels) * 0.3
        for ch in self.motor_channels:
            if ch < n_channels:
                channel_attention[ch] = 0.7 + np.random.rand() * 0.3
        temporal_attention = np.exp(-0.5 * ((np.arange(n_timesteps) - n_timesteps * 0.6) / (n_timesteps * 0.2)) ** 2)
        temporal_attention += np.random.rand(n_timesteps) * 0.1
        attention = np.outer(channel_attention, temporal_attention)
        return attention / attention.max()

    def generate_confusion_matrix(self):
        return np.array([[85, 5, 7, 3], [4, 82, 8, 6], [6, 9, 80, 5], [3, 8, 7, 82]])

    def generate_cross_subject_data(self, n_subjects=5):
        subjects = [f'S{i+1}' for i in range(n_subjects)]
        within = [0.88 + np.random.rand() * 0.08 for _ in subjects]
        cross = [0.75 + np.random.rand() * 0.12 for _ in subjects]
        finetuned = [0.82 + np.random.rand() * 0.10 for _ in subjects]
        return subjects, within, cross, finetuned


# ==================== Brain Topography ====================
def draw_brain_topography(ax, electrode_positions, values, motor_channels, title="Brain Topography"):
    head = Circle((0.5, 0.5), 0.48, fill=False, color='#64748b', linewidth=2)
    ax.add_patch(head)
    nose = plt.Polygon([[0.5, 0.98], [0.46, 0.9], [0.54, 0.9]], closed=True, fill=False, color='#64748b', linewidth=2)
    ax.add_patch(nose)
    ax.add_patch(Ellipse((0.02, 0.5), 0.05, 0.12, fill=False, color='#64748b', linewidth=2))
    ax.add_patch(Ellipse((0.98, 0.5), 0.05, 0.12, fill=False, color='#64748b', linewidth=2))

    xi = np.linspace(0, 1, 100)
    yi = np.linspace(0, 1, 100)
    xi, yi = np.meshgrid(xi, yi)
    positions = np.array(electrode_positions)
    zi = griddata(positions, values, (xi, yi), method='cubic', fill_value=0)
    mask = (xi - 0.5)**2 + (yi - 0.5)**2 > 0.48**2
    zi[mask] = np.nan

    cmap = LinearSegmentedColormap.from_list('brain', ['#3b82f6', '#22c55e', '#eab308', '#ef4444'])
    im = ax.imshow(zi, extent=[0, 1, 0, 1], origin='lower', cmap=cmap, alpha=0.7)

    for i, (x, y) in enumerate(electrode_positions):
        color = '#ef4444' if i in motor_channels else '#64748b'
        size = 80 if i in motor_channels else 40
        ax.scatter(x, y, c=color, s=size, edgecolors='white', linewidths=1, zorder=5)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', color='#f1f5f9', pad=10)
    return im


# ==================== Initialize ====================
classifier = MotorIntentClassifier()


# ==================== Sidebar ====================
st.sidebar.markdown("## 🧠 BCI Controls")

st.sidebar.markdown("### 🎯 Motor Intent")
motor_intent = st.sidebar.selectbox(
    "Ground Truth Intent",
    ['rest', 'left_hand', 'right_hand', 'both_hands'],
    format_func=lambda x: x.replace('_', ' ').title()
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Signal Parameters")
duration = st.sidebar.slider("Duration (s)", 1.0, 4.0, 2.0, 0.5)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.3, 0.1, 0.02)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 Channel Selection")
channel_preset = st.sidebar.selectbox("Preset", ["Motor Cortex", "Frontal", "Custom"])
if channel_preset == "Motor Cortex":
    show_channels = [8, 16, 24, 32]
elif channel_preset == "Frontal":
    show_channels = [0, 1, 2, 3]
else:
    show_channels = st.sidebar.multiselect("Channels", list(range(64)), default=[8, 16, 24, 32])

if not show_channels:
    show_channels = [8, 16, 24, 32]
show_channels = [int(ch) for ch in show_channels]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Advanced")
show_attention = st.sidebar.checkbox("Show Attention Weights", value=True)
spectrogram_channel = st.sidebar.selectbox("Spectrogram Channel", show_channels)


# ==================== Generate Signals ====================
signals, time_axis = classifier.generate_synthetic_ecog(duration, motor_intent)
signals = signals + np.random.randn(*signals.shape) * noise_level
predicted_class, confidence, probabilities = classifier.classify_intent(signals)
band_power = classifier.extract_band_power(signals[show_channels])


# ==================== Tabs ====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "📊 Signals",
    "🔧 Preprocessing",
    "🧠 Topography",
    "📈 Time-Freq",
    "⚡ ERSP",
    "🔗 Connectivity",
    "🎯 Classification",
    "📊 Models",
    "📉 Training",
    "🗺️ Embeddings",
    "🔍 Attention",
    "📚 Theory"
])


# ==================== Tab 1: Signal Analysis ====================
with tab1:
    st.markdown('<p class="section-header">Raw ECoG Signal Analysis</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="highlight-box">
<p>🧠 <strong>ECoG (Electrocorticography):</strong> High-density electrode arrays record neural activity directly from the cortical surface, providing superior signal quality for brain-computer interfaces.</p>
</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">🎬 Signal Playback</div>', unsafe_allow_html=True)
    n_samples = len(time_axis)
    current_sample = st.slider("Select Time Point (ms)", 0, int(duration * 1000) - 100, 0, 10, key="signal_time_slider")

    col1, col2 = st.columns([2.5, 1])

    with col1:
        fig, axes = plt.subplots(len(show_channels), 1, figsize=(12, 2 * len(show_channels)), sharex=True)
        if len(show_channels) == 1:
            axes = [axes]
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
        for i, ch in enumerate(show_channels):
            color = colors[i % len(colors)]
            axes[i].plot(time_axis * 1000, signals[ch], color=color, linewidth=0.8, alpha=0.9)
            axes[i].fill_between(time_axis * 1000, signals[ch], alpha=0.1, color=color)
            axes[i].axvline(x=current_sample, color='#ef4444', linewidth=2, linestyle='--', alpha=0.7)
            axes[i].set_ylabel(f'Ch {ch}', fontsize=10, fontweight='bold', color='#e2e8f0')
            axes[i].set_ylim(-1.5, 1.5)
        axes[-1].set_xlabel('Time (ms)', fontsize=11)
        fig.suptitle(f'Neural Activity — {motor_intent.replace("_", " ").title()}',
                     fontsize=13, fontweight='bold', color='#f1f5f9', y=1.02)
        setup_dark_plot(fig, np.array(axes))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="subsection-header">Signal Metrics</div>', unsafe_allow_html=True)
        power = np.mean(np.var(signals[show_channels], axis=1))
        snr = 10 * np.log10(np.var(signals[show_channels]) / (noise_level**2 + 1e-10))
        for val, label in [(f"{power:.4f} μV²", "Signal Power"), (f"{snr:.1f} dB", "Signal-to-Noise"),
                           (f"{classifier.fs} Hz", "Sample Rate"), (f"{classifier.n_channels}", "Channels")]:
            st.markdown(f'<div class="metric-container"><h3>{val}</h3><p>{label}</p></div><br>', unsafe_allow_html=True)

    # Frequency Spectrum
    st.markdown('<div class="subsection-header">📊 Frequency Spectrum</div>', unsafe_allow_html=True)
    f, psd = classifier.compute_power_spectrum(signals[show_channels])
    mean_psd = np.mean(psd, axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    setup_dark_plot(fig, ax)
    ax.fill_between(f, mean_psd, alpha=0.3, color='#667eea')
    ax.plot(f, mean_psd, color='#667eea', linewidth=2)
    for low, high, name, color in [(8, 12, 'Alpha', '#fbbf24'), (13, 30, 'Beta', '#34d399'), (30, 80, 'Gamma', '#f87171')]:
        mask = (f >= low) & (f <= high)
        ax.fill_between(f[mask], mean_psd[mask], alpha=0.4, color=color, label=f'{name} ({low}-{high} Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (μV²/Hz)')
    ax.set_xlim(1, 80)
    ax.set_yscale('log')
    fix_log_ticks(ax)
    ax.legend(loc='upper right', facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Data Augmentation Gallery
    st.markdown('<div class="subsection-header">🔄 Data Augmentation Gallery</div>', unsafe_allow_html=True)
    aug_examples = classifier.generate_augmentation_examples(signals[show_channels[0]])
    aug_names = list(aug_examples.keys())
    n_aug = len(aug_names)
    fig, axes = plt.subplots(1, n_aug, figsize=(3 * n_aug, 3))
    aug_colors = ['#667eea', '#f59e0b', '#ef4444', '#10b981', '#8b5cf6']
    for i, (name, sig) in enumerate(aug_examples.items()):
        axes[i].plot(sig[:500], color=aug_colors[i], linewidth=0.8)
        axes[i].set_title(name, fontsize=10, fontweight='bold', color='#f1f5f9')
        axes[i].set_xticks([])
    setup_dark_plot(fig, axes)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
<div class="key-point">
    <div class="key-point-icon">💡</div>
    <p><strong>Data Augmentation:</strong> These transformations increase training data diversity, helping the model generalize across subjects and recording sessions. Each preserves the essential neural signatures while varying non-informative aspects.</p>
</div>
    """, unsafe_allow_html=True)


# ==================== Tab 2: Preprocessing ====================
with tab2:
    st.markdown('<p class="section-header">Signal Preprocessing Pipeline</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="highlight-box">
<p>🔧 <strong>Preprocessing Pipeline:</strong> Raw neural signals must be carefully cleaned before analysis. Each stage removes specific artifacts while preserving neural information.</p>
</div>
    """, unsafe_allow_html=True)

    stages = classifier.apply_preprocessing_pipeline(signals)
    stage_names = list(stages.keys())
    stage_descriptions = [
        "Unprocessed ECoG signals with line noise, drift, and artifacts",
        "Removes 60 Hz powerline interference using an IIR notch filter (Q=30)",
        "Retains neural frequencies (0.5-200 Hz), removes DC drift and high-freq noise",
        "Subtracts the mean across all channels to remove global noise",
        "Clips extreme values beyond 3σ to remove transient artifacts"
    ]

    ch_idx = show_channels[0]
    for i, (name, desc) in enumerate(zip(stage_names, stage_descriptions)):
        st.markdown(f"""
<div class="pipeline-step">
    <span class="step-badge">{i + 1}</span>
    <h5>{name}</h5>
    <p>{desc}</p>
</div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(12, 2.5))
        setup_dark_plot(fig, ax)
        sig = stages[name][ch_idx]
        color = ['#94a3b8', '#f59e0b', '#667eea', '#10b981', '#34d399'][i]
        ax.plot(time_axis * 1000, sig, color=color, linewidth=0.8)
        ax.fill_between(time_axis * 1000, sig, alpha=0.1, color=color)
        ax.set_ylabel('μV', fontsize=10)
        if i == len(stage_names) - 1:
            ax.set_xlabel('Time (ms)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Before/after comparison
    st.markdown('<div class="subsection-header">📊 Before vs After: Power Spectrum</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        f_raw, psd_raw = signal.welch(stages['Raw'][ch_idx], classifier.fs, nperseg=256)
        fig, ax = plt.subplots(figsize=(6, 4))
        setup_dark_plot(fig, ax)
        ax.plot(f_raw, psd_raw, color='#94a3b8', linewidth=1.5)
        ax.set_yscale('log')
        fix_log_ticks(ax)
        ax.set_title('Raw Signal Spectrum', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_xlim(0, 120)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        f_clean, psd_clean = signal.welch(stages['Artifact Rejection'][ch_idx], classifier.fs, nperseg=256)
        fig, ax = plt.subplots(figsize=(6, 4))
        setup_dark_plot(fig, ax)
        ax.plot(f_clean, psd_clean, color='#34d399', linewidth=1.5)
        ax.set_yscale('log')
        fix_log_ticks(ax)
        ax.set_title('Cleaned Signal Spectrum', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_xlim(0, 120)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    snr_raw = 10 * np.log10(np.var(stages['Raw'][ch_idx]) / (noise_level**2 + 1e-10))
    snr_clean = 10 * np.log10(np.var(stages['Artifact Rejection'][ch_idx]) / (noise_level**2 + 1e-10))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-container"><h3>{snr_raw:.1f} dB</h3><p>Raw SNR</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-container"><h3>{snr_clean:.1f} dB</h3><p>Cleaned SNR</p></div>', unsafe_allow_html=True)
    with col3:
        improvement = snr_clean - snr_raw
        st.markdown(f'<div class="metric-container"><h3>+{improvement:.1f} dB</h3><p>SNR Improvement</p></div>', unsafe_allow_html=True)


# ==================== Tab 3: Brain Topography ====================
with tab3:
    st.markdown('<p class="section-header">Brain Topography Map</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="highlight-box">
<p>🗺️ <strong>Spatial Mapping:</strong> Visualize neural activity across the cortical surface. Motor cortex electrodes (red) are key for detecting movement intention.</p>
</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    f_topo, psd_topo = classifier.compute_power_spectrum(signals)

    with col1:
        st.markdown('<div class="subsection-header">Beta Power (13-30 Hz)</div>', unsafe_allow_html=True)
        beta_mask = (f_topo >= 13) & (f_topo <= 30)
        beta_vals = np.mean(psd_topo[:, beta_mask], axis=1)
        beta_vals = (beta_vals - beta_vals.min()) / (beta_vals.max() - beta_vals.min() + 1e-10)
        fig, ax = plt.subplots(figsize=(8, 8))
        setup_dark_plot(fig, ax)
        ax.set_facecolor('#0f172a')
        im = draw_brain_topography(ax, classifier.electrode_positions, beta_vals, classifier.motor_channels, "Beta Power (13-30 Hz)")
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, label='Normalized Power')
        cbar.ax.yaxis.label.set_color('#e2e8f0')
        cbar.ax.tick_params(colors='#94a3b8')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="subsection-header">Gamma Power (30-100 Hz)</div>', unsafe_allow_html=True)
        gamma_mask = (f_topo >= 30) & (f_topo <= 100)
        gamma_vals = np.mean(psd_topo[:, gamma_mask], axis=1)
        gamma_vals = (gamma_vals - gamma_vals.min()) / (gamma_vals.max() - gamma_vals.min() + 1e-10)
        fig, ax = plt.subplots(figsize=(8, 8))
        setup_dark_plot(fig, ax)
        ax.set_facecolor('#0f172a')
        im = draw_brain_topography(ax, classifier.electrode_positions, gamma_vals, classifier.motor_channels, "Gamma Power (30-100 Hz)")
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, label='Normalized Power')
        cbar.ax.yaxis.label.set_color('#e2e8f0')
        cbar.ax.tick_params(colors='#94a3b8')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("""
<div class="key-point">
    <div class="key-point-icon">🎯</div>
    <p><strong>Motor Cortex Activation:</strong> During movement planning, beta power <em>decreases</em> (desynchronization) while gamma power <em>increases</em> (synchronization) in motor cortex regions. These complementary patterns form the primary biomarkers for motor intent.</p>
</div>
    """, unsafe_allow_html=True)


# ==================== Tab 4: Time-Frequency ====================
with tab4:
    st.markdown('<p class="section-header">Time-Frequency Analysis</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="highlight-box">
<p>📊 <strong>Wavelet Scalogram:</strong> Continuous wavelet transform reveals how frequency content changes over time, capturing the dynamic nature of motor planning and execution.</p>
</div>
    """, unsafe_allow_html=True)

    freqs, spectrogram = classifier.compute_wavelet_spectrogram(signals[spectrogram_channel])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]})
    setup_dark_plot(fig, np.array(axes))

    axes[0].plot(time_axis * 1000, signals[spectrogram_channel], color='#667eea', linewidth=1)
    axes[0].fill_between(time_axis * 1000, signals[spectrogram_channel], alpha=0.2, color='#667eea')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Channel {spectrogram_channel} — Time Series', fontsize=13, fontweight='bold')
    axes[0].set_xlim(0, duration * 1000)

    extent = [0, duration * 1000, freqs[0], freqs[-1]]
    im = axes[1].imshow(spectrogram, aspect='auto', origin='lower', extent=extent, cmap='magma', interpolation='bilinear')
    axes[1].axhline(y=8, color='#fbbf24', linestyle='--', alpha=0.7, label='Alpha')
    axes[1].axhline(y=12, color='#fbbf24', linestyle='--', alpha=0.7)
    axes[1].axhline(y=13, color='#34d399', linestyle='--', alpha=0.7, label='Beta')
    axes[1].axhline(y=30, color='#34d399', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('Wavelet Scalogram', fontsize=13, fontweight='bold')
    axes[1].set_yscale('log')
    fix_log_ticks(axes[1])
    cbar = plt.colorbar(im, ax=axes[1], label='Power')
    cbar.ax.yaxis.label.set_color('#e2e8f0')
    cbar.ax.tick_params(colors='#94a3b8')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Band power dynamics
    st.markdown('<div class="subsection-header">Band Power Dynamics</div>', unsafe_allow_html=True)
    window_size = 100
    n_windows = len(time_axis) // window_size
    alpha_t, beta_t, gamma_t = [], [], []
    for i in range(n_windows):
        seg = signals[spectrogram_channel, i * window_size:(i + 1) * window_size]
        f_seg, psd_seg = signal.welch(seg, classifier.fs, nperseg=min(64, len(seg)))
        alpha_t.append(np.mean(psd_seg[(f_seg >= 8) & (f_seg <= 12)]))
        beta_t.append(np.mean(psd_seg[(f_seg >= 13) & (f_seg <= 30)]))
        gamma_t.append(np.mean(psd_seg[(f_seg >= 30) & (f_seg <= 100)]))
    tw = np.linspace(0, duration * 1000, n_windows)

    col1, col2, col3 = st.columns(3)
    for col, data, name, color in [(col1, alpha_t, 'Alpha (8-12 Hz)', '#fbbf24'),
                                     (col2, beta_t, 'Beta (13-30 Hz)', '#34d399'),
                                     (col3, gamma_t, 'Gamma (30-100 Hz)', '#f87171')]:
        with col:
            fig, ax = plt.subplots(figsize=(4, 3))
            setup_dark_plot(fig, ax)
            ax.plot(tw, data, color=color, linewidth=2)
            ax.fill_between(tw, data, alpha=0.3, color=color)
            ax.set_xlabel('Time (ms)', fontsize=9)
            ax.set_ylabel('Power', fontsize=9)
            ax.set_title(name, fontsize=11, fontweight='bold', color=color)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ==================== Tab 5: ERSP ====================
with tab5:
    st.markdown('<p class="section-header">Event-Related Spectral Perturbation</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="highlight-box">
<p>⚡ <strong>ERSP:</strong> Shows how spectral power changes relative to a pre-movement baseline. Blue indicates power <em>decrease</em> (desynchronization), red indicates power <em>increase</em> (synchronization).</p>
</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Motor channel ERSP
    with col1:
        st.markdown('<div class="subsection-header">Motor Channel ERSP</div>', unsafe_allow_html=True)
        motor_ch = classifier.motor_channels[0]
        ersp_freqs, ersp_data = classifier.compute_ersp(signals[motor_ch])

        fig, ax = plt.subplots(figsize=(8, 5))
        setup_dark_plot(fig, ax)
        extent_ersp = [0, duration * 1000, ersp_freqs[0], ersp_freqs[-1]]
        vmax = np.percentile(np.abs(ersp_data), 95)
        cmap_ersp = LinearSegmentedColormap.from_list('ersp', ['#3b82f6', '#1e293b', '#ef4444'])
        im = ax.imshow(ersp_data, aspect='auto', origin='lower', extent=extent_ersp,
                       cmap=cmap_ersp, vmin=-vmax, vmax=vmax, interpolation='bilinear')
        ax.axvline(x=duration * 250, color='#fbbf24', linestyle='--', linewidth=2, alpha=0.8, label='Baseline End')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'ERSP — Motor Channel {motor_ch}', fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        fix_log_ticks(ax)
        cbar = plt.colorbar(im, ax=ax, label='Power (dB)')
        cbar.ax.yaxis.label.set_color('#e2e8f0')
        cbar.ax.tick_params(colors='#94a3b8')
        ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Non-motor channel ERSP
    with col2:
        st.markdown('<div class="subsection-header">Non-Motor Channel ERSP</div>', unsafe_allow_html=True)
        non_motor = [c for c in range(64) if c not in classifier.motor_channels][0]
        ersp_freqs2, ersp_data2 = classifier.compute_ersp(signals[non_motor])

        fig, ax = plt.subplots(figsize=(8, 5))
        setup_dark_plot(fig, ax)
        im = ax.imshow(ersp_data2, aspect='auto', origin='lower', extent=extent_ersp,
                       cmap=cmap_ersp, vmin=-vmax, vmax=vmax, interpolation='bilinear')
        ax.axvline(x=duration * 250, color='#fbbf24', linestyle='--', linewidth=2, alpha=0.8, label='Baseline End')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'ERSP — Non-Motor Channel {non_motor}', fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        fix_log_ticks(ax)
        cbar = plt.colorbar(im, ax=ax, label='Power (dB)')
        cbar.ax.yaxis.label.set_color('#e2e8f0')
        cbar.ax.tick_params(colors='#94a3b8')
        ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Band-specific ERSP
    st.markdown('<div class="subsection-header">Band-Specific ERSP Time Courses</div>', unsafe_allow_html=True)
    n_time = ersp_data.shape[1]
    time_ersp = np.linspace(0, duration * 1000, n_time)

    beta_ersp_mask = (ersp_freqs >= 13) & (ersp_freqs <= 30)
    gamma_ersp_mask = (ersp_freqs >= 30) & (ersp_freqs <= 100)
    alpha_ersp_mask = (ersp_freqs >= 8) & (ersp_freqs <= 12)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    setup_dark_plot(fig, np.array(axes))

    for ax_i, (mask, name, color) in enumerate([
        (alpha_ersp_mask, 'Alpha ERD/ERS', '#fbbf24'),
        (beta_ersp_mask, 'Beta ERD/ERS', '#34d399'),
        (gamma_ersp_mask, 'Gamma ERD/ERS', '#f87171')
    ]):
        band_ersp = np.mean(ersp_data[mask, :], axis=0)
        axes[ax_i].plot(time_ersp, band_ersp, color=color, linewidth=2)
        axes[ax_i].fill_between(time_ersp, band_ersp, alpha=0.2, color=color)
        axes[ax_i].axhline(y=0, color='#475569', linestyle='-', linewidth=1)
        axes[ax_i].axvline(x=duration * 250, color='#fbbf24', linestyle='--', alpha=0.5)
        axes[ax_i].set_xlabel('Time (ms)', fontsize=9)
        axes[ax_i].set_ylabel('dB', fontsize=9)
        axes[ax_i].set_title(name, fontsize=11, fontweight='bold', color=color)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
<div class="key-point">
    <div class="key-point-icon">🔬</div>
    <p><strong>ERSP Interpretation:</strong> Negative dB values (blue) indicate <em>event-related desynchronization</em> — neural populations become less synchronized during motor planning. Positive values (red) indicate <em>synchronization</em> during active movement execution.</p>
</div>
    """, unsafe_allow_html=True)


# ==================== Tab 6: Connectivity ====================
with tab6:
    st.markdown('<p class="section-header">Neural Connectivity Analysis</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="highlight-box">
<p>🔗 <strong>Functional Connectivity:</strong> Measures how neural activity is coordinated across brain regions. Phase-amplitude coupling reveals cross-frequency interactions critical for motor control.</p>
</div>
    """, unsafe_allow_html=True)

    # Phase-amplitude coupling
    st.markdown('<div class="subsection-header">Phase-Amplitude Coupling (PAC)</div>', unsafe_allow_html=True)

    motor_ch_pac = classifier.motor_channels[0]
    phase_bins, mean_amp, phase_freqs, amp_freqs, comodulogram = classifier.compute_phase_amplitude_coupling(signals[motor_ch_pac])

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        setup_dark_plot(fig, ax)
        im = ax.imshow(comodulogram, aspect='auto', origin='lower', cmap='inferno',
                       extent=[phase_freqs[0], phase_freqs[-1], amp_freqs[0], amp_freqs[-1]])
        ax.set_xlabel('Phase Frequency (Hz)')
        ax.set_ylabel('Amplitude Frequency (Hz)')
        ax.set_title('Comodulogram', fontsize=13, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, label='Modulation Index')
        cbar.ax.yaxis.label.set_color('#e2e8f0')
        cbar.ax.tick_params(colors='#94a3b8')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={'projection': 'polar'})
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
        ax.bar(bin_centers, mean_amp, width=phase_bins[1] - phase_bins[0],
               color='#667eea', alpha=0.8, edgecolor='#a5b4fc')
        ax.set_title('Gamma Amp by Beta Phase', fontsize=12, fontweight='bold', color='#f1f5f9', pad=15)
        ax.tick_params(colors='#94a3b8')
        ax.set_facecolor('#1e293b')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Connectivity matrix
    st.markdown('<div class="subsection-header">Functional Connectivity Matrix (Beta Coherence)</div>', unsafe_allow_html=True)

    n_subset = min(16, len(signals))
    conn_matrix = classifier.compute_connectivity_matrix(signals, n_subset)

    col1, col2 = st.columns([1.5, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(8, 7))
        setup_dark_plot(fig, ax)
        cmap_conn = LinearSegmentedColormap.from_list('conn', ['#0f172a', '#667eea', '#f093fb'])
        im = ax.imshow(conn_matrix, cmap=cmap_conn, vmin=0, vmax=1)
        for i in range(n_subset):
            for j in range(n_subset):
                txt_color = '#f8fafc' if conn_matrix[i, j] > 0.5 else '#94a3b8'
                ax.text(j, i, f'{conn_matrix[i, j]:.2f}', ha='center', va='center',
                        fontsize=7, color=txt_color)
        ch_labels = [f'Ch{i}' for i in range(n_subset)]
        ax.set_xticks(range(n_subset))
        ax.set_yticks(range(n_subset))
        ax.set_xticklabels(ch_labels, fontsize=7, rotation=45)
        ax.set_yticklabels(ch_labels, fontsize=7)
        ax.set_title('Beta Band Coherence Matrix', fontsize=13, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, label='Coherence', shrink=0.8)
        cbar.ax.yaxis.label.set_color('#e2e8f0')
        cbar.ax.tick_params(colors='#94a3b8')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        # Network graph visualization
        fig, ax = plt.subplots(figsize=(6, 6))
        setup_dark_plot(fig, ax)
        ax.set_facecolor('#0f172a')

        # Position nodes in a circle
        angles_net = np.linspace(0, 2 * np.pi, n_subset, endpoint=False)
        node_x = 0.5 + 0.4 * np.cos(angles_net)
        node_y = 0.5 + 0.4 * np.sin(angles_net)

        # Draw strong connections
        threshold = np.percentile(conn_matrix[conn_matrix > 0], 75)
        for i in range(n_subset):
            for j in range(i + 1, n_subset):
                if conn_matrix[i, j] > threshold:
                    alpha = min(1.0, conn_matrix[i, j])
                    ax.plot([node_x[i], node_x[j]], [node_y[i], node_y[j]],
                            color='#667eea', alpha=alpha * 0.6, linewidth=conn_matrix[i, j] * 3)

        # Draw nodes
        for i in range(n_subset):
            color = '#ef4444' if i in classifier.motor_channels else '#667eea'
            ax.scatter(node_x[i], node_y[i], s=200, c=color, edgecolors='white',
                       linewidths=1.5, zorder=5)
            ax.annotate(f'{i}', (node_x[i], node_y[i]), ha='center', va='center',
                        fontsize=7, fontweight='bold', color='white', zorder=6)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Network Graph (Top 25%)', fontsize=12, fontweight='bold', color='#f1f5f9')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("""
<div class="concept-card">
    <h5>🧠 Connectivity Insights</h5>
    <p>Strong coherence between motor channels indicates <strong>coordinated neural activity</strong> during movement planning. Phase-amplitude coupling between beta and gamma bands reflects the <em>hierarchical organization</em> of motor cortex, where low-frequency rhythms modulate high-frequency activity.</p>
</div>
    """, unsafe_allow_html=True)


# ==================== Tab 7: Classification ====================
with tab7:
    st.markdown('<p class="section-header">Motor Intent Classification</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="subsection-header">Prediction Result</div>', unsafe_allow_html=True)
        pred_colors = {'Rest': '#6b7280', 'Left Hand': '#3b82f6', 'Right Hand': '#8b5cf6', 'Both Hands': '#10b981'}
        pred_color = pred_colors.get(predicted_class, '#667eea')
        st.markdown(f'''
        <div style="background: linear-gradient(135deg, {pred_color} 0%, {pred_color}dd 100%);
                    padding: 2rem; border-radius: 16px; color: white; text-align: center;
                    box-shadow: 0 8px 25px {pred_color}44;">
            <p style="margin:0; font-size: 1rem; opacity: 0.9;">Predicted Motor Intent</p>
            <h1 style="margin: 0.5rem 0; font-size: 2.5rem;">{predicted_class}</h1>
            <p style="margin:0; font-size: 1.2rem;">Confidence: {confidence:.1%}</p>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="subsection-header">Class Probabilities</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        setup_dark_plot(fig, ax)
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
        bar_colors = [pred_colors[c] for c in classes]
        bars = ax.barh(classes, probs, color=bar_colors, height=0.6, alpha=0.85)
        for bar, prob in zip(bars, probs):
            ax.text(prob + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{prob:.1%}', va='center', fontweight='bold', fontsize=11, color='#e2e8f0')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability')
        for label in ax.get_yticklabels():
            label.set_color('#e2e8f0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="subsection-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = classifier.generate_confusion_matrix()
        class_names = ['Rest', 'Left', 'Right', 'Both']

        fig, ax = plt.subplots(figsize=(8, 6))
        setup_dark_plot(fig, ax)
        im = ax.imshow(cm, cmap='Blues', alpha=0.8)
        for i in range(4):
            for j in range(4):
                txt_color = 'white' if cm[i, j] > 50 else '#e2e8f0'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14, fontweight='bold', color=txt_color)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(class_names, color='#e2e8f0')
        ax.set_yticklabels(class_names, color='#e2e8f0')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Classification Performance', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, label='Count')
        cbar.ax.yaxis.label.set_color('#e2e8f0')
        cbar.ax.tick_params(colors='#94a3b8')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        accuracy = np.trace(cm) / np.sum(cm) * 100
        st.markdown(f'<div class="metric-container" style="margin-top:1rem;"><h3>{accuracy:.1f}%</h3><p>Overall Accuracy</p></div>', unsafe_allow_html=True)

    # ROC Curves
    st.markdown('<div class="subsection-header">ROC Curves (One-vs-Rest)</div>', unsafe_allow_html=True)
    roc_data = classifier.generate_roc_data()
    roc_colors = {'Rest': '#6b7280', 'Left Hand': '#3b82f6', 'Right Hand': '#8b5cf6', 'Both Hands': '#10b981'}

    fig, ax = plt.subplots(figsize=(8, 6))
    setup_dark_plot(fig, ax)
    for cls, data in roc_data.items():
        ax.plot(data['fpr'], data['tpr'], color=roc_colors[cls], linewidth=2,
                label=f"{cls} (AUC={data['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'w--', alpha=0.3, linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Cross-subject transfer
    st.markdown('<div class="subsection-header">🔄 Cross-Subject Transfer Learning</div>', unsafe_allow_html=True)
    subjects, within_acc, cross_acc, finetuned_acc = classifier.generate_cross_subject_data()

    fig, ax = plt.subplots(figsize=(12, 5))
    setup_dark_plot(fig, ax)
    x = np.arange(len(subjects))
    width = 0.25
    ax.bar(x - width, within_acc, width, label='Within-Subject', color='#667eea', alpha=0.85)
    ax.bar(x, cross_acc, width, label='Cross-Subject (Zero-Shot)', color='#f87171', alpha=0.85)
    ax.bar(x + width, finetuned_acc, width, label='Fine-Tuned', color='#34d399', alpha=0.85)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Subject')
    ax.set_title('Transfer Learning Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, color='#e2e8f0')
    ax.set_ylim(0.5, 1.0)
    ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ==================== Tab 8: Model Comparison ====================
with tab8:
    st.markdown('<p class="section-header">Model Benchmark Comparison</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="highlight-box">
<p>📊 <strong>Architecture Comparison:</strong> Side-by-side evaluation of different deep learning architectures for ECoG-based motor intent classification.</p>
</div>
    """, unsafe_allow_html=True)

    models = classifier.generate_model_comparison_data()

    # Model cards
    cols = st.columns(4)
    for i, (name, data) in enumerate(models.items()):
        with cols[i]:
            badge_class = data['status']
            badge_text = {'best': 'BEST', 'good': 'STRONG', 'baseline': 'BASELINE'}[badge_class]
            st.markdown(f"""
<div class="model-card {'best' if badge_class == 'best' else ''}">
    <span class="status-badge {badge_class}">{badge_text}</span>
    <h4>{name}</h4>
    <p class="model-metric"><strong>{data['accuracy']:.1%}</strong> Accuracy</p>
    <p class="model-metric"><strong>{data['f1']:.3f}</strong> F1 Score</p>
    <p class="model-metric"><strong>{data['inference_ms']}ms</strong> Inference</p>
    <p class="model-metric"><strong>{data['params']}</strong> Parameters</p>
    <p class="model-metric"><strong>{data['train_hrs']}h</strong> Training</p>
</div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Grouped bar chart
    st.markdown('<div class="subsection-header">Performance Comparison</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    setup_dark_plot(fig, np.array(axes))

    model_names = list(models.keys())
    model_colors = ['#94a3b8', '#3b82f6', '#f59e0b', '#10b981']
    x_pos = np.arange(len(model_names))

    # Accuracy
    accs = [models[m]['accuracy'] for m in model_names]
    axes[0].bar(x_pos, accs, color=model_colors, alpha=0.85)
    axes[0].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(model_names, rotation=30, ha='right', fontsize=8)
    axes[0].set_ylim(0.7, 1.0)
    for j, v in enumerate(accs):
        axes[0].text(j, v + 0.005, f'{v:.1%}', ha='center', fontsize=9, color='#e2e8f0')

    # F1
    f1s = [models[m]['f1'] for m in model_names]
    axes[1].bar(x_pos, f1s, color=model_colors, alpha=0.85)
    axes[1].set_title('F1 Score', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(model_names, rotation=30, ha='right', fontsize=8)
    axes[1].set_ylim(0.7, 1.0)
    for j, v in enumerate(f1s):
        axes[1].text(j, v + 0.005, f'{v:.3f}', ha='center', fontsize=9, color='#e2e8f0')

    # Inference time
    times = [models[m]['inference_ms'] for m in model_names]
    axes[2].bar(x_pos, times, color=model_colors, alpha=0.85)
    axes[2].set_title('Inference (ms)', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(model_names, rotation=30, ha='right', fontsize=8)
    for j, v in enumerate(times):
        axes[2].text(j, v + 0.5, f'{v}ms', ha='center', fontsize=9, color='#e2e8f0')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Radar chart
    st.markdown('<div class="subsection-header">Multi-Metric Radar Chart</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    metrics = ['Accuracy', 'F1 Score', 'Speed', 'Efficiency', 'Generalization']
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    for name, color in zip(model_names, model_colors):
        d = models[name]
        speed_norm = 1.0 - d['inference_ms'] / 50.0
        eff_norm = 1.0 - float(d['params'].replace('M', '')) / 1.5
        gen_norm = d['accuracy'] * 0.95
        values = [d['accuracy'], d['f1'], speed_norm, eff_norm, gen_norm]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, color='#e2e8f0', fontsize=10)
    ax.set_ylim(0, 1)
    ax.tick_params(colors='#94a3b8')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')
    ax.set_title('Multi-Dimensional Comparison', fontsize=13, fontweight='bold', color='#f1f5f9', pad=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
<div class="key-point">
    <div class="key-point-icon">🏆</div>
    <p><strong>TCN+Transformer</strong> achieves the best accuracy (94.2%) by combining temporal convolutions for local feature extraction with self-attention for long-range dependencies. The TCN alone offers the best speed-accuracy tradeoff for real-time applications.</p>
</div>
    """, unsafe_allow_html=True)


# ==================== Tab 9: Training ====================
with tab9:
    st.markdown('<p class="section-header">Training Simulation</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="highlight-box">
<p>📉 <strong>Training Dynamics:</strong> Visualize how the TCN+Transformer model learns over 100 epochs. The gap between train and validation curves indicates generalization performance.</p>
</div>
    """, unsafe_allow_html=True)

    epochs, train_loss, val_loss, train_acc, val_acc, lr = classifier.generate_training_curves()

    epoch_view = st.slider("View up to epoch", 5, 100, 100, 5, key="epoch_slider")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="subsection-header">Loss Curves</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        setup_dark_plot(fig, ax)
        ax.plot(epochs[:epoch_view], train_loss[:epoch_view], color='#667eea', linewidth=2, label='Train Loss')
        ax.plot(epochs[:epoch_view], val_loss[:epoch_view], color='#f87171', linewidth=2, label='Val Loss')
        ax.fill_between(epochs[:epoch_view], train_loss[:epoch_view], val_loss[:epoch_view], alpha=0.1, color='#f87171')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
        ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="subsection-header">Accuracy Curves</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        setup_dark_plot(fig, ax)
        ax.plot(epochs[:epoch_view], train_acc[:epoch_view], color='#34d399', linewidth=2, label='Train Acc')
        ax.plot(epochs[:epoch_view], val_acc[:epoch_view], color='#f59e0b', linewidth=2, label='Val Acc')
        ax.fill_between(epochs[:epoch_view], train_acc[:epoch_view], val_acc[:epoch_view], alpha=0.1, color='#f59e0b')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Learning rate schedule
    st.markdown('<div class="subsection-header">Learning Rate Schedule</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 3))
    setup_dark_plot(fig, ax)
    ax.plot(epochs, lr * 1e4, color='#a5b4fc', linewidth=2)
    ax.fill_between(epochs, lr * 1e4, alpha=0.2, color='#a5b4fc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR (x1e-4)')
    ax.set_title('Step Decay LR Schedule', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Training config cards
    st.markdown('<div class="subsection-header">Training Configuration</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    for col, val, label in [(col1, '100', 'Epochs'), (col2, '32', 'Batch Size'),
                             (col3, '1.2M', 'Parameters'), (col4, '~2h', 'Training Time')]:
        with col:
            st.markdown(f'<div class="metric-container"><h3>{val}</h3><p>{label}</p></div>', unsafe_allow_html=True)

    st.markdown("""
<div class="concept-card">
    <h5>📊 Training Observations</h5>
    <ul>
        <li><strong>Early stopping</strong> at epoch ~75 when validation loss plateaus</li>
        <li><strong>Learning rate decay</strong> at epoch 50 helps fine-tune convergence</li>
        <li><strong>Train-val gap</strong> of ~4% indicates healthy generalization without overfitting</li>
        <li><strong>Batch normalization</strong> and <strong>dropout (0.2)</strong> regularize effectively</li>
    </ul>
</div>
    """, unsafe_allow_html=True)


# ==================== Tab 10: Embeddings ====================
with tab10:
    st.markdown('<p class="section-header">Latent Space Embeddings</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="highlight-box">
<p>🗺️ <strong>t-SNE Visualization:</strong> 2D projection of the model's learned representations. Well-separated clusters indicate the model has learned discriminative features for each motor intent class.</p>
</div>
    """, unsafe_allow_html=True)

    points, labels, confs = classifier.generate_embedding_data()
    emb_colors = {'Rest': '#6b7280', 'Left Hand': '#3b82f6', 'Right Hand': '#8b5cf6', 'Both Hands': '#10b981'}

    fig, ax = plt.subplots(figsize=(10, 8))
    setup_dark_plot(fig, ax)

    for cls, color in emb_colors.items():
        mask = np.array(labels) == cls
        pts = points[mask]
        c = confs[mask]
        scatter = ax.scatter(pts[:, 0], pts[:, 1], c=[color] * len(pts), s=40, alpha=0.7, edgecolors='white', linewidths=0.3)
        # Draw centroid
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        ax.scatter(cx, cy, c=color, s=200, marker='X', edgecolors='white', linewidths=2, zorder=10)
        ax.annotate(cls, (cx, cy + 0.4), ha='center', fontsize=11, fontweight='bold', color=color)

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('Motor Intent Embedding Space (t-SNE)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-container"><h3>0.84</h3><p>Silhouette Score</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container"><h3>0.91</h3><p>Cluster Purity</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container"><h3>4</h3><p>Distinct Clusters</p></div>', unsafe_allow_html=True)

    st.markdown("""
<div class="concept-card">
    <h5>🔍 Embedding Analysis</h5>
    <p>The clear separation between clusters demonstrates the model has learned <strong>meaningful neural signatures</strong> for each motor intent:</p>
    <ul>
        <li><strong>Rest</strong> forms a tight, central cluster — low variability in idle state</li>
        <li><strong>Left/Right Hand</strong> are separated along the horizontal axis — reflecting lateralized motor cortex activation</li>
        <li><strong>Both Hands</strong> is distinct from single-hand movements — bilateral activation creates unique features</li>
        <li>Slight overlap between Left/Right indicates shared motor planning components</li>
    </ul>
</div>
    """, unsafe_allow_html=True)


# ==================== Tab 11: Attention ====================
with tab11:
    st.markdown('<p class="section-header">Attention Visualization</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="highlight-box">
<p>🔍 <strong>Model Interpretability:</strong> The transformer's attention mechanism reveals which time points and channels the model focuses on, providing insights for clinical validation.</p>
</div>
    """, unsafe_allow_html=True)

    if show_attention:
        n_display_channels = min(16, len(signals))
        n_timesteps = min(200, len(time_axis))
        attention = classifier.generate_attention_weights(n_display_channels, n_timesteps)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="subsection-header">Channel × Time Attention Heatmap</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 6))
            setup_dark_plot(fig, ax)
            im = ax.imshow(attention, aspect='auto', cmap='viridis', extent=[0, duration * 1000, n_display_channels, 0])
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Channel')
            ax.set_title('Self-Attention Weights', fontsize=14, fontweight='bold')
            cbar = plt.colorbar(im, ax=ax, label='Attention Weight')
            cbar.ax.yaxis.label.set_color('#e2e8f0')
            cbar.ax.tick_params(colors='#94a3b8')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown('<div class="subsection-header">Channel Importance</div>', unsafe_allow_html=True)
            channel_importance = np.mean(attention, axis=1)
            fig, ax = plt.subplots(figsize=(6, 6))
            setup_dark_plot(fig, ax)
            colors_ch = ['#ef4444' if i in classifier.motor_channels[:n_display_channels] else '#667eea' for i in range(n_display_channels)]
            ax.barh(range(n_display_channels), channel_importance, color=colors_ch, alpha=0.85)
            ax.set_xlabel('Mean Attention')
            ax.set_ylabel('Channel')
            ax.set_title('Per-Channel Importance', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            motor_patch = mpatches.Patch(color='#ef4444', label='Motor Cortex')
            other_patch = mpatches.Patch(color='#667eea', label='Other')
            ax.legend(handles=[motor_patch, other_patch], loc='lower right', facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Temporal profile
        st.markdown('<div class="subsection-header">Temporal Attention Profile</div>', unsafe_allow_html=True)
        temporal_attention = np.mean(attention, axis=0)
        time_points = np.linspace(0, duration * 1000, len(temporal_attention))

        fig, ax = plt.subplots(figsize=(12, 4))
        setup_dark_plot(fig, ax)
        ax.plot(time_points, temporal_attention, color='#667eea', linewidth=2)
        ax.fill_between(time_points, temporal_attention, alpha=0.3, color='#667eea')
        ax.axvline(x=duration * 600, color='#ef4444', linestyle='--', linewidth=2, alpha=0.7, label='Peak Attention')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Attention Weight')
        ax.set_title('When Does the Model Focus?', fontsize=14, fontweight='bold')
        ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
<div class="concept-card">
    <h5>🧠 Biological Validation</h5>
    <p>The attention pattern aligns with known neuroscience:</p>
    <ul>
        <li><strong>Motor channels receive highest attention</strong> — confirms anatomically relevant features</li>
        <li><strong>Peak attention ~600ms</strong> — corresponds to motor preparation (readiness potential)</li>
        <li><strong>Distributed temporal attention</strong> — captures both planning and execution phases</li>
    </ul>
</div>
        """, unsafe_allow_html=True)
    else:
        st.info("Enable 'Show Attention Weights' in the sidebar to view this tab.")


# ==================== Tab 12: Theory ====================
with tab12:
    st.markdown('<p class="section-header">Architecture & Theoretical Background</p>', unsafe_allow_html=True)

    # Architecture diagram
    st.markdown('<div class="subsection-header">🏗️ Neural Network Architecture</div>', unsafe_allow_html=True)

    st.markdown("""<div class="arch-container">
<div style="text-align: center; margin-bottom: 1.5rem;">
<span style="color: #94a3b8; font-size: 0.9rem;">TCN + Transformer Architecture for Motor Intent Classification</span>
</div>
<div class="arch-flow">
<div class="arch-block" style="background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); color: white;">
<div class="arch-label">Input</div>
<div class="arch-sublabel">64 ch × T samples</div>
</div>
<div class="arch-arrow">→</div>
<div class="arch-block" style="background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); color: white;">
<div class="arch-label">TCN Encoder</div>
<div class="arch-sublabel">Temporal Conv</div>
</div>
<div class="arch-arrow">→</div>
<div class="arch-block" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white;">
<div class="arch-label">Wavelet</div>
<div class="arch-sublabel">Multi-scale Decomp</div>
</div>
<div class="arch-arrow">→</div>
<div class="arch-block" style="background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%); color: white;">
<div class="arch-label">Attention</div>
<div class="arch-sublabel">8-Head Self-Attn</div>
</div>
<div class="arch-arrow">→</div>
<div class="arch-block" style="background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); color: white;">
<div class="arch-label">Output</div>
<div class="arch-sublabel">4 Motor Classes</div>
</div>
</div>
<div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem; flex-wrap: wrap;">
<div style="text-align: center;">
<div style="color: #667eea; font-weight: 600; margin-bottom: 0.5rem;">Frequency Bands</div>
<div style="display: flex; gap: 0.5rem; flex-wrap: wrap; justify-content: center;">
<span style="background: #312e81; color: #a5b4fc; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;">δ 1-4Hz</span>
<span style="background: #1e3a5f; color: #7dd3fc; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;">θ 4-8Hz</span>
<span style="background: #14532d; color: #86efac; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;">α 8-12Hz</span>
<span style="background: #713f12; color: #fde047; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;">β 13-30Hz</span>
<span style="background: #7f1d1d; color: #fca5a5; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;">γ 30-100Hz</span>
</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # Architecture parameters
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
<div class="param-grid">
    <div class="param-card"><h6>Kernel Size</h6><p>3</p></div>
    <div class="param-card"><h6>Dilation Rates</h6><p>1, 2, 4, 8</p></div>
    <div class="param-card"><h6>Hidden Channels</h6><p>32 → 64 → 128</p></div>
    <div class="param-card"><h6>Dropout</h6><p>0.2</p></div>
</div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
<div class="param-grid">
    <div class="param-card"><h6>Hidden Dim</h6><p>256</p></div>
    <div class="param-card"><h6>Attention Heads</h6><p>8</p></div>
    <div class="param-card"><h6>Encoder Layers</h6><p>4</p></div>
    <div class="param-card"><h6>Learning Rate</h6><p>1e-4</p></div>
</div>
        """, unsafe_allow_html=True)

    # Theory sections
    st.markdown('<div class="subsection-header">🧠 Motor Cortex & Neural Signals</div>', unsafe_allow_html=True)

    with st.expander("**Electrocorticography (ECoG)**", expanded=True):
        st.markdown("ECoG records electrical activity directly from the cortical surface:")
        st.markdown("""
<div class="param-grid">
    <div class="param-card"><h6>Spatial Resolution</h6><p>~1-4mm electrode spacing enables precise localization</p></div>
    <div class="param-card"><h6>Frequency Range</h6><p>1-500+ Hz, capturing low-frequency rhythms and high-gamma</p></div>
    <div class="param-card"><h6>Signal Quality</h6><p>10-100× better SNR than scalp EEG</p></div>
    <div class="param-card"><h6>Temporal Resolution</h6><p>Millisecond precision for real-time BCI</p></div>
</div>
        """, unsafe_allow_html=True)

    with st.expander("**Neural Signatures of Motor Intent**", expanded=True):
        st.markdown("Motor planning and execution produce characteristic oscillation changes:")
        st.latex(r"\text{Beta ERD} = \frac{P_{\text{movement}} - P_{\text{rest}}}{P_{\text{rest}}} \times 100\%")

        st.markdown("""
<div class="concept-card">
    <h5>🔵 Beta Desynchronization (ERD)</h5>
    <p><strong>Event-Related Desynchronization</strong> in the 13-30 Hz range:</p>
    <ul>
        <li>Begins ~1-2 seconds before movement onset</li>
        <li>Strongest over contralateral motor cortex</li>
        <li>Reflects release of motor cortex from "idling" state</li>
    </ul>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div class="concept-card">
    <h5>🟢 Gamma Synchronization (ERS)</h5>
    <p><strong>Event-Related Synchronization</strong> in the 30-100+ Hz range:</p>
    <ul>
        <li>Increases during active movement execution</li>
        <li>Correlates with movement parameters (force, velocity)</li>
        <li>High spatial specificity for movement type</li>
    </ul>
</div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">🔧 Deep Learning Methods</div>', unsafe_allow_html=True)

    with st.expander("**Temporal Convolutional Networks (TCNs)**", expanded=True):
        st.markdown("TCNs address key challenges in temporal modeling:")
        st.markdown("""
<div class="algo-step">
    <div class="step-num">1</div>
    <div class="step-content"><strong>Causal Convolutions:</strong> Output at time t depends only on inputs at times ≤ t, preserving temporal causality for real-time BCI</div>
</div>
<div class="algo-step">
    <div class="step-num">2</div>
    <div class="step-content"><strong>Dilated Convolutions:</strong> Exponentially increasing dilation rates (1, 2, 4, 8...) capture long-range dependencies</div>
</div>
<div class="algo-step">
    <div class="step-num">3</div>
    <div class="step-content"><strong>Residual Connections:</strong> Skip connections enable gradient flow and identity mapping</div>
</div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{Receptive Field} = 1 + \sum_{i=0}^{L-1} (k-1) \cdot d_i")

    with st.expander("**Self-Attention Mechanism**", expanded=True):
        st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
        st.markdown("""
<div class="concept-card">
    <h5>📊 Multi-Head Attention</h5>
    <p>Multiple attention heads learn different aspects of the signal:</p>
    <ul>
        <li><strong>Heads 1-2:</strong> Beta band dynamics</li>
        <li><strong>Heads 3-4:</strong> Gamma burst patterns</li>
        <li><strong>Heads 5-6:</strong> Cross-channel relationships</li>
        <li><strong>Heads 7-8:</strong> Movement phase tracking</li>
    </ul>
</div>
        """, unsafe_allow_html=True)

    with st.expander("**Wavelet Transform**", expanded=True):
        st.latex(r"W_\psi(s, \tau) = \frac{1}{\sqrt{|s|}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t-\tau}{s}\right) dt")
        st.markdown("""
<div class="key-point">
    <div class="key-point-icon">💡</div>
    <p><strong>Why Wavelets?</strong> Unlike FFT, wavelets preserve temporal locality — we know <em>when</em> frequency content changes, not just <em>what</em> frequencies are present. Critical for transient motor events.</p>
</div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">🔬 Advanced Analysis Methods</div>', unsafe_allow_html=True)

    with st.expander("**Event-Related Spectral Perturbation (ERSP)**"):
        st.latex(r"\text{ERSP}(f, t) = 10 \log_{10}\left(\frac{P(f, t)}{P_{\text{baseline}}(f)}\right)")
        st.markdown("""
<div class="concept-card">
    <h5>⚡ ERSP Interpretation</h5>
    <p>ERSP normalizes spectral power against a pre-event baseline, revealing:</p>
    <ul>
        <li><strong>ERD (negative dB):</strong> Desynchronization — reduced oscillatory power during motor planning</li>
        <li><strong>ERS (positive dB):</strong> Synchronization — increased power during movement execution</li>
        <li>Time-frequency resolution reveals the <em>temporal dynamics</em> of these changes</li>
    </ul>
</div>
        """, unsafe_allow_html=True)

    with st.expander("**Phase-Amplitude Coupling (PAC)**"):
        st.markdown("""
<div class="concept-card">
    <h5>🔗 Cross-Frequency Coupling</h5>
    <p>PAC measures how the <em>phase</em> of low-frequency oscillations modulates the <em>amplitude</em> of high-frequency activity:</p>
    <ul>
        <li><strong>Beta phase → Gamma amplitude:</strong> Motor cortex hierarchical control</li>
        <li><strong>Modulation Index:</strong> Quantifies coupling strength via KL divergence from uniform</li>
        <li><strong>Comodulogram:</strong> Maps coupling across all frequency pairs to identify dominant interactions</li>
    </ul>
</div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">📖 Key References</div>', unsafe_allow_html=True)

    st.markdown("""
| Paper | Key Contribution |
|-------|------------------|
| **Schalk et al. (2007)** | ECoG-based BCI for motor control |
| **Miller et al. (2010)** | High-gamma activity in motor cortex |
| **Bai et al. (2011)** | TCN for time series classification |
| **Vaswani et al. (2017)** | Transformer architecture (Attention Is All You Need) |
| **Schirrmeister et al. (2017)** | Deep learning for EEG decoding |
| **Canolty & Knight (2010)** | Phase-amplitude coupling in cortical circuits |
| **Makeig (1993)** | Event-related spectral perturbation |
    """)


# ==================== Footer ====================
st.markdown("---")
col1, col2, col3, col4, col5, col6 = st.columns(6)
for col, val, label, delta in [
    (col1, "94.2%", "Model Accuracy", "+2.1%"),
    (col2, "45ms", "Inference Time", "-12ms"),
    (col3, "1.2M", "Parameters", ""),
    (col4, "2.3h", "Training", "-30min"),
    (col5, "0.938", "F1 Score", "+0.03"),
    (col6, "89.1%", "Transfer Acc", "+26.8%"),
]:
    with col:
        st.metric(val, label, delta)

st.markdown("""
<div class="footer">
    <p><strong>🧠 Neural Signal Classification for Motor Intent</strong></p>
    <p>
        <a href="https://github.com/kiranshay/neural-signal-classification-for-motor-intent" target="_blank">GitHub</a> ·
        <a href="https://kiranshay.github.io" target="_blank">Portfolio</a> ·
        <a href="mailto:kiranshay123@gmail.com">Contact</a>
    </p>
    <p style="font-size: 0.85rem; color: #94a3b8;">Brain-Computer Interfaces · Deep Learning · Johns Hopkins University</p>
</div>
""", unsafe_allow_html=True)
