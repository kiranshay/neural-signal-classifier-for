"""Train BCI motor intent classifier on synthetic ECoG data."""
import numpy as np
from scipy import signal
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import joblib
import os

np.random.seed(42)

CLASSES = ['Rest', 'Left Hand', 'Right Hand', 'Both Hands']
N_TRIALS = 1500  # per class
FS = 512  # sampling rate
DURATION = 2.0  # seconds per trial
N_CHANNELS = 8

def generate_ecog_trial(motor_intent, fs=FS, duration=DURATION, n_channels=N_CHANNELS):
    """Generate synthetic ECoG with physiologically plausible class differences.

    Key neuroscience principles encoded:
    - Rest: balanced alpha (8-13Hz) and beta (13-30Hz) power
    - Left Hand: beta desynchronization in RIGHT hemisphere channels (contralateral), gamma increase
    - Right Hand: beta desynchronization in LEFT hemisphere channels, gamma increase
    - Both Hands: bilateral beta desynchronization, strong gamma bilaterally
    """
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    data = np.zeros((n_channels, n_samples))

    # Base signal: 1/f noise + alpha + beta for all channels
    for ch in range(n_channels):
        # 1/f background
        freqs = np.fft.rfftfreq(n_samples, 1/fs)
        freqs[0] = 1
        spectrum = 1.0 / np.sqrt(freqs)
        phases = np.random.uniform(0, 2*np.pi, len(freqs))
        pink = np.fft.irfft(spectrum * np.exp(1j * phases), n=n_samples)

        # Alpha (8-13 Hz)
        alpha_amp = np.random.uniform(0.8, 1.2)
        alpha_freq = np.random.uniform(9, 11)
        alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * t + np.random.uniform(0, 2*np.pi))

        # Beta (15-25 Hz)
        beta_amp = np.random.uniform(0.6, 1.0)
        beta_freq = np.random.uniform(18, 22)
        beta = beta_amp * np.sin(2 * np.pi * beta_freq * t + np.random.uniform(0, 2*np.pi))

        # Gamma (30-80 Hz)
        gamma_amp = np.random.uniform(0.1, 0.3)
        gamma_freq = np.random.uniform(40, 60)
        gamma = gamma_amp * np.sin(2 * np.pi * gamma_freq * t + np.random.uniform(0, 2*np.pi))

        data[ch] = pink * 0.5 + alpha + beta + gamma + np.random.randn(n_samples) * 0.3

    # Apply class-specific modulations
    # Channels 0-3: left hemisphere, Channels 4-7: right hemisphere
    left_chs = [0, 1, 2, 3]
    right_chs = [4, 5, 6, 7]

    if motor_intent == 'Rest':
        pass  # no modulation

    elif motor_intent == 'Left Hand':
        # Contralateral (RIGHT hemisphere) beta desynchronization
        for ch in right_chs:
            # Suppress beta band
            b, a = signal.butter(4, [13, 30], btype='band', fs=fs)
            beta_component = signal.filtfilt(b, a, data[ch])
            data[ch] -= beta_component * np.random.uniform(0.5, 0.7)
            # Enhance gamma
            gamma_boost = np.random.uniform(0.3, 0.6) * np.sin(2*np.pi*np.random.uniform(35,55)*t)
            data[ch] += gamma_boost

    elif motor_intent == 'Right Hand':
        # Contralateral (LEFT hemisphere) beta desynchronization
        for ch in left_chs:
            b, a = signal.butter(4, [13, 30], btype='band', fs=fs)
            beta_component = signal.filtfilt(b, a, data[ch])
            data[ch] -= beta_component * np.random.uniform(0.5, 0.7)
            gamma_boost = np.random.uniform(0.3, 0.6) * np.sin(2*np.pi*np.random.uniform(35,55)*t)
            data[ch] += gamma_boost

    elif motor_intent == 'Both Hands':
        # Bilateral beta desynchronization + strong gamma
        for ch in range(n_channels):
            b, a = signal.butter(4, [13, 30], btype='band', fs=fs)
            beta_component = signal.filtfilt(b, a, data[ch])
            data[ch] -= beta_component * np.random.uniform(0.4, 0.65)
            gamma_boost = np.random.uniform(0.4, 0.7) * np.sin(2*np.pi*np.random.uniform(35,55)*t)
            data[ch] += gamma_boost

    return data

def extract_features(data, fs=FS):
    """Extract frequency-domain features from multichannel ECoG."""
    features = []
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 80)}

    for ch in range(data.shape[0]):
        freqs, psd = signal.welch(data[ch], fs=fs, nperseg=256)

        # Band powers
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            features.append(np.log1p(np.mean(psd[mask])))

        # Band power ratios
        beta_mask = (freqs >= 13) & (freqs <= 30)
        gamma_mask = (freqs >= 30) & (freqs <= 80)
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        beta_power = np.mean(psd[beta_mask])
        gamma_power = np.mean(psd[gamma_mask])
        alpha_power = np.mean(psd[alpha_mask])
        features.append(gamma_power / (beta_power + 1e-10))
        features.append(beta_power / (alpha_power + 1e-10))

        # Hjorth parameters
        diff1 = np.diff(data[ch])
        diff2 = np.diff(diff1)
        activity = np.var(data[ch])
        mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
        complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
        features.extend([np.log1p(activity), mobility, complexity])

    # Inter-hemisphere features (left vs right asymmetry)
    for band_idx, band_name in enumerate(bands):
        left_power = np.mean([features[ch * 10 + band_idx] for ch in range(4)])
        right_power = np.mean([features[ch * 10 + band_idx] for ch in range(4, 8)])
        features.append(left_power - right_power)  # laterality index

    return np.array(features)

print("Generating synthetic ECoG training data...")
X, y = [], []
for cls in CLASSES:
    for i in range(N_TRIALS):
        trial = generate_ecog_trial(cls)
        feats = extract_features(trial)
        X.append(feats)
        y.append(cls)
        if (i+1) % 500 == 0:
            print(f"  {cls}: {i+1}/{N_TRIALS} trials")

X = np.array(X)
y = np.array(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Train Random Forest with calibrated probabilities
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_cal = CalibratedClassifierCV(rf, cv=5, method='isotonic')
rf_cal.fit(X_train_scaled, y_train_enc)

# Cross-validation scores for multiple models
print("Cross-validating models...")
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
}
cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train_enc, cv=5, scoring='accuracy')
    cv_scores[name] = {'mean': float(scores.mean()), 'std': float(scores.std())}
    print(f"  {name}: {scores.mean():.3f} +/- {scores.std():.3f}")

# Evaluation on test set
y_pred = rf_cal.predict(X_test_scaled)
y_proba = rf_cal.predict_proba(X_test_scaled)

# Confusion matrix
cm = confusion_matrix(y_test_enc, y_pred)

# ROC curves (one-vs-rest)
roc_data = {}
for i, cls_name in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve((y_test_enc == i).astype(int), y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    roc_data[le.inverse_transform([i])[0]] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(roc_auc)}

# Feature importance
feature_names = []
bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
for ch in range(8):
    for band in bands:
        feature_names.append(f'ch{ch}_{band}_power')
    feature_names.extend([f'ch{ch}_gamma_beta_ratio', f'ch{ch}_beta_alpha_ratio'])
    feature_names.extend([f'ch{ch}_activity', f'ch{ch}_mobility', f'ch{ch}_complexity'])
for band in bands:
    feature_names.append(f'laterality_{band}')

# Save
os.makedirs('models', exist_ok=True)

joblib.dump({
    'model': rf_cal,
    'scaler': scaler,
    'label_encoder': le,
    'feature_names': feature_names,
}, 'models/bci_classifier.pkl')

joblib.dump({
    'confusion_matrix': cm,
    'class_names': le.classes_.tolist(),
    'roc_data': roc_data,
    'cv_scores': cv_scores,
    'classification_report': classification_report(y_test_enc, y_pred, target_names=[le.inverse_transform([i])[0] for i in range(len(le.classes_))], output_dict=True),
    'feature_importance': dict(zip(feature_names, rf.fit(X_train_scaled, y_train_enc).feature_importances_.tolist())),
}, 'models/eval_metrics.pkl')

print(f"\nTest accuracy: {(y_pred == y_test_enc).mean():.3f}")
print("Models saved to models/")
