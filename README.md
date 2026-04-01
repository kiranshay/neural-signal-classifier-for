# Neural Signal Classification for Motor Intent

Interactive Streamlit demo exploring brain-computer interface (BCI) concepts
through synthetic ECoG signal generation, preprocessing, and visualization.

## What It Does

The app simulates a motor-intent classification pipeline end to end:

- **Synthetic ECoG generation** with configurable noise, duration, and motor intent
- **Preprocessing pipeline** (notch filter, bandpass, CAR, artifact rejection)
- **Time-frequency analysis** using continuous wavelet transforms
- **Brain topography** mapping of beta and gamma band power
- **Event-Related Spectral Perturbation (ERSP)** visualization
- **Phase-amplitude coupling** and functional connectivity matrices
- **Classification demo** using spectral-power heuristics (not a trained model)
- **Architecture & theory tab** covering TCNs, transformers, and wavelet math

## Running Locally

```bash
pip install streamlit numpy scipy matplotlib
streamlit run app.py
```

## Limitations

- **All data is synthetic.** Signals are generated from sine waves plus noise,
  not recorded from human subjects or loaded from any dataset.
- **No trained model.** Classification uses hand-crafted spectral-power
  thresholds, not a learned neural network. PyTorch is not used.
- **Hardcoded metrics.** Accuracy figures, confusion matrices, ROC curves,
  training curves, and embedding clusters are illustrative — they are not
  computed from real experiments.
- **Not validated on real BCI data.** The demo teaches signal-processing
  concepts but should not be cited as a research result.

## References

- Schalk, G. et al. (2007). Decoding two-dimensional movement trajectories
  using electrocorticographic signals. *J. Neural Eng.* 4(3), 264.
- Miller, K. J. et al. (2010). Cortical activity during motor execution,
  motor imagery, and imagery-based online feedback. *PNAS* 107(9), 4430-4435.
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Schirrmeister, R. T. et al. (2017). Deep learning with convolutional
  neural networks for EEG decoding and visualization. *Hum. Brain Mapp.*
- Canolty, R. T. & Knight, R. T. (2010). The functional role of
  cross-frequency coupling. *Trends Cogn. Sci.* 14(11), 506-515.
- Makeig, S. (1993). Auditory event-related dynamics of the EEG spectrum
  and effects of exposure to tones. *Electroencephalogr. Clin. Neurophysiol.*

## License

MIT
