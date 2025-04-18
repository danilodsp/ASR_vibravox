# plot_vibravox.py
"""
Visualize audio samples from the Vibravox dataset.
"""

import matplotlib.pyplot as plt
from datasets import load_dataset

# Load the vibravox dataset (speechless_clean subset)
subset = "speechless_clean"
vibravox = load_dataset("Cnam-LMSSC/vibravox", subset)

# Plot a few audio samples
for i, example in enumerate(vibravox['train'][:3]):
    plt.figure(figsize=(10, 3))
    plt.plot(example['audio.headset_microphone']['array'])
    plt.title(f'Vibravox Example {i}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
