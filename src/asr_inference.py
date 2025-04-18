from transformers import pipeline
from datasets import load_dataset

# Load a pre-trained ASR pipeline (Wav2Vec2)
asr = pipeline('automatic-speech-recognition', model='facebook/wav2vec2-base-960h')

# Load Vibravox sample
dataset = load_dataset('Cnam-LMSSC/vibravox', 'speechless_clean')
sample = dataset['train'][0]
audio = sample['audio.headset_microphone']['array']

# Run ASR inference
result = asr(audio, sampling_rate=sample['audio.headset_microphone']['sampling_rate'])
print('Transcription:', result['text'])
