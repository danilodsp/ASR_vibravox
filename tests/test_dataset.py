from datasets import load_dataset

def test_vibravox_loads():
    dataset = load_dataset('Cnam-LMSSC/vibravox', 'speechless_clean')
    assert 'train' in dataset
    assert len(dataset['train']) > 0
