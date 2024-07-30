import librosa
import numpy as np
import python_speech_features as psf
from pydub import AudioSegment
from io import BytesIO
import os

def convert_to_wav(file):
    audio = AudioSegment.from_file(file, format="webm")
    wav_io = BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io

def get_fbanks(audio_file):
    
    def normalize_frames(signal, epsilon=1e-12):
        print('in normalize frames fct')
        return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in signal])
    print('in get fbanks fct')
    y, sr = librosa.load(audio_file, sr=16000)
    assert sr == 16000

    trim_len = int(0.25 * sr)
    if y.shape[0] < 1 * sr:
        print("y.shape[0] < 1 * sr")
        # if less than 1 seconds, don't use that audio
        return None
    print("trimming")
    y = y[trim_len:-trim_len]

    # frame width of 25 ms with a stride of 15 ms. This will have an overlap of 10s
    print("doing psf.fbank")
    filter_banks, energies = psf.fbank(y, samplerate=sr, nfilt=64, winlen=0.025, winstep=0.01)
    print("normalizing")
    filter_banks = normalize_frames(signal=filter_banks)
    print("reshaping")
    filter_banks = filter_banks.reshape((filter_banks.shape[0], 64, 1))
    return filter_banks


def extract_fbanks(path):
    print("in extarct fbanks fct")
    fbanks = get_fbanks(path)
    num_frames = fbanks.shape[0]
    print("num_frames:",num_frames)
    # sample sets of 64 frames each

    numpy_arrays = []
    start = 0
    while start < num_frames + 64:
        slice_ = fbanks[start:start + 64]
        if slice_ is not None and slice_.shape[0] == 64:
            assert slice_.shape[0] == 64
            assert slice_.shape[1] == 64
            assert slice_.shape[2] == 1

            slice_ = np.moveaxis(slice_, 2, 0)
            slice_ = slice_.reshape((1, 1, 64, 64))
            numpy_arrays.append(slice_)
        start = start + 64

    print('num samples extracted: {}'.format(len(numpy_arrays)))
    return np.concatenate(numpy_arrays, axis=0)



