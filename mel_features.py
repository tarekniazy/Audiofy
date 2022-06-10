
import numpy as np

from scipy.signal import get_window

from scipy.fftpack import dct as sci_dct

from utils import *


def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)



def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    
    
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = mel_to_freq(mels)
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs



def get_filters(filter_points, FFT_size,mel_filter_num):
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2)+1))
    
    for j in range(0,mel_filter_num):
        for i in range(int(filter_points[j]), int(filter_points[j+1])):
            filters[j,i] = (i - filter_points[j]) / (filter_points[j+1]-filter_points[j])
        for i in range(int(filter_points[j+1]), int(filter_points[j+2])):
            filters[j,i] = (filter_points[j+2]-i) / (filter_points[j+2]-filter_points[j+1])
    
    return filters






def mfcc(sig,sr):

    frames,_=framing(sig=sig)

    window = get_window("hann", 512, fftbins=True)

    audio_windowed=frames*window

    audio_fft=stft_basic(audio_windowed)


    audio_powered=(1/512)*(np.abs(audio_fft)**2)


    energy = np.sum(audio_powered,1) # this stores the total energy in each frame
    energy = np.where(energy == 0,np.finfo(float).eps,energy) # if energy is zero, we get problems with log

    freq_min = 0
    freq_high = 8000
    mel_filter_num = 20


    filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, 512, sample_rate=16000)



    filters = get_filters(filter_points, 512,20)



    audio_filtered = np.dot(audio_powered,np.transpose(filters))


    zero_handeled=zero_handling(audio_filtered)

    audio_log = np.log(zero_handeled)




    dct_filters = sci_dct(audio_log, type=2, axis=1, norm='ortho')[:,:20]


    dct_filters[:,0] = np.log(energy)

    cepstral_coefficents = dct_filters


    return cepstral_coefficents,audio_filtered,energy


