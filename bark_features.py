
import numpy as np

from utils import *


def intensity_power_law(w):
    """
    Apply the intensity power law.
    Args:
        w (array) : signal information.
    Resturn:
        array after intensity power law.
    """
    def f(w, c, p):
        return w**2 + c * 10**p

    E = (f(w, 56.8, 6) * w**4) / (f(w, 6.3, 6) * f(w, .38, 9) *
                                  f(w**3, 9.58, 26))
    return E**(1 / 3)


def bark_filter(fb, fc):
    """
    Compute a Bark filter around a certain center frequency in bark.
    Args:
        fb (int): frequency in Bark.
        fc (int): center frequency in Bark.
    Returns:
        (float) : associated Bark filter value/amplitude.
    """
    if fc - 2.5 <= fb <= fc - 0.5:
        return 10**(2.5 * (fb - fc + 0.5))
    elif fc - 0.5 < fb < fc + 0.5:
        return 1
    elif fc + 0.5 <= fb <= fc + 1.3:
        return 10**(-2.5 * (fb - fc - 0.5))
    else:
        return 0

def freq_to_bark(f):

    return 6. * np.arcsinh(f / 600.)


def bark_to_freq(fb):

    return 600. * np.sinh(fb / 6.)



def fft_to_bark(fft, fs, nfft):

    return freq_to_bark((fft * fs) / (nfft + 1))


def bark_to_fft(fb, fs, nfft):

    return (nfft + 1) * bark_to_freq(fb) / fs


def bark_filter_banks(nfilts=20,
                      nfft=512,
                      fs=16000,
                      low_freq=0,
                      high_freq=None,
                      scale="constant"):

    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0



    low_bark = freq_to_bark(low_freq)
    high_bark = freq_to_bark(high_freq)
    bark_points = np.linspace(low_bark, high_bark, nfilts + 4)

    bins = np.floor(bark_to_fft(bark_points, fs, nfft))
    fbank = np.zeros([nfilts, nfft // 2 + 1])

    if scale == "descendant" or scale == "constant":
        c = 1
    else:
        c = 0

    for j in range(2, nfilts + 2):
        if scale == "descendant":
            c -= 1 / nfilts
            c = c * (c > 0) + 0 * (c < 0)

        elif scale == "ascendant":
            c += 1 / nfilts
            c = c * (c < 1) + 1 * (c > 1)

        for i in range(int(bins[j - 2]), int(bins[j + 2])):
            fc = bark_points[j]
            fb = fft_to_bark(i, fs, nfft)
            fbank[j - 2, i] = c * bark_filter(fb, fc)
    return np.abs(fbank)



def bfcc(sig,rate):

    frames,_=framing(sig=sig)

    window = get_window("hann", 512, fftbins=True)

    audio_windowed=frames*window

    audio_fft=stft_basic(audio_windowed)


    audio_powered=(1/512)*(np.abs(audio_fft)**2)



    freq_min = 0
    freq_high = 8000


    bark_fbanks_mat = bark_filter_banks(nfilts=26,
                                                  nfft=512,
                                                  fs=rate,
                                                  low_freq=freq_min,
                                                  high_freq=freq_high,
                                                  )


    features = np.dot(audio_powered, bark_fbanks_mat.T)

    ipl_features = intensity_power_law(w=features)

    bfccs = sci_dct(ipl_features, type=2, axis=1, norm='ortho')[:,:12]


    return bfccs