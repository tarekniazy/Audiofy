import numpy as np

from scipy.signal import get_window


def framing(sig):

    
    # compute frame length and frame step (convert from seconds to samples)
    frame_length = 512
    frame_step =  int(frame_length // 2)
    signal_length = len(sig)
    frames_overlap = frame_length - frame_step


    rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
    pad_signal = np.append(sig, np.array([0] * int(frame_step - rest_samples) * int(rest_samples != 0.)))


    nrows = ((pad_signal.size - int(frame_length)) // int(frame_step)) + 1


    n = pad_signal.strides[0]
    frames= np.lib.stride_tricks.as_strided(pad_signal,
                                        shape=(nrows, int(frame_length)),
                                        strides=(int(frame_step)*n, n))
    return frames, frame_length


def stft_basic(x):

    X=[]

    for i in x:
        X.append(np.fft.fft(i)[:1025])

        # X_win = np.fft.fft(x_win)

    return np.asarray(X)



def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def met_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)



def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    
    
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = met_to_freq(mels)
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs



def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters


def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis


def mfcc_trial(sig,sr):

    frames,_=framing(sig=sig)

    window = get_window("hann", 512, fftbins=True)

    audio_windowed=frames*window

    audio_fft=stft_basic(audio_windowed)


    audio_powered=np.abs(audio_fft)**2

    energy = np.sum(audio_powered,1) # this stores the total energy in each frame
    energy = np.where(energy == 0,np.finfo(float).eps,energy) # if energy is zero, we get problems with log

    freq_min = 0
    freq_high = 8000
    mel_filter_num = 20


    filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, 512, sample_rate=44100)

    filters = get_filters(filter_points, 1024)

    enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
    filters *= enorm[:, np.newaxis]


    audio_filtered = np.dot(filters,np.transpose(audio_powered))




    audio_log = np.log(audio_filtered)

    dct_filter_num = 40

    dct_filters = dct(dct_filter_num, mel_filter_num)

    cepstral_coefficents = np.dot(dct_filters, audio_log)

    return cepstral_coefficents[:20].T,audio_filtered.T,energy




