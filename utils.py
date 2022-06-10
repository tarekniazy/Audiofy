import numpy as np

from scipy.signal import get_window

from scipy.fftpack import dct as sci_dct



def framing(sig):
    """
    This function takes the signal as an input and frames it
    """

    
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

    X=np.fft.rfft(x,512)

    return np.asarray(X)


def zero_handling(x):
    """
    handle the issue with zero values if they are exposed to become an argument
    for any log function.
    Args:
        x (array): input vector.
    Returns:
        vector with zeros substituted with epsilon values.
    """
    return np.where(x == 0, np.finfo(float).eps, x)