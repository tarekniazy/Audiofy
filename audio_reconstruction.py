import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from gammatone_features import *
from mel_features import *
from feature_extraction import *
from iirfilter import *

def normalize(data, n, quantize=True):
    limit = pow(2, n)
    data = np.clip(data, -limit, limit)/limit
    if quantize:
        data = np.round(data * 128)/ 128.0
    return data

def iir_design(band_frequency, samplerate, order=1): # the ban frequency is the middel fre
    b = []
    a = []
    # b2 = []
    # a2 = []
    fre = band_frequency / (samplerate/2)
    fre = np.clip(fre, 0.001, 1)

    for i in range(1, len(band_frequency)-1):
        # b_, a_ = signal.iirfilter(4, [fre[i] - (fre[i]-fre[i-1])/2, fre[i]+ (fre[i+1]-fre[i])/2],
        #                           btype='bandpass', output='ba')

        b_, a_ = iirfilter(4, [fre[i] - (fre[i]-fre[i-1])/2, fre[i]+ (fre[i+1]-fre[i])/2])

        b.append(b_)
        a.append(a_)
        # b2.append(b2_)
        # a2.append(a2_)
    
    # print("COMPARE")
    # print((np.array(a)==np.array(a2)).all())
    # print((np.array(b)==np.array(b2)).all())
    return b, a



def bandpass_filter_iir(sig, b_in, a_in, step, gains):
    from scipy import signal
    x = sig
    y = np.zeros(step*len(gains))
    state = signal.lfilter_zi(b_in, a_in)
    g=0
    for n in range(0, len(gains)):
        g = max(0.6*g, gains[n])    # r=0.6 pre RNNoise paper https://arxiv.org/pdf/1709.08243.pdf
        b = b_in*g
        a = a_in

        filtered, state = signal.lfilter(b, a, x[n*step: min((n+1)*step, step*len(gains))], zi=state)
        y[n*step: min((n+1)*step, step*len(gains))] = filtered

    return y


def filter_voice(sig, rate, gains, nband=26, lowfreq=0, highfreq=8000):
    mel_filter_num = 20
    filter_points, band_freq = get_filter_points(lowfreq, highfreq, mel_filter_num, 512, sample_rate=16000)
    

    gammatone_points = get_gammatone_points(lowfreq, highfreq, 14)
    # band_freq = mel_to_freq(filter_points)

    # print("BAND ",band_freq)

    band_frequency = band_freq[1:-1] # the middle point of each band

    b, a = iir_design(band_freq, rate)

    # print("B dimension is ",np.array(b).shape)
    b_g,a_g=iir_design(gammatone_points,rate)
    # print("B dimension is ",np.array(b_g).shape)

    b = np.concatenate((b,b_g),axis=0)
    a = np.concatenate((a,a_g),axis=0)
    # b.append(b_g)
    # a.append(a_g)
    # print("B dimension is ",np.array(b).shape)
    step = int(0.032 * rate )
    filtered_signal = np.zeros(gains.shape[0]*step)


    padded_sig=np.zeros(gains.shape[0]*step-len(sig))
    sig=np.concatenate([sig,padded_sig])
    for i in range(len(b)):
        # filtered_signal = bandpass_filter_iir(sig, b[i].copy(), a[i].copy(), step, gains[:, i])

        filtered_signal += bandpass_filter_iir(sig, b[i].copy(), a[i].copy(), step, gains[:, i])

    
    filtered_signal =filtered_signal * 0.6
    return filtered_signal

def voice_denoise(sig,rate, model, timestamp_size, numcep=26, plot=False):
    sig = sig / 32768



    mfcc_data,_,_,gfcc_data,_,_=extract_features(sig,rate)
    
    mfcc_feat = mfcc_data.astype('float32')
    gfcc_feat=gfcc_data.astype('float32')



    
   

    # differential of mfcc, add 0 to the beginning

    # num_sequence = len(vad) // timestamp_size



    diff = np.diff(mfcc_feat, axis=0)
    diff = np.concatenate([[mfcc_feat[0]], diff], axis=0)  # first derivative
    diff1 = np.diff(diff, axis=0)
    diff1 = np.concatenate([[diff[0]], diff1], axis=0) # second derivative
    diff = diff[:, :10]
    diff1 = diff1[:, :10]



    diff_gfcc = np.diff(gfcc_feat, axis=0)
    diff_gfcc = np.concatenate([[gfcc_feat[0]], diff_gfcc], axis=0)  # first derivative
    diff1_gfcc = np.diff(diff_gfcc, axis=0)
    diff1_gfcc = np.concatenate([[diff_gfcc[0]], diff1_gfcc], axis=0) # second derivative
    diff_gfcc = diff_gfcc[:, :]
    diff1_gfcc = diff1_gfcc[:, :]

    # concat both differential and original mfcc
    # mfcc_feat, diff, diff1,gfcc_feat
    #,diff_gfcc,diff1_gfcc
    feat = np.concatenate([mfcc_feat, diff, diff1,gfcc_feat,diff_gfcc,diff1_gfcc], axis=-1)


    scaler = StandardScaler()
    scaler.fit(feat)
    feat=scaler.transform(feat)
    # requantise the MFCC (same as training data)
    feat = normalize(feat, 2, quantize=False)
    # plt.hist(feat.flatten(), bins=1000)
    # plt.show()



    feat = np.reshape(feat, (feat.shape[0], 1, feat.shape[1]))
    feat = feat[: feat.shape[0] // timestamp_size * timestamp_size]

    prediction = model.predict(feat, batch_size=timestamp_size)

    if(type(prediction) is list):
        predicted_gains = prediction[0]
        predicted_vad = prediction[1]
    else:
        predicted_gains = prediction
        predicted_vad = None


    # now process the signal.



    filtered_sig = filter_voice(sig, rate=rate, gains=predicted_gains, nband=mfcc_feat.shape[-1])


    # if(plot):
    #     for i in range(10):
    #         plt.plot(predicted_gains[:, i], label='band'+str(i))
    #     if(predicted_vad is not None):
    #         plt.plot(predicted_vad, 'r', label='VAD')
    #     plt.ylabel("Gains")
    #     plt.xlabel("MFCC Sample")
    #     plt.legend()
    #     plt.show()
    return filtered_sig[:len(sig)]
