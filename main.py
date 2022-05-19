import numpy as np
from tensorflow.keras.models import load_model, save_model
import scipy.io.wavfile as wav

# differential of mfcc, add 0 to the beginning
def get_diff_list(data):
    L = []
    for d in data:
        L.append(np.concatenate([[d[0]], np.diff(d, axis=-2)], axis=-2))
    return np.array(L)

def normalize(data, n, quantize=True):
    limit = pow(2, n)
    data = np.clip(data, -limit, limit)/limit
    if quantize:
        data = np.round(data * 128)/ 128.0
    return data

def main():
    # load test dataset. Generate by gen_dataset.py see the file for details.
    try:
        dataset = np.load('dataset2.npz', allow_pickle=True)
    except:
        raise Exception("dataset.npz not found, please run 'gen_dataset.py' to create dataset")

    
    clnsp_mfcc = dataset['clnsp_mfcc']    # mfcc
    noisy_mfcc = dataset['noisy_mfcc']
    vad = dataset['vad']                  # voice active detection
    gains =dataset['gains']

    ### 
    clnsp_mfcc_diff = get_diff_list(clnsp_mfcc)
    noisy_mfcc_diff = get_diff_list(noisy_mfcc)
    clnsp_mfcc_diff1 = get_diff_list(clnsp_mfcc_diff)
    noisy_mfcc_diff1 = get_diff_list(noisy_mfcc_diff)
    #######
    clnsp_mfcc = np.concatenate(clnsp_mfcc, axis=0)
    noisy_mfcc = np.concatenate(noisy_mfcc, axis=0)

    clnsp_mfcc_diff = np.concatenate(clnsp_mfcc_diff, axis=0)
    noisy_mfcc_diff = np.concatenate(noisy_mfcc_diff, axis=0)

    clnsp_mfcc_diff1 = np.concatenate(clnsp_mfcc_diff1, axis=0)
    noisy_mfcc_diff1 = np.concatenate(noisy_mfcc_diff1, axis=0)
    ########
    vad = np.concatenate(vad, axis=0)
    gains = np.concatenate(gains, axis=0)

    timestamp_size = 100
    num_sequence = len(vad) // timestamp_size

    diff = np.copy(noisy_mfcc_diff[:num_sequence * timestamp_size, :10])
    diff1 = np.copy(noisy_mfcc_diff1[:num_sequence * timestamp_size, :10])
    feat = np.copy(noisy_mfcc[:num_sequence * timestamp_size, :])


    x_train = np.concatenate([feat, diff, diff1], axis=-1)
    x_train = normalize(x_train, 3, quantize=False)

    x_train = np.copy(x_train[:num_sequence * timestamp_size, :])
    x_train = np.reshape(x_train, (num_sequence* timestamp_size, 1, x_train.shape[-1]))
    y_train = np.copy(gains[:num_sequence * timestamp_size,:])
    y_train = np.reshape(y_train, (num_sequence* timestamp_size, gains.shape[-1]))
    vad_train = np.copy(vad[:num_sequence * timestamp_size]).astype(np.float32)
    vad_train = np.reshape(vad_train, (num_sequence * timestamp_size, 1))


    #### TRAIN MODEL
    #history = train(x_train, y_train, vad_train, batch_size=timestamp_size, epochs=5, model_name="model.h5")

    # get the model
    model = load_model("model.h5", custom_objects={'mycost': mycost, 'msse':msse, 'my_crossentropy':my_crossentropy, 'my_accuracy':my_accuracy})

    # denoise a file for test.
    # Make sure the MFCC parameters inside the voice_denoise() are the same as our gen_dataset.
    (rate, sig) = wav.read("_noisy_sample.wav")
    filtered_sig = voice_denoise(sig, rate, model, vad,timestamp_size, numcep=y_train.shape[-1], plot=True) # use plot=True argument to see the gains/vad
    print("done Denoising")
    wav.write("_nn_filtered_sample.wav", rate, np.asarray(filtered_sig * 32768, dtype=np.int16))

    # now generate the NNoM model
    # generate_model(model, x_train[:timestamp_size*4], name='denoise_weights.h')
    return