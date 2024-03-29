from math import ceil
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

import gammatone_features

from sklearn.preprocessing import MinMaxScaler

from feature_extraction import *

from RNN import *

import sys
sys.path.append(os.path.abspath("../../scripts"))
from generate_dataset import *
from audio_reconstruction import *


# differential of mfcc, add 0 to the beginning
def get_diff_list(data):
    L = []
    for d in data:
        L.append(np.concatenate([[d[0]], np.diff(d, axis=-2)], axis=-2))
    return np.array(L)

# we need to reset state in RNN. becasue we dont each batch are different. however, we need statful=true for nnom






def main():
    # load test dataset. Generate by gen_dataset.py see the file for details.
    try:
        dataset = np.load('dataset.npz', allow_pickle=True)
        # dataset2 = np.load('dataset2.npz', allow_pickle=True)
    except:
        raise Exception("dataset.npz not found, please run 'gen_dataset.py' to create dataset")

    # combine them together
    # clnsp_mfcc = dataset['clnsp_mfcc']    # mfcc
    noisy_mfcc = dataset['noisy_mfcc']
    
    vad = dataset['vad']                  # voice active detection
    gains =dataset['gains']


    print("Gaiiiiiiins......",gains.shape)

    # clnsp_gfcc = dataset['clnsp_gfcc']    # mfcc
    noisy_gfcc = dataset['noisy_gfcc']

    # noisy_bfcc = dataset['noisy_bfcc']



    # gains2 = dataset2['gains'] 



    # get mfcc derivative from dataset.


    timestamp_size = 128

    # clnsp_mfcc_diff = get_diff_list(clnsp_mfcc)
    noisy_mfcc_diff = get_diff_list(noisy_mfcc)
    # clnsp_mfcc_diff1 = get_diff_list(clnsp_mfcc_diff)
    noisy_mfcc_diff1 = get_diff_list(noisy_mfcc_diff)

    noisy_gfcc_diff = get_diff_list(noisy_gfcc)
    # clnsp_mfcc_diff1 = get_diff_list(clnsp_mfcc_diff)
    noisy_gfcc_diff1 = get_diff_list(noisy_gfcc_diff)

    # combine all pices to one large array
    # clnsp_mfcc = np.concatenate(clnsp_mfcc, axis=0)
    noisy_mfcc = np.concatenate(noisy_mfcc, axis=0)

    # clnsp_gfcc = np.concatenate(clnsp_gfcc, axis=0)
    noisy_gfcc = np.concatenate(noisy_gfcc, axis=0)

    


    gfcc_padd_len=ceil(noisy_gfcc.shape[0]/timestamp_size)*timestamp_size-noisy_gfcc.shape[0]

    gfcc_padd_value=np.zeros((gfcc_padd_len,noisy_gfcc.shape[1]))

    noisy_gfcc=np.concatenate([noisy_gfcc,gfcc_padd_value])

    
    
    mfcc_padd_len=ceil(noisy_mfcc.shape[0]/timestamp_size)*timestamp_size-noisy_mfcc.shape[0]

    mfcc_padd_value=np.zeros((mfcc_padd_len,noisy_mfcc.shape[1]))

    noisy_mfcc=np.concatenate([noisy_mfcc,mfcc_padd_value])



    # noisy_bfcc = np.concatenate(noisy_bfcc, axis=0)




    # clnsp_mfcc_diff = np.concatenate(clnsp_mfcc_diff, axis=0)
    noisy_mfcc_diff = np.concatenate(noisy_mfcc_diff, axis=0)
    # clnsp_mfcc_diff1 = np.concatenate(clnsp_mfcc_diff1, axis=0)
    noisy_mfcc_diff1 = np.concatenate(noisy_mfcc_diff1, axis=0)

    noisy_mfcc_diff=np.concatenate([noisy_mfcc_diff,mfcc_padd_value])

    noisy_mfcc_diff1=np.concatenate([noisy_mfcc_diff1,mfcc_padd_value])

    




    noisy_gfcc_diff = np.concatenate(noisy_gfcc_diff, axis=0)
    # clnsp_mfcc_diff1 = np.concatenate(clnsp_mfcc_diff1, axis=0)
    noisy_gfcc_diff1 = np.concatenate(noisy_gfcc_diff1, axis=0)

    noisy_gfcc_diff=np.concatenate([noisy_gfcc_diff,gfcc_padd_value])

    noisy_gfcc_diff1=np.concatenate([noisy_gfcc_diff1,gfcc_padd_value])



    vad = np.concatenate(vad, axis=0)
    gains = np.concatenate(gains, axis=0)
    # gains2 = np.concatenate(gains2, axis=0)




 

    # preprocess data
    num_sequence = len(vad) // timestamp_size

    # prepare data
    diff = np.copy(noisy_mfcc_diff[:, :10])
    diff1 = np.copy(noisy_mfcc_diff1[:, :10])

    
    
    
    diff_gfcc = np.copy(noisy_gfcc_diff[:, :])
    diff1_gfcc = np.copy(noisy_gfcc_diff1[:, :])



    feat = np.copy(noisy_mfcc[:, :])

    

    noisy_gfcc=np.copy(noisy_gfcc[:, :])

    # noisy_bfcc=np.copy(noisy_bfcc[:num_sequence * timestamp_size, :])



    # concat mfcc, 1st and 2nd derivative together as the training data.



    # ,noisy_gfcc
    # ,diff_gfcc,diff1_gfcc

    
    
    x_train = np.concatenate([feat, diff, diff1,noisy_gfcc,diff_gfcc,diff1_gfcc], axis=-1)


    # convert MFCC range to -1 to 1.0 In quantization, we will saturate them to leave more resolution in smaller numbers
    # we saturate the peak to leave some more resolution in other band.
    # 




    # processed_x=np.empty((x_train.shape[0]))


    # x_train[x_train==NULL]=0

    # for i in range(x_train.shape[0]):
    #     for j in range(x_train[i].shape[0]):

    #         if x_train[i][j] ==NULL:
    #             x_train[i][j]=0

    # x_train = np.reshape(np.delete(x_train,np.where(x_train==NULL)),(x_train.shape[0],-1))








    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = scaler.fit(x_train)
    # x_train = scaler.transform(x_train)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)


    x_train = normalize(x_train,2, quantize=False)







    

    x_train = np.copy(x_train[:num_sequence * timestamp_size, :])

    x_train = np.reshape(x_train, (num_sequence* timestamp_size, 1, x_train.shape[-1]))
    y_train = np.copy(gains[:num_sequence * timestamp_size,:])
    y_train = np.reshape(y_train, (num_sequence* timestamp_size, gains.shape[-1]))
    vad_train = np.copy(vad[:num_sequence * timestamp_size]).astype(np.float32)
    vad_train = np.reshape(vad_train, (num_sequence * timestamp_size, 1))



    # plt.hist(x_train.flatten(), bins=1000)
    # plt.show()

    # plt.hist(y_train.flatten(), bins=1000)
    # plt.show()

    # plt.hist(vad_train.flatten(), bins=1000)
    # plt.show()



    # train the model, choose either one.
    # history = train(x_train, y_train, vad_train, batch_size=timestamp_size, model_name="model.h5")

    # history = train_gains(x_train, y_train, batch_size=timestamp_size, epochs=5, model_name="model.h5")

    # history = train_simple(x_train, y_train, vad_train, batch_size=timestamp_size, epochs=10, model_name="model.h5")

    # get the model
    model = load_model("model.h5", custom_objects={'mycost': mycost, 'msse':msse, 'my_crossentropy':my_crossentropy, 'my_accuracy':my_accuracy})

    # denoise a file for test.
    # Make sure the MFCC parameters inside the voice_denoise() are the same as our gen_dataset.
    (rate, sig) = wav.read("_noisy_sample.wav")
    



    filtered_sig = voice_denoise(sig, 16000, model,timestamp_size=timestamp_size, numcep=y_train.shape[-1], plot=True) # use plot=True argument to see the gains/vad

    
    print(filtered_sig)
    wav.write("_nn_filtered_sample.wav", rate, np.asarray(filtered_sig * 32768, dtype=np.int16))

    # now generate the NNoM model
    # generate_model(model, x_train[:timestamp_size*4], name='denoise_weights.h')
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # import tensorflow as tf


    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if(physical_devices is not None):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    main()
    print("Done..")

   