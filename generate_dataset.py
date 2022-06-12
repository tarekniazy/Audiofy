import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import os
import re



from feature_extraction import *


def plot_frequency_respond(b, a=None, fs=16000):
    a = a if len(a) == len(b)  else np.ones(len(b))
    for i in range(len(b)):
        w, h = signal.freqz(b[i], a[i])
        plt.plot(w*0.15915494327*fs, 20 * np.log10(np.maximum(abs(h), 1e-5)), 'b')
    plt.title('Digital filter frequency response')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [Hz]')
    plt.show()



def generate_data(path,random_volume=True, vad_active_delay=0.07, vad_threshold=1e-1,winstep=0.032/2):

    mfcc_data = []
    gfcc_data=[]
    # bfcc_data=[]
    filename_label = []
    total_energy = []
    band_energy = []
    gfcc_energy=[]
    vad = []
    files = os.listdir(path)
    for f in files:
        filename = f



        if ('wav' not in filename):
            continue

        (rate, sig) = wav.read(path+'/'+f,)
       





        # sig,rate = librosa.load(path+'/'+f)
        # convert file to [-1, 1)

        sig=np.nan_to_num(sig)
        # sig[np.where(sig==NULL)]=0
        # if (NULL in sig):
            # sig=np.delete(sig,np.where(sig==NULL))



     
        



        sig = sig/32768  


       
        if random_volume:
            sig = sig * np.random.uniform(0.8, 1)


        mfcc_feat,band_eng, total_eng,gfcc_feat,_,gfcc_band_energy=extract_features(sig,rate)





        

        # voice active detections, only valid with clean speech. Detected by total energy vs threshold.
        v = (total_eng > vad_threshold).astype(int)
        vad_delay = int(vad_active_delay*(16000*winstep))
        conv_win = np.concatenate([np.zeros(vad_delay), np.ones(vad_delay)]) # delay the VAD for a vad_active_delay second
        v = np.convolve(v, conv_win, mode='same')
        v = (v > 0).astype(int)

        total_energy.append(total_eng)
        band_energy.append(band_eng)
        gfcc_energy.append(gfcc_band_energy)
        vad.append(v)
        mfcc_data.append(mfcc_feat.astype('float32'))
        gfcc_data.append(gfcc_feat.astype('float32'))
        # bfcc_data.append(bfcc_feat.astype('float32'))




        filename_label.append(filename)
    return mfcc_data, filename_label, total_energy, vad, band_energy,gfcc_data,gfcc_energy#,bfcc_data

    


if __name__ == "__main__":
  

    vad_energy_threashold = 0.1

    noisy_speech_dir = 'MS-SNSD/NoisySpeech_training'
    clean_speech_dir = 'MS-SNSD/CleanSpeech_training'
    noise_dir = 'MS-SNSD/Noise_training'

    # clean sound, mfcc, and vad

    clean_speech_mfcc, clean_file_label, total_energy, vad, clnsp_band_energy,clean_speech_gfcc,clean_gfcc_energy = \
        generate_data(clean_speech_dir, vad_threshold=vad_energy_threashold)

    # add noise to clean speech, then generate the noise MFCC

    noisy_speech_mfcc, noisy_file_label, _, _ , noisy_band_energy,noisy_speech_gfcc,noisy_gfcc_energy= \
        generate_data(noisy_speech_dir, vad_threshold=vad_energy_threashold)

    # MFCC for noise only

    noise_only_mfcc, noise_only_label, _, _ , noise_band_energy,noise_only_gfcc,_= \
        generate_data(noise_dir, random_volume=False)

    # plt.plot(vad[5], label='voice active')
    # plt.plot(total_energy[5], label='energy')
    # plt.legend()
    # plt.show()

    # combine them together
    clnsp_mfcc = []
    noisy_mfcc = []
    noise_mfcc = []
    
    clnsp_gfcc = []
    noisy_gfcc = []
    noise_gfcc = []

    # clnsp_bfcc = []
    # noisy_bfcc = []
    # noise_bfcc = []

    voice_active = []
    gains_array = []





    for idx_nosiy, label in enumerate(noisy_file_label):

        
        # get file encode from file name e.g. "noisy614_SNRdb_30.0_clnsp614.wav"
        nums = re.findall(r'\d+', label)
        file_code = nums[0]
        db_code = nums[1]

        # get clean sound name
        idx_clnsp = clean_file_label.index('clnsp'+str(file_code)+'.wav')

        # truth gains y_train
        gains = np.sqrt(clnsp_band_energy[idx_clnsp]/ noisy_band_energy[idx_nosiy])
        #gains = clnsp_band_energy[idx_clnsp] / noisy_band_energy[idx_nosiy]
        gains = np.clip(gains, 0, 1)

        
        gfcc_gains = np.sqrt(clean_gfcc_energy[idx_clnsp]/ noisy_gfcc_energy[idx_nosiy])
        #gains = clnsp_band_energy[idx_clnsp] / noisy_band_energy[idx_nosiy]
        gfcc_gains = np.clip(gfcc_gains, 0, 1)

    

        voice_active.append(vad[idx_clnsp])
        
        
        clnsp_mfcc.append(clean_speech_mfcc[idx_clnsp])
        noisy_mfcc.append(noisy_speech_mfcc[idx_nosiy])
        noise_mfcc.append(noise_only_mfcc[idx_nosiy]) 


        clnsp_gfcc.append(clean_speech_gfcc[idx_clnsp])
        noisy_gfcc.append(noisy_speech_gfcc[idx_nosiy])
        noise_gfcc.append(noise_only_gfcc[idx_nosiy]) 

        
        # clnsp_bfcc.append(clean_speech_bfcc[idx_clnsp])
        # noisy_bfcc.append(noisy_speech_bfcc[idx_nosiy])
        # noise_bfcc.append(noise_only_bfcc[idx_nosiy]) 



        gains=np.concatenate((gains,gfcc_gains),axis=1)
        
        gains_array.append(gains)

        # print(gains.shape)

        # print(gfcc_gains.shape)

        # gains_array.append(gfcc_gains)

        

        #>>> Uncomment to plot the MFCC image
        # mfcc_feat1 = np.swapaxes(clean_speech_mfcc[idx_clnsp], 0, 1)
        # mfcc_feat2 = np.swapaxes(noisy_speech_mfcc[idx_nosiy], 0, 1)
        # fig, ax = plt.subplots(2)
        # ax[0].set_title('MFCC Audio:' + str(idx_clnsp))
        # ax[0].imshow(mfcc_feat1, origin='lower', aspect='auto', vmin=-8, vmax=8)
        # ax[1].imshow(mfcc_feat2, origin='lower', aspect='auto', vmin=-8, vmax=8)
        # plt.show()

    # save the dataset.

    # print(len(gains_array))

    np.savez("dataset.npz", clnsp_mfcc=clnsp_mfcc, noisy_mfcc=noisy_mfcc, noise_mfcc=noise_mfcc, vad=voice_active, gains=gains_array,clnsp_gfcc=clnsp_gfcc,noise_gfcc=noise_gfcc,noisy_gfcc=noisy_gfcc)
    print("Dataset generation has been saved to:", "dataset.npz")